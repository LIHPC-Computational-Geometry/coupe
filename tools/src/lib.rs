use anyhow::Context as _;
use anyhow::Result;
use coupe::Partition as _;
use coupe::PointND;
use mesh_io::medit::Mesh;
use mesh_io::weight;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::any;
use std::mem;

#[cfg(feature = "metis")]
mod metis;
#[cfg(feature = "scotch")]
mod scotch;

pub struct Problem<const D: usize> {
    pub points: Vec<PointND<D>>,
    pub weights: weight::Array,
    pub adjacency: sprs::CsMat<f64>,
}

pub type Metadata = Option<Box<dyn std::fmt::Debug>>;

pub type Runner<'a> = Box<dyn FnMut(&mut [usize]) -> Result<Metadata> + Send + Sync + 'a>;

fn runner_error(message: &'static str) -> Runner {
    Box::new(move |_partition| Err(anyhow::anyhow!("{}", message)))
}

pub trait ToRunner<const D: usize> {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a>;
}

impl<const D: usize, R> ToRunner<D> for coupe::Random<R>
where
    R: 'static + rand::Rng + Send + Sync,
{
    fn to_runner<'a>(&'a mut self, _: &'a Problem<D>) -> Runner<'a> {
        Box::new(move |partition| {
            self.partition(partition, ())?;
            Ok(None)
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::Greedy {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        Box::new(move |partition| {
            match &problem.weights {
                Integers(is) => {
                    let weights = is.iter().map(|weight| weight[0]);
                    self.partition(partition, weights)?;
                }
                Floats(fs) => {
                    let weights = fs.iter().map(|weight| coupe::Real::from(weight[0]));
                    self.partition(partition, weights)?;
                }
            }
            Ok(None)
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::KarmarkarKarp {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        Box::new(move |partition| {
            match &problem.weights {
                Integers(is) => {
                    let weights = is.iter().map(|weight| weight[0]);
                    self.partition(partition, weights)?;
                }
                Floats(fs) => {
                    let weights = fs.iter().map(|weight| coupe::Real::from(weight[0]));
                    self.partition(partition, weights)?;
                }
            }
            Ok(None)
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::CompleteKarmarkarKarp {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        Box::new(move |partition| {
            match &problem.weights {
                Integers(is) => {
                    let weights = is.iter().map(|weight| weight[0]);
                    self.partition(partition, weights)?;
                }
                Floats(fs) => {
                    let weights = fs.iter().map(|weight| coupe::Real::from(weight[0]));
                    self.partition(partition, weights)?;
                }
            }
            Ok(None)
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::VnBest {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        Box::new(move |partition| {
            let algo_iterations = match &problem.weights {
                Integers(is) => {
                    let weights = is.iter().map(|weight| weight[0]);
                    self.partition(partition, weights)
                }
                Floats(fs) => {
                    let weights = fs.iter().map(|weight| coupe::Real::from(weight[0]));
                    self.partition(partition, weights)
                }
            }?;
            Ok(Some(Box::new(algo_iterations)))
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::VnFirst {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        match &problem.weights {
            Integers(is) => {
                let weights: Vec<_> = is.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    let algo_iterations = self.partition(partition, &weights)?;
                    Ok(Some(Box::new(algo_iterations)))
                })
            }
            Floats(fs) => {
                let weights: Vec<_> = fs.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    let algo_iterations = self.partition(partition, &weights)?;
                    Ok(Some(Box::new(algo_iterations)))
                })
            }
        }
    }
}

impl<const D: usize> ToRunner<D> for coupe::Rcb {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        Box::new(move |partition| {
            let points = problem.points.par_iter().cloned();
            match &problem.weights {
                Integers(is) => {
                    let weights = is.par_iter().map(|weight| weight[0]);
                    self.partition(partition, (points, weights))?;
                }
                Floats(fs) => {
                    let weights = fs.par_iter().map(|weight| weight[0]);
                    self.partition(partition, (points, weights))?;
                }
            }
            Ok(None)
        })
    }
}

impl<const D: usize> ToRunner<D> for coupe::HilbertCurve {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        if D != 2 {
            return runner_error("hilbert is only implemented for 2D meshes");
        }
        // SAFETY: is a noop since D == 2
        let points =
            unsafe { mem::transmute::<&Vec<PointND<D>>, &Vec<PointND<2>>>(&problem.points) };
        match &problem.weights {
            Integers(_) => runner_error("hilbert is only implemented for floats"),
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    self.partition(partition, (points, &weights))?;
                    Ok(None)
                })
            }
        }
    }
}

impl<const D: usize> ToRunner<D> for coupe::FiducciaMattheyses {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        let adjacency = {
            let shape = problem.adjacency.shape();
            let (indptr, indices, f64_data) = problem.adjacency.view().into_raw_storage();
            let i64_data = f64_data.iter().map(|f| *f as i64).collect();
            sprs::CsMat::new(shape, indptr.to_vec(), indices.to_vec(), i64_data)
        };
        match &problem.weights {
            Integers(is) => {
                let weights: Vec<i64> = is.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    let metadata = self.partition(partition, (adjacency.view(), &weights))?;
                    Ok(Some(Box::new(metadata)))
                })
            }
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    let metadata = self.partition(partition, (adjacency.view(), &weights))?;
                    Ok(Some(Box::new(metadata)))
                })
            }
        }
    }
}

impl<const D: usize> ToRunner<D> for coupe::KernighanLin {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        let adjacency = problem.adjacency.view();
        match &problem.weights {
            Integers(_) => runner_error("kl is only implemented for floats"),
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    self.partition(partition, (adjacency, &weights))?;
                    Ok(None)
                })
            }
        }
    }
}

pub fn parse_algorithm<const D: usize>(spec: &str) -> Result<Box<dyn ToRunner<D>>> {
    let mut args = spec.split(',');
    let name = args.next().context("it's empty")?;

    fn optional<T>(maybe_arg: Option<Result<T>>, default: T) -> Result<T> {
        Ok(maybe_arg.transpose()?.unwrap_or(default))
    }

    fn require<T>(maybe_arg: Option<Result<T>>) -> Result<T> {
        maybe_arg.context("not enough arguments")?
    }

    fn parse<T>(arg: Option<&str>) -> Option<Result<T>>
    where
        T: std::str::FromStr + any::Any,
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        arg.map(|arg| {
            let f = arg.parse::<T>().with_context(|| {
                format!("arg {:?} is not a valid {}", arg, any::type_name::<T>())
            })?;
            Ok(f)
        })
    }

    Ok(match name {
        "random" => {
            use rand::SeedableRng as _;

            let part_count = require(parse(args.next()))?;
            let seed: [u8; 32] = {
                let mut bytes = args.next().unwrap_or("").as_bytes().to_vec();
                bytes.resize(32_usize, 0_u8);
                bytes.try_into().unwrap()
            };
            let rng = rand_pcg::Pcg64::from_seed(seed);
            Box::new(coupe::Random { rng, part_count })
        }
        "greedy" => Box::new(coupe::Greedy {
            part_count: require(parse(args.next()))?,
        }),
        "kk" => Box::new(coupe::KarmarkarKarp {
            part_count: require(parse(args.next()))?,
        }),
        "ckk" => Box::new(coupe::CompleteKarmarkarKarp {
            tolerance: require(parse(args.next()))?,
        }),
        "vn-best" => Box::new(coupe::VnBest {
            part_count: require(parse(args.next()))?,
        }),
        "vn-first" => Box::new(coupe::VnFirst {
            part_count: require(parse(args.next()))?,
        }),
        "rcb" => Box::new(coupe::Rcb {
            iter_count: require(parse(args.next()))?,
            ..Default::default()
        }),
        "hilbert" => Box::new(coupe::HilbertCurve {
            part_count: require(parse(args.next()))?,
            order: optional(parse(args.next()), 12)?,
        }),
        "fm" => {
            let max_imbalance = parse(args.next()).transpose()?;
            let max_bad_move_in_a_row = optional(parse(args.next()), 0)?;
            let mut max_passes = parse(args.next()).transpose()?;
            if max_passes == Some(0) {
                max_passes = None;
            }
            let mut max_flips_per_pass = parse(args.next()).transpose()?;
            if max_flips_per_pass == Some(0) {
                max_flips_per_pass = None;
            }
            Box::new(coupe::FiducciaMattheyses {
                max_imbalance,
                max_bad_move_in_a_row,
                max_passes,
                max_flips_per_pass,
            })
        }
        "kl" => Box::new(coupe::KernighanLin {
            max_bad_move_in_a_row: optional(parse(args.next()), 1)?,
            ..Default::default()
        }),

        #[cfg(feature = "metis")]
        "metis:recursive" => Box::new(metis::Recursive {
            part_count: require(parse(args.next()))?,
        }),

        #[cfg(feature = "metis")]
        "metis:kway" => Box::new(metis::KWay {
            part_count: require(parse(args.next()))?,
        }),

        #[cfg(feature = "scotch")]
        "scotch:std" => Box::new(scotch::Standard {
            part_count: require(parse(args.next()))?,
        }),

        _ => anyhow::bail!("unknown algorithm {:?}", name),
    })
}

pub fn barycentres<const D: usize>(mesh: &Mesh) -> Vec<PointND<D>> {
    mesh.elements()
        .filter_map(|(element_type, nodes, _element_ref)| {
            if element_type.dimension() != mesh.dimension() {
                return None;
            }
            let mut barycentre = [0.0; D];
            for node_idx in nodes {
                let node_coordinates = mesh.node(*node_idx);
                for (bc_coord, node_coord) in barycentre.iter_mut().zip(node_coordinates) {
                    *bc_coord += node_coord;
                }
            }
            for bc_coord in &mut barycentre {
                *bc_coord /= nodes.len() as f64;
            }
            Some(PointND::from(barycentre))
        })
        .collect()
}

/// The adjacency matrix that models the dual graph of the given mesh.
pub fn dual(mesh: &Mesh) -> sprs::CsMat<f64> {
    let dimension = mesh.dimension();

    let elements = || {
        mesh.elements()
            .filter(|(el_type, _nodes, _ref)| el_type.dimension() == dimension)
            .map(|(_el_type, nodes, _ref)| nodes)
            .enumerate()
    };

    let element_to_nodes = {
        // To speed up node lookup, we store topology information in a more
        // compact array of element chunks.  Chunks store the nodes of elements
        // of the same type, and their start offset.
        struct ElementChunk<'a> {
            start_idx: usize,
            node_per_element: usize,
            nodes: &'a [usize],
        }
        let topology: Vec<ElementChunk> = mesh
            .topology()
            .iter()
            .filter(|(el_type, _nodes, _refs)| el_type.dimension() == dimension)
            .scan(0, |start_idx, (el_type, nodes, _refs)| {
                let item = ElementChunk {
                    start_idx: *start_idx,
                    node_per_element: el_type.node_count(),
                    nodes,
                };
                *start_idx += nodes.len() / item.node_per_element;
                Some(item)
            })
            .collect();
        move |e: usize| -> &[usize] {
            for item in &topology {
                let e = (e - item.start_idx) * item.node_per_element;
                if e < item.nodes.len() {
                    return &item.nodes[e..e + item.node_per_element];
                }
            }
            unreachable!();
        }
    };

    let mut node_to_elements = vec![Vec::new(); mesh.node_count()];
    for (e, nodes) in elements() {
        for node in nodes {
            let node_elements = &mut node_to_elements[*node];
            if node_elements.is_empty() {
                node_elements.reserve(8);
            }
            if let Err(idx) = node_elements.binary_search(&e) {
                node_elements.insert(idx, e);
            }
        }
    }

    let mut adjacency = sprs::CsMat::empty(sprs::CSR, mesh.element_count());
    adjacency.reserve_nnz(mesh.element_count());

    for (e1, e1_nodes) in elements() {
        let neighbors = e1_nodes
            .iter()
            .flat_map(|node| &node_to_elements[*node])
            .cloned();
        for e2 in neighbors {
            if e1 == e2 {
                continue;
            }
            let e2_nodes = element_to_nodes(e2);
            let nodes_in_common = e1_nodes
                .iter()
                .filter(|e1_node| e2_nodes.contains(e1_node))
                .count();
            let are_neighbors = dimension <= nodes_in_common;
            if are_neighbors {
                adjacency.insert(e1, e2, 1.0);
            }
        }
    }

    adjacency
}
