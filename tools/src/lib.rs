use anyhow::Context as _;
use anyhow::Result;
use coupe::Partition as _;
use coupe::PointND;
use mesh_io::medit::Mesh;
use mesh_io::weight;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::slice::ParallelSlice as _;
use std::any;
use std::io;
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
                    let weights = fs.iter().map(|weight| weight[0]);
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
            tolerance: optional(parse(args.next()), 0.05)?,
            ..Default::default()
        }),
        "fastrcb" => Box::new(coupe::Rcb {
            iter_count: require(parse(args.next()))?,
            tolerance: optional(parse(args.next()), 0.05)?,
            fast: true,
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
            let mut max_moves_per_pass = parse(args.next()).transpose()?;
            if max_moves_per_pass == Some(0) {
                max_moves_per_pass = None;
            }
            Box::new(coupe::FiducciaMattheyses {
                max_imbalance,
                max_bad_move_in_a_row,
                max_passes,
                max_moves_per_pass,
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
    let element_to_nodes = |e: usize| -> &[usize] {
        for item in &topology {
            let e = (e - item.start_idx) * item.node_per_element;
            if e < item.nodes.len() {
                return &item.nodes[e..e + item.node_per_element];
            }
        }
        unreachable!();
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

    let el_count: usize = topology
        .iter()
        .map(|chunk| chunk.nodes.len() / chunk.node_per_element)
        .sum();
    let indice_locks = vec![Vec::new(); el_count];
    topology.par_iter().for_each(|chunk| {
        let end_idx = chunk.start_idx + chunk.nodes.len() / chunk.node_per_element;
        chunk
            .nodes
            .par_chunks_exact(chunk.node_per_element)
            .zip(chunk.start_idx..end_idx)
            .for_each(|(e1_nodes, e1)| {
                let mut neighbors: Vec<usize> = e1_nodes
                    .iter()
                    .flat_map(|node| &node_to_elements[*node])
                    .cloned()
                    .filter(|e2| {
                        e1 != *e2 && {
                            let e2_nodes = element_to_nodes(*e2);
                            let nodes_in_common = e1_nodes
                                .iter()
                                .filter(|e1_node| e2_nodes.contains(e1_node))
                                .count();
                            dimension <= nodes_in_common
                        }
                    })
                    .collect();
                neighbors.sort_unstable();
                neighbors.dedup();
                let ptr = &indice_locks[e1] as *const Vec<usize> as *mut Vec<usize>;
                unsafe { ptr.write(neighbors) }
            })
    });

    let mut indptr: Vec<usize> = Some(0)
        .into_par_iter()
        .chain(indice_locks.par_iter().map(|neighbors| neighbors.len()))
        .collect();
    for i in 1..indptr.len() {
        indptr[i] += indptr[i - 1];
    }

    let size = indptr.len() - 1;
    let indices = vec![0; indptr[indptr.len() - 1]];
    indptr
        .par_iter()
        .zip(&indptr[1..])
        .zip(indice_locks)
        .for_each(|((start, end), neighbors)| {
            let src = neighbors.as_ptr();
            let dst = indices[*start..*end].as_ptr() as *mut usize;
            unsafe { std::ptr::copy_nonoverlapping(src, dst, end - start) }
        });

    let data = vec![1.0; indices.len()];

    sprs::CsMat::new((size, size), indptr, indices, data)
}

#[derive(Copy, Clone, PartialEq)]
pub enum EdgeWeightDistribution {
    Uniform,
    Linear,
    Sqrt,
}

#[derive(Debug)]
pub struct EdgeWeightDistError;

impl std::fmt::Display for EdgeWeightDistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expected 'uniform', 'linear' or 'sqrt'")
    }
}
impl std::error::Error for EdgeWeightDistError {}

impl std::str::FromStr for EdgeWeightDistribution {
    type Err = EdgeWeightDistError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match &*s.to_ascii_lowercase() {
            "uniform" => EdgeWeightDistribution::Uniform,
            "linear" => EdgeWeightDistribution::Linear,
            "sqrt" => EdgeWeightDistribution::Sqrt,
            _ => return Err(EdgeWeightDistError),
        })
    }
}

pub fn set_edge_weights(
    adjacency: &mut sprs::CsMat<f64>,
    vertex_weights: &weight::Array,
    distribution: EdgeWeightDistribution,
) {
    let vertex_weights = |vertex: usize| match vertex_weights {
        weight::Array::Integers(is) => is[vertex][0] as f64,
        weight::Array::Floats(fs) => fs[vertex][0],
    };
    for (node, mut neighbors) in adjacency.outer_iterator_mut().enumerate() {
        for (neighbor, edge_weight) in neighbors.iter_mut() {
            let node_weight = vertex_weights(node);
            let neighbor_weight = vertex_weights(neighbor);
            *edge_weight = match distribution {
                EdgeWeightDistribution::Uniform => 1.0,
                EdgeWeightDistribution::Linear => node_weight + neighbor_weight,
                EdgeWeightDistribution::Sqrt => node_weight.sqrt() + neighbor_weight.sqrt(),
            };
        }
    }
}

pub enum MeshFormat {
    MeditAscii,
    MeditBinary,
}

#[derive(Debug)]
pub struct MeshFormatError;

impl std::fmt::Display for MeshFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expected 'mesh' or 'meshb'")
    }
}
impl std::error::Error for MeshFormatError {}

impl std::str::FromStr for MeshFormat {
    type Err = MeshFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match &*s.to_ascii_lowercase() {
            "mesh" => Self::MeditAscii,
            "meshb" => Self::MeditBinary,
            _ => return Err(MeshFormatError),
        })
    }
}

pub fn write_mesh(mesh: &Mesh, format: MeshFormat) -> Result<()> {
    match format {
        MeshFormat::MeditAscii => println!("{mesh}"),
        MeshFormat::MeditBinary => {
            let stdout = io::stdout();
            let stdout = stdout.lock();
            let stdout = io::BufWriter::new(stdout);
            mesh.write_to(stdout).context("failed to write mesh")?;
        }
    }
    Ok(())
}
