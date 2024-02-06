#![allow(clippy::just_underscores_and_digits)]

use anyhow::Context as _;
use anyhow::Result;
use coupe::nalgebra::allocator::Allocator;
use coupe::nalgebra::ArrayStorage;
use coupe::nalgebra::Const;
use coupe::nalgebra::DefaultAllocator;
use coupe::nalgebra::DimDiff;
use coupe::nalgebra::DimSub;
use coupe::nalgebra::ToTypenum;
use coupe::sprs::CsMat;
use coupe::sprs::CsMatView;
use coupe::sprs::CSR;
use coupe::Partition as _;
use coupe::PointND;
use coupe::PositiveInteger;
use coupe::PositiveWeight;
use mesh_io::weight;
use mesh_io::ElementType;
use mesh_io::Mesh;
use once_cell::sync::OnceCell;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::slice::ParallelSlice as _;
use std::any;
use std::fs::File;
use std::io;
use std::mem;

#[cfg(feature = "metis")]
mod metis;
#[cfg(feature = "scotch")]
mod scotch;
#[cfg(any(feature = "metis", feature = "scotch"))]
mod zoom_in;

#[cfg_attr(not(feature = "ittapi"), path = "ittapi_stub.rs")]
pub mod ittapi;

pub struct Problem<const D: usize> {
    mesh: Mesh,
    weights: weight::Array,
    edge_weights: EdgeWeightDistribution,
    points: OnceCell<Vec<PointND<D>>>,
    adjacency: OnceCell<CsMat<f64>>,
}

impl<const D: usize> Problem<D> {
    pub fn new(mesh: Mesh, weights: weight::Array, edge_weights: EdgeWeightDistribution) -> Self {
        Self {
            mesh,
            weights,
            edge_weights,
            points: OnceCell::new(),
            adjacency: OnceCell::new(),
        }
    }

    pub fn without_mesh(weights: weight::Array) -> Self {
        Self {
            mesh: Mesh::default(),
            weights,
            edge_weights: EdgeWeightDistribution::Uniform,
            points: OnceCell::new(),
            adjacency: OnceCell::new(),
        }
    }

    pub fn points(&self) -> &[PointND<D>] {
        self.points.get_or_init(|| barycentres(&self.mesh))
    }

    pub fn adjacency(&self) -> CsMatView<f64> {
        self.adjacency
            .get_or_init(|| {
                let mut adjacency = dual(&self.mesh);
                if self.edge_weights != EdgeWeightDistribution::Uniform {
                    set_edge_weights(&mut adjacency, &self.weights, self.edge_weights);
                }
                adjacency
            })
            .view()
    }

    pub fn weights(&self) -> &weight::Array {
        &self.weights
    }
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

impl<const D: usize, T, W> ToRunner<D> for coupe::TargetorWIP<T, W>
where
    T: PositiveInteger + std::marker::Send + std::marker::Sync,
    W: PositiveWeight + std::marker::Send + std::marker::Sync,
    Vec<W>: FromIterator<i64>,
{
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;

        Box::new(move |partition| {
            // self.partition(partition, problem.weights)?;
            // let weights = &problem.weights.iter().map(|weight| weight);
            // self.partition(partition, weights)?;
            // Ok(None)
            match &problem.weights {
                Integers(is) => {
                    let weights: Vec<Vec<W>> = is
                        .iter()
                        .map(|inner_vec| inner_vec.to_vec())
                        .collect();
                    // let weights = is
                    //     .iter()
                    //     .map(|weight| Vec<W>::from(*weight))
                    //     .collect::<Vec<Vec<W>>>(V)
                    //     .to_vec();
                    self.partition(partition, weights)?;
                } // Floats(fs) => {
                //     //         let weights = fs
                //     //             .iter()
                //     //             .map(|inner_vec| inner_vec.iter().map(|&value| W::from(value)).collect())
                //     //             .collect();
                //     let weights = fs.iter().map(|weight| *weight);
                //     self.partition(partition, weights)?;
                // }
                Floats(_fs) => (),
            }
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
                    let weights = fs.iter().map(|weight| weight[0]);
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
                    let weights = fs.iter().map(|weight| weight[0]);
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
            let points = problem.points().par_iter().cloned();
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
        if D == 2 {
            // SAFETY: is a noop since D == 2
            let points =
                unsafe { mem::transmute::<&[PointND<D>], &[PointND<2>]>(problem.points()) };
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
        } else if D == 3 {
            // SAFETY: is a noop since D == 3
            let points =
                unsafe { mem::transmute::<&[PointND<D>], &[PointND<3>]>(problem.points()) };
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
        } else {
            runner_error("hilbert is only implemented for 2D and 3D meshes")
        }
    }
}

impl<const D: usize> ToRunner<D> for coupe::KMeans
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        match &problem.weights {
            Integers(_) => runner_error("kmeans is only implemented for floats"),
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                Box::new(move |partition| {
                    self.partition(partition, (problem.points(), &weights))?;
                    Ok(None)
                })
            }
        }
    }
}

impl<const D: usize> ToRunner<D> for coupe::ArcSwap {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        let adjacency = {
            let shape = problem.adjacency().shape();
            let (indptr, indices, f64_data) = problem.adjacency().into_raw_storage();
            let i64_data = f64_data.iter().map(|f| *f as i64).collect();
            CsMat::new(shape, indptr.to_vec(), indices.to_vec(), i64_data)
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

impl<const D: usize> ToRunner<D> for coupe::FiducciaMattheyses {
    fn to_runner<'a>(&'a mut self, problem: &'a Problem<D>) -> Runner<'a> {
        use weight::Array::*;
        let adjacency = {
            let shape = problem.adjacency().shape();
            let (indptr, indices, f64_data) = problem.adjacency().into_raw_storage();
            let i64_data = f64_data.iter().map(|f| *f as i64).collect();
            CsMat::new(shape, indptr.to_vec(), indices.to_vec(), i64_data)
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
        let adjacency = problem.adjacency();
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

pub fn parse_algorithm<const D: usize>(spec: &str) -> Result<Box<dyn ToRunner<D>>>
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
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
        "vn-best" => Box::new(coupe::VnBest),
        "vn-first" => Box::new(coupe::VnFirst),
        "rcb" => Box::new(coupe::Rcb {
            iter_count: require(parse(args.next()))?,
            tolerance: optional(parse(args.next()), 0.05)?,
        }),
        "hilbert" => Box::new(coupe::HilbertCurve {
            part_count: require(parse(args.next()))?,
            order: optional(parse(args.next()), 12)?,
        }),
        "kmeans" => Box::<coupe::KMeans>::default(),
        "arcswap" => {
            let max_imbalance = parse(args.next()).transpose()?;
            Box::new(coupe::ArcSwap { max_imbalance })
        }
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

        "targetor" => {
            // let raw_nb_intervals = parse(args.next()).transpose()?;
            // let nb_intervals = vec![2, 3];
            // let parts_target_loads = vec![vec![10, 10], vec![15, 5]];
            let mut wip_nb_intervals: Vec<i64> = Vec::new();
            let mut wip_parts_target_loads: Vec<Vec<i64>> = Vec::new();

            loop {
                // if let Some(tolerance) = tolerance {
                //     if tolerance < 0.001 {
                //         anyhow::bail!("METIS does not support tolerances below 0.001");
                //     }
                // }
                let nb_split: Option<i32> = parse(args.next()).transpose()?;
                if nb_split.is_none() {
                    break;
                }
                let criterion_load_p0 = parse(args.next()).transpose()?;
                let criterion_load_p1 = parse(args.next()).transpose()?;
                if criterion_load_p0.is_none() || criterion_load_p1.is_none() {
                    panic!("expected three arguments per criterion")
                }

                wip_nb_intervals.push(nb_split.unwrap().into());
                wip_parts_target_loads
                    .push(vec![criterion_load_p0.unwrap(), criterion_load_p1.unwrap()]);
            }

            // let nb_intervals = raw_nb_intervals.unwrap().split('-');
            // nb_intervals
            //     .iter_mut()
            //     .map(|num| *num = num.parse::<usize>().unwrap().to_string());

            // let raw_parts_target_loads = parse(args.next()).transpose()?;
            // let parts_target_loads = raw_parts_target_loads.split('-');
            // parts_target_loads
            //     .iter_mut()
            //     .map(|num| *num = num.parse::<usize>().unwrap().to_string());
            // let (_, upper_bound) = args.size_hint();
            // match upper_bound {
            //     Some(ub) => {
            //         println!("{}", ub);
            //         if ub % 2 != 0 {
            //             panic!("expected two argument per criterion, got {} in total", ub)
            //         }
            //     }
            //     None => {
            //         println!("None");
            //     }
            // }

            // let max_imbalance = parse(args.next()).transpose()?;
            // let max_bad_move_in_a_row = optional(parse(args.next()), 0)?;
            // let mut max_passes = parse(args.next()).transpose()?;
            // if max_passes == Some(0) {
            //     max_passes = None;
            // }

            Box::new(
                coupe::TargetorWIP::new(wip_nb_intervals, wip_parts_target_loads), //     coupe::TargetorWIP {
                                                                                   //     nb_intervals,
                                                                                   //     parts_target_loads,
                                                                                   //     None,
                                                                                   // })
            )
        }

        #[cfg(feature = "metis")]
        "metis:recursive" => {
            let part_count = require(parse(args.next()))?;
            let tolerance = parse(args.next()).transpose()?;
            if let Some(tolerance) = tolerance {
                if tolerance < 0.001 {
                    anyhow::bail!("METIS does not support tolerances below 0.001");
                }
            }
            Box::new(metis::Recursive {
                part_count,
                tolerance,
            })
        }

        #[cfg(feature = "metis")]
        "metis:kway" => {
            let part_count = require(parse(args.next()))?;
            let tolerance = parse(args.next()).transpose()?;
            if let Some(tolerance) = tolerance {
                if tolerance < 0.001 {
                    anyhow::bail!("METIS does not support tolerances below 0.001");
                }
            }
            Box::new(metis::KWay {
                part_count,
                tolerance,
            })
        }

        #[cfg(feature = "scotch")]
        "scotch:std" => Box::new(scotch::Standard {
            part_count: require(parse(args.next()))?,
        }),

        _ => anyhow::bail!("unknown algorithm {:?}", name),
    })
}

/// The number of elements that are taken into account for partitioning.
pub fn used_element_count(mesh: &Mesh) -> usize {
    let element_dim = match mesh
        .topology()
        .iter()
        .map(|(el_type, _, _)| el_type.dimension())
        .max()
    {
        Some(v) => v,
        None => return 0,
    };
    mesh.topology()
        .iter()
        .filter(|(element_type, _nodes, _node_refs)| element_type.dimension() == element_dim)
        .map(|(element_type, nodes, _node_refs)| nodes.len() / element_type.node_count())
        .sum()
}

pub fn barycentres<const D: usize>(mesh: &Mesh) -> Vec<PointND<D>> {
    let element_dim = match mesh
        .topology()
        .iter()
        .map(|(el_type, _, _)| el_type.dimension())
        .max()
    {
        Some(v) => v,
        None => return Vec::new(),
    };
    mesh.elements()
        .filter_map(|(element_type, nodes, _element_ref)| {
            if element_type.dimension() != element_dim || element_type == ElementType::Edge {
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
pub fn dual(mesh: &Mesh) -> CsMat<f64> {
    let dimension = match mesh
        .topology()
        .iter()
        .map(|(el_type, _, _)| el_type.dimension())
        .max()
    {
        Some(v) => v,
        None => return CsMat::empty(CSR, 0),
    };
    let ignored_element = |el_type: ElementType| -> bool {
        el_type.dimension() != dimension || el_type == ElementType::Edge
    };

    let elements = || {
        mesh.elements()
            .filter(|(el_type, _nodes, _ref)| !ignored_element(*el_type))
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
        .filter(|(el_type, _nodes, _refs)| !ignored_element(*el_type))
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

    CsMat::new((size, size), indptr, indices, data)
}

#[derive(Copy, Clone, PartialEq, Eq)]
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
    adjacency: &mut CsMat<f64>,
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
    VtkAscii,
    VtkBinary,
}

#[derive(Debug)]
pub struct MeshFormatError;

impl std::fmt::Display for MeshFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expected 'mesh', 'meshb', 'vtk-ascii' or 'vtk-binary'")
    }
}
impl std::error::Error for MeshFormatError {}

impl std::str::FromStr for MeshFormat {
    type Err = MeshFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match &*s.to_ascii_lowercase() {
            "mesh" => Self::MeditAscii,
            "meshb" => Self::MeditBinary,
            "vtk-ascii" => Self::VtkAscii,
            "vtk-binary" => Self::VtkBinary,
            _ => return Err(MeshFormatError),
        })
    }
}

/// Helper to handle help and version options automatically.
///
/// Also returns an error when the number of free arguments is higher than
/// `max_args`.
pub fn parse_args(
    mut options: getopts::Options,
    help: &str,
    max_args: usize,
) -> Result<getopts::Matches> {
    options.optflag("h", "help", "print this help menu");
    options.optflag("", "version", "print version information");

    let matches = options.parse(std::env::args().skip(1))?;

    if matches.opt_present("h") {
        println!("{}", options.usage(help));
        std::process::exit(0);
    }

    if matches.opt_present("version") {
        let arg0 = std::env::args_os().next().unwrap();
        let arg0 = std::path::PathBuf::from(arg0);
        let bin = arg0.file_name().unwrap().to_string_lossy();
        let ver = env!("COUPE_VERSION");

        println!("{bin} version {ver}");
        print!("Features:");

        #[cfg(feature = "intel-perf")]
        print!(" +intel-perf");
        #[cfg(not(feature = "intel-perf"))]
        print!(" -intel-perf");

        #[cfg(feature = "metis")]
        print!(" +metis");
        #[cfg(not(feature = "metis"))]
        print!(" -metis");

        #[cfg(feature = "scotch")]
        print!(" +scotch");
        #[cfg(not(feature = "scotch"))]
        print!(" -scotch");

        println!();
        std::process::exit(0);
    }

    if matches.free.len() > max_args {
        anyhow::bail!(
            "too many arguments, expected at most {max_args}\n\n{}",
            options.usage(help),
        );
    }

    Ok(matches)
}

/// Helper to read a mesh either from stdin or from a file.
pub fn read_mesh(filename: Option<&String>) -> Result<Mesh> {
    Ok(match filename.cloned().as_deref() {
        None | Some("-") => {
            let stdin = io::stdin();
            let stdin = stdin.lock();
            let stdin = io::BufReader::new(stdin);
            Mesh::from_reader(stdin).context("failed to read mesh from stdin")?
        }
        Some(filename) => Mesh::from_file(filename).context("failed to read mesh from file")?,
    })
}

/// Helper function to retrieve the Write implementation matching the given
/// command-line argument.
pub fn writer(filename: Option<&String>) -> Result<impl io::Write> {
    let w: Box<dyn io::Write> = match filename.cloned().as_deref() {
        None | Some("-") => {
            let stdout = io::stdout();
            let stdout = stdout.lock();
            Box::new(stdout)
        }
        Some(filename) => {
            let file = File::create(filename).context("failed to create output file")?;
            Box::new(file)
        }
    };
    Ok(io::BufWriter::new(w))
}

/// Helper to write a mesh, either to stdout or to a file, in the given format.
pub fn write_mesh(
    mesh: &Mesh,
    format: Option<MeshFormat>,
    filename: Option<&String>,
) -> Result<()> {
    use std::io::Write;

    let mut w = writer(filename)?;
    let format = format
        .or_else(|| {
            let path = std::path::Path::new(filename?);
            let extension = path.extension()?;
            let extension = extension.to_str()?;
            if extension.eq_ignore_ascii_case("mesh") {
                Some(MeshFormat::MeditAscii)
            } else if extension.eq_ignore_ascii_case("meshb") {
                Some(MeshFormat::MeditBinary)
            } else if extension.eq_ignore_ascii_case("vtk") {
                Some(MeshFormat::VtkAscii)
            } else {
                None
            }
        })
        .unwrap_or(MeshFormat::MeditBinary);
    match format {
        MeshFormat::MeditAscii => writeln!(w, "{}", mesh.display_medit_ascii())?,
        MeshFormat::MeditBinary => mesh.serialize_medit_binary(w)?,
        MeshFormat::VtkAscii => writeln!(w, "{}", mesh.display_vtk_ascii())?,
        MeshFormat::VtkBinary => mesh.serialize_vtk_binary(w)?,
    }
    Ok(())
}
