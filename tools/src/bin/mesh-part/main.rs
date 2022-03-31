use anyhow::Context as _;
use anyhow::Result;
use coupe::Partition as _;
use coupe::PointND;
use mesh_io::medit::Mesh;
use mesh_io::weight;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::any;
use std::env;
use std::fs;
use std::io;
use std::mem;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::Registry;
use tracing_tree::HierarchicalLayer;

#[cfg(feature = "metis")]
mod metis;
#[cfg(feature = "scotch")]
mod scotch;

struct Problem<const D: usize> {
    points: Vec<PointND<D>>,
    weights: weight::Array,
    adjacency: sprs::CsMat<f64>,
}

trait Algorithm<const D: usize> {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()>;
}

impl<const D: usize, R> Algorithm<D> for coupe::Random<R>
where
    R: rand::Rng,
{
    fn run(&mut self, partition: &mut [usize], _: &Problem<D>) -> Result<()> {
        self.partition(partition, ())?;
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::Greedy {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
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
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::KarmarkarKarp {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
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
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::CompleteKarmarkarKarp {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
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
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::VnBest {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
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
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::VnFirst {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
        match &problem.weights {
            Integers(is) => {
                let weights: Vec<_> = is.iter().map(|weight| weight[0]).collect();
                self.partition(partition, &weights)?;
            }
            Floats(fs) => {
                let weights: Vec<_> = fs.iter().map(|weight| weight[0]).collect();
                self.partition(partition, &weights)?;
            }
        }
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::Rcb {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
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
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::HilbertCurve {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
        if D != 2 {
            anyhow::bail!("hilbert is only implemented for 2D meshes");
        }
        // SAFETY: is a noop since D == 2
        let points =
            unsafe { mem::transmute::<&Vec<PointND<D>>, &Vec<PointND<2>>>(&problem.points) };
        match &problem.weights {
            Integers(_) => anyhow::bail!("hilbert is only implemented for floats"),
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                self.partition(partition, (points, weights))?;
            }
        }
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::FiducciaMattheyses {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
        let adjacency = problem.adjacency.view();
        match &problem.weights {
            Integers(is) => {
                let weights: Vec<i64> = is.iter().map(|weight| weight[0]).collect();
                self.partition(partition, (adjacency, &weights))?;
            }
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                self.partition(partition, (adjacency, &weights))?;
            }
        }
        Ok(())
    }
}

impl<const D: usize> Algorithm<D> for coupe::KernighanLin {
    fn run(&mut self, partition: &mut [usize], problem: &Problem<D>) -> Result<()> {
        use weight::Array::*;
        let adjacency = problem.adjacency.view();
        match &problem.weights {
            Integers(_) => anyhow::bail!("kl is only implemented for floats"),
            Floats(fs) => {
                let weights: Vec<f64> = fs.iter().map(|weight| weight[0]).collect();
                self.partition(partition, (adjacency, &weights))?;
            }
        }
        Ok(())
    }
}

fn parse_algorithm<const D: usize>(spec: &str) -> Result<Box<dyn Algorithm<D>>> {
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
        }),
        "hilbert" => Box::new(coupe::HilbertCurve {
            part_count: require(parse(args.next()))?,
            order: optional(parse(args.next()), 12)?,
        }),
        "fm" => Box::new(coupe::FiducciaMattheyses {
            max_imbalance: Some(optional(parse(args.next()), 0.1)?),
            max_bad_move_in_a_row: optional(parse(args.next()), 0)?,
            max_passes: parse(args.next()).transpose()?,
            max_flips_per_pass: parse(args.next()).transpose()?,
        }),
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

fn main_d<const D: usize>(
    matches: getopts::Matches,
    mesh: Mesh,
    weights: weight::Array,
) -> Result<Vec<usize>> {
    let points: Vec<_> = mesh
        .elements()
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
        .collect();

    let adjacency = coupe_tools::dual(&mesh);

    let mut partition = vec![0; points.len()];
    let problem = Problem {
        points,
        weights,
        adjacency,
    };

    let algorithms: Vec<_> = matches
        .opt_strs("a")
        .into_iter()
        .map(|algorithm_spec| {
            parse_algorithm(&algorithm_spec)
                .with_context(|| format!("invalid algorithm {:?}", algorithm_spec))
        })
        .collect::<Result<_>>()?;

    for mut algorithm in algorithms {
        algorithm
            .run(&mut partition, &problem)
            .context("failed to apply algorithm")?;
    }

    Ok(partition)
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optmulti(
        "a",
        "algorithm",
        "name of the algorithm to run, see ALGORITHMS",
        "NAME",
    );
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("t", "trace", "emit a chrome trace", "FILE");
    options.optopt("w", "weights", "weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: mesh-part [options]"));
        eprint!(include_str!("help_after.txt"));

        #[cfg(feature = "metis")]
        eprint!(include_str!("help_after_metis.txt"));

        #[cfg(feature = "scotch")]
        eprint!(include_str!("help_after_scotch.txt"));

        return Ok(());
    }

    let registry = Registry::default().with(EnvFilter::from_env("LOG")).with(
        HierarchicalLayer::new(4)
            .with_thread_ids(true)
            .with_targets(true)
            .with_bracketed_fields(true),
    );
    let _chrome_trace_guard = match matches.opt_str("t") {
        Some(filename) => {
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(filename)
                .build();
            registry.with(chrome_layer).init();
            Some(guard)
        }
        None => {
            registry.init();
            None
        }
    };

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mesh = Mesh::from_file(mesh_file).context("failed to read mesh file")?;

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weights'")?;
    let weight_file = fs::File::open(weight_file).context("failed to open weight file")?;
    let weight_file = io::BufReader::new(weight_file);
    let weights = weight::read(weight_file).context("failed to read weight file")?;

    let partition = match mesh.dimension() {
        2 => main_d::<2>(matches, mesh, weights)?,
        3 => main_d::<3>(matches, mesh, weights)?,
        n => anyhow::bail!("expected 2D or 3D mesh, got a {n}D mesh"),
    };

    let stdout = io::stdout();
    let stdout = stdout.lock();
    let stdout = io::BufWriter::new(stdout);
    mesh_io::partition::write(stdout, partition).context("failed to print partition")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_adjacency_convert() {
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 15);
        adjacency.reserve_outer_dim(15);
        adjacency.insert(0, 1, 1.0);
        adjacency.insert(0, 5, 1.0);

        adjacency.insert(1, 0, 1.0);
        adjacency.insert(1, 2, 1.0);
        adjacency.insert(1, 6, 1.0);

        adjacency.insert(2, 1, 1.0);
        adjacency.insert(2, 3, 1.0);
        adjacency.insert(2, 7, 1.0);

        adjacency.insert(3, 2, 1.0);
        adjacency.insert(3, 4, 1.0);
        adjacency.insert(3, 8, 1.0);

        adjacency.insert(4, 3, 1.0);
        adjacency.insert(4, 9, 1.0);

        adjacency.insert(5, 0, 1.0);
        adjacency.insert(5, 6, 1.0);
        adjacency.insert(5, 10, 1.0);

        adjacency.insert(6, 1, 1.0);
        adjacency.insert(6, 5, 1.0);
        adjacency.insert(6, 7, 1.0);
        adjacency.insert(6, 11, 1.0);

        adjacency.insert(7, 2, 1.0);
        adjacency.insert(7, 6, 1.0);
        adjacency.insert(7, 8, 1.0);
        adjacency.insert(7, 12, 1.0);

        adjacency.insert(8, 3, 1.0);
        adjacency.insert(8, 7, 1.0);
        adjacency.insert(8, 9, 1.0);
        adjacency.insert(8, 13, 1.0);

        adjacency.insert(9, 4, 1.0);
        adjacency.insert(9, 8, 1.0);
        adjacency.insert(9, 14, 1.0);

        adjacency.insert(10, 5, 1.0);
        adjacency.insert(10, 11, 1.0);

        adjacency.insert(11, 6, 1.0);
        adjacency.insert(11, 10, 1.0);
        adjacency.insert(11, 12, 1.0);

        adjacency.insert(12, 7, 1.0);
        adjacency.insert(12, 11, 1.0);
        adjacency.insert(12, 13, 1.0);

        adjacency.insert(13, 8, 1.0);
        adjacency.insert(13, 12, 1.0);
        adjacency.insert(13, 14, 1.0);

        adjacency.insert(14, 9, 1.0);
        adjacency.insert(14, 13, 1.0);

        let (xadj, adjncy, _) = adjacency.into_raw_storage();

        assert_eq!(
            xadj.as_slice(),
            &[0, 2, 5, 8, 11, 13, 16, 20, 24, 28, 31, 33, 36, 39, 42, 44]
        );
        assert_eq!(
            adjncy.as_slice(),
            &[
                1, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 3, 9, 0, 6, 10, 1, 5, 7, 11, 2, 6, 8, 12, 3, 7, 9,
                13, 4, 8, 14, 5, 11, 6, 10, 12, 7, 11, 13, 8, 12, 14, 9, 13
            ]
        );
    }
}
