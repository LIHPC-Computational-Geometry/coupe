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

struct Problem<const D: usize> {
    points: Vec<PointND<D>>,
    weights: weight::Array,
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
        "rcb" => Box::new(coupe::Rcb {
            iter_count: require(parse(args.next()))?,
        }),
        "hilbert" => Box::new(coupe::HilbertCurve {
            part_count: require(parse(args.next()))?,
            order: optional(parse(args.next()), 12)?,
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

    let mut partition = vec![0; points.len()];
    let problem = Problem { points, weights };

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
    options.optopt("w", "weights", "weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: mesh-part [options]"));
        eprint!(include_str!("help_after.txt"));
        return Ok(());
    }

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
