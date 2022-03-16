use anyhow::Context as _;
use anyhow::Result;
use coupe::RunInfo;
use mesh_io::weight;
use std::env;
use std::fs;
use std::io;

struct Problem {
    dimension: usize,
    points: Vec<f64>,
    weights: weight::Array,
}

type Algorithm = Box<dyn FnMut(&mut [usize], &Problem) -> Result<RunInfo>>;

fn parse_algorithm(spec: String) -> Result<Algorithm> {
    let mut args = spec.split(',');
    let name = args.next().context("empty algorithm spec")?;

    fn optional<T>(maybe_arg: Option<Result<T>>, default: T) -> Result<T> {
        Ok(maybe_arg.transpose()?.unwrap_or(default))
    }

    fn required<T>(maybe_arg: Option<Result<T>>) -> Result<T> {
        maybe_arg.context("not enough arguments")?
    }

    fn usize_arg(arg: Option<&str>) -> Option<Result<usize>> {
        arg.map(|arg| {
            let f = arg
                .parse::<usize>()
                .with_context(|| format!("arg {:?} is not a valid positive integer", arg))?;
            Ok(f)
        })
    }

    Ok(match name {
        "random" => {
            use rand::SeedableRng as _;

            let part_count = required(usize_arg(args.next()))?;
            let seed: [u8; 32] = {
                let mut bytes = args.next().unwrap_or("").as_bytes().to_vec();
                bytes.resize(32_usize, 0_u8);
                bytes.try_into().unwrap()
            };
            let mut rng = rand_pcg::Pcg64::from_seed(seed);
            Box::new(move |partition, problem| {
                let algo = coupe::Random::new(&mut rng, part_count);
                let weights = vec![0.0; problem.points.len()];
                let res = match problem.dimension {
                    2 => coupe::Partitioner::<2>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    3 => coupe::Partitioner::<3>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    _ => anyhow::bail!("random is only wired up for 2D and 3D"),
                };
                partition.copy_from_slice(&res);
                Ok(RunInfo::default())
            })
        }
        "greedy" => {
            let part_count = required(usize_arg(args.next()))?;
            Box::new(move |partition, problem| {
                let algo = coupe::Greedy::new(part_count);
                let weights: Vec<f64> = match &problem.weights {
                    weight::Array::Floats(ws) => ws.iter().map(|weight| weight[0]).collect(),
                    weight::Array::Integers(_) => panic!("greedy not wired up for integers"),
                };
                let res = match problem.dimension {
                    2 => coupe::Partitioner::<2>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    3 => coupe::Partitioner::<3>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    _ => anyhow::bail!("greedy is only wired up for 2D and 3D"),
                };
                partition.copy_from_slice(&res);
                Ok(RunInfo::default())
            })
        }
        "kk" => {
            let part_count = required(usize_arg(args.next()))?;
            Box::new(move |partition, problem| {
                let algo = coupe::KarmarkarKarp::new(part_count);
                let weights: Vec<f64> = match &problem.weights {
                    weight::Array::Floats(ws) => ws.iter().map(|weight| weight[0]).collect(),
                    weight::Array::Integers(_) => panic!("kk not wired up for integers"),
                };
                let res = match problem.dimension {
                    2 => coupe::Partitioner::<2>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    3 => coupe::Partitioner::<3>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    _ => anyhow::bail!("kk is only wired up for 2D and 3D"),
                };
                partition.copy_from_slice(&res);
                Ok(RunInfo::default())
            })
        }
        "rcb" => {
            let iter_count = required(usize_arg(args.next()))?;
            Box::new(move |partition, problem| {
                let algo = coupe::Rcb::new(iter_count);
                let weights: Vec<f64> = match &problem.weights {
                    weight::Array::Floats(ws) => ws.iter().map(|weight| weight[0]).collect(),
                    weight::Array::Integers(_) => panic!("rcb not implemented for integers"),
                };
                let res = match problem.dimension {
                    2 => coupe::Partitioner::<2>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    3 => coupe::Partitioner::<3>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    _ => anyhow::bail!("rcb is only wired up for 2D and 3D"),
                };
                partition.copy_from_slice(&res);
                Ok(RunInfo::default())
            })
        }
        "hilbert" => {
            let num_partitions = required(usize_arg(args.next()))?;
            let order = optional(usize_arg(args.next()), 12)? as u32;
            Box::new(move |partition, problem| {
                let algo = coupe::HilbertCurve {
                    num_partitions,
                    order,
                };
                let weights: Vec<f64> = match &problem.weights {
                    weight::Array::Floats(ws) => ws.iter().map(|weight| weight[0]).collect(),
                    weight::Array::Integers(_) => anyhow::bail!("rcb not implemented for integers"),
                };
                let res = match problem.dimension {
                    2 => coupe::Partitioner::<2>::partition(
                        &algo,
                        problem.points.as_slice(),
                        &weights,
                    )
                    .into_ids(),
                    _ => anyhow::bail!("hilbert is only implemented in 2D"),
                };
                partition.copy_from_slice(&res);
                Ok(RunInfo::default())
            })
        }
        _ => anyhow::bail!("unknown algorithm {:?}", name),
    })
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

    let algorithms: Vec<_> = matches
        .opt_strs("a")
        .into_iter()
        .map(parse_algorithm)
        .collect::<Result<_>>()?;

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mesh = mesh_io::medit::Mesh::from_file(mesh_file).context("failed to read mesh file")?;

    let points: Vec<_> = mesh
        .elements()
        .filter_map(|(element_type, nodes, _element_ref)| {
            if element_type.dimension() != mesh.dimension() {
                return None;
            }
            let mut barycentre = vec![0.0; mesh.dimension()];
            for node_idx in nodes {
                let node_coordinates = mesh.node(*node_idx);
                for (bc_coord, node_coord) in barycentre.iter_mut().zip(node_coordinates) {
                    *bc_coord += node_coord;
                }
            }
            for bc_coord in &mut barycentre {
                *bc_coord = *bc_coord / nodes.len() as f64;
            }
            Some(barycentre)
        })
        .flatten()
        .collect();

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weights'")?;
    let weight_file = fs::File::open(weight_file).context("failed to open weight file")?;
    let weight_file = io::BufReader::new(weight_file);
    let weights = weight::read(weight_file).context("failed to read weight file")?;

    let problem = Problem {
        dimension: mesh.dimension(),
        points,
        weights,
    };
    let mut partition = vec![0; problem.points.len() / problem.dimension];

    for mut algorithm in algorithms {
        algorithm(&mut partition, &problem).context("failed to apply algorithm")?;
    }

    let stdout = io::stdout();
    let stdout = stdout.lock();
    let stdout = io::BufWriter::new(stdout);
    mesh_io::partition::write(stdout, partition).context("failed to print partition")?;

    Ok(())
}
