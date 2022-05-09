use anyhow::Context as _;
use anyhow::Result;
use coupe::rayon::iter::IntoParallelRefIterator as _;
use coupe::rayon::iter::ParallelIterator as _;
use itertools::Itertools as _;
use mesh_io::weight;
use rand::SeedableRng as _;
use rand_distr::Distribution as _;
use std::env;
use std::process;

mod database;

#[derive(Clone, Copy)]
enum Distribution {
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std_dev: f64 },
    Exp { lambda: f64 },
    Pareto { scale: f64, shape: f64 },
    Beta { alpha: f64, beta: f64, scale: f64 },
}

impl Distribution {
    pub fn parse(s: &str) -> Result<Distribution> {
        let mut code = s.split(',');
        let name = code.next().context("empty definition")?;
        let mut args = code.zip(1..).map(|(part, i)| {
            let arg = part
                .parse::<f64>()
                .with_context(|| format!("arg #{} is invalid", i))?;
            if !arg.is_finite() {
                anyhow::bail!("arg #{} is not finite", i);
            }
            Ok(arg)
        });
        let require = |maybe_arg: Option<Result<f64>>| maybe_arg.context("not enough arguments")?;
        Ok(match name {
            "uniform" => {
                let low = require(args.next())?;
                let high = require(args.next())?;
                Self::Uniform { low, high }
            }
            "normal" => {
                let mean = require(args.next())?;
                let std_dev = require(args.next())?;
                let _ = rand_distr::Normal::new(mean, std_dev)?;
                Self::Normal { mean, std_dev }
            }
            "exp" => {
                let lambda = require(args.next())?;
                let _ = rand_distr::Exp::new(lambda)?;
                Self::Exp { lambda }
            }
            "pareto" => {
                let scale = require(args.next())?;
                let shape = require(args.next())?;
                let _ = rand_distr::Pareto::new(scale, shape)?;
                Self::Pareto { scale, shape }
            }
            "beta" => {
                let alpha = require(args.next())?;
                let beta = require(args.next())?;
                let scale = args.next().transpose()?.unwrap_or(1.0);
                let _ = rand_distr::Beta::new(alpha, beta)?;
                Self::Beta { alpha, beta, scale }
            }
            _ => anyhow::bail!("unknown distribution {:?}", name),
        })
    }

    pub fn samples<'a, R>(self, rng: R) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        R: rand::Rng + 'a,
    {
        match self {
            Self::Uniform { low, high } => {
                Box::new(rand_distr::Uniform::new(low, high).sample_iter(rng))
            }
            Self::Normal { mean, std_dev } => Box::new(
                rand_distr::Normal::new(mean, std_dev)
                    .unwrap()
                    .sample_iter(rng),
            ),
            Self::Exp { lambda } => {
                Box::new(rand_distr::Exp::new(lambda).unwrap().sample_iter(rng))
            }
            Self::Pareto { scale, shape } => Box::new(
                rand_distr::Pareto::new(scale, shape)
                    .unwrap()
                    .sample_iter(rng),
            ),
            Self::Beta { alpha, beta, scale } => Box::new(
                rand_distr::Beta::new(alpha, beta)
                    .unwrap()
                    .sample_iter(rng)
                    .map(move |f| f * scale),
            ),
        }
    }

    pub fn into_params(self) -> [f64; 3] {
        match self {
            Distribution::Uniform { low, high } => [low, high, 0.0],
            Distribution::Normal { mean, std_dev } => [mean, std_dev, 0.0],
            Distribution::Exp { lambda } => [lambda, 0.0, 0.0],
            Distribution::Pareto { scale, shape } => [scale, shape, 0.0],
            Distribution::Beta { alpha, beta, scale } => [alpha, beta, scale],
        }
    }
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optmulti(
        "a",
        "algorithm",
        "name of the algorithm to run, see mesh-part(1)'s ALGORITHMS",
        "NAME",
    );
    options.optmulti(
        "d",
        "distribution",
        "definition of the weight distribution, see DISTRIBUTION",
        "DEFINITION",
    );
    options.optflag(
        "i",
        "integers",
        "generate integers instead of floating-point numbers",
    );
    options.optopt(
        "l",
        "weight-count",
        "number of weights to partition, if no mesh is given",
        "INTEGER",
    );
    options.optopt(
        "n",
        "iterations",
        "number of times the algorithm is run",
        "INTEGER",
    );
    options.optopt("o", "output", "path to the output database", "FILENAME");
    options.optopt(
        "s",
        "seed",
        "32-bit seed for experiment reproduction",
        "BYTES",
    );

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: num-part [options]"));
        eprint!(include_str!("help_after.txt"));
        process::exit(1);
    }

    let algorithm_specs = matches.opt_strs("a");
    if algorithm_specs.is_empty() {
        anyhow::bail!("missing required option 'algorithm'");
    }
    let mut algorithms: Vec<_> = algorithm_specs
        .iter()
        .map(|algorithm_spec| {
            coupe_tools::parse_algorithm(algorithm_spec)
                .with_context(|| format!("invalid algorithm {:?}", algorithm_spec))
        })
        .collect::<Result<_>>()?;

    let seed: [u8; 32] = {
        let mut bytes = matches.opt_str("s").unwrap_or_default().into_bytes();
        bytes.resize(32_usize, 0_u8);
        bytes.try_into().unwrap()
    };
    let mut rng = rand_pcg::Pcg64::from_seed(seed);

    let distribution_spec = matches
        .opt_str("d")
        .context("missing required option 'distribution'")?;
    let distribution = Distribution::parse(&distribution_spec)
        .context("invalid value for option 'distribution'")?;
    let mut samples = distribution.samples(&mut rng);

    let iteration_count: usize = matches.opt_get("n")?.unwrap_or(1);
    if iteration_count == 0 {
        anyhow::bail!("-n, --iterations  must be greater than zero");
    }
    let weight_count: usize = matches
        .opt_get("l")?
        .context("missing required option 'weight-count'")?;
    if weight_count == 0 {
        anyhow::bail!("-l, --weight-count  must be greater than zero");
    }
    let criterion_count: usize = matches.opt_count("d");
    if criterion_count != 1 {
        eprintln!("Warning: multi-criteria runs are not supported yet");
    }
    let use_integers = matches.opt_present("i");

    let db_path = matches
        .opt_str("o")
        .unwrap_or_else(|| "num_part.db".to_string());
    let mut db = database::open(Some(db_path)).context("failed to open the database")?;
    let seed_id = db.insert_seed(&seed).context("failed to save seed")?;
    let distribution_id = db
        .insert_distribution(&distribution_spec, distribution.into_params())
        .context("failed to save distribution")?;

    let mut partition = vec![0; weight_count].into_boxed_slice();
    for iter_no in 1..=iteration_count {
        let problem = if use_integers {
            let chunks = (&mut samples).take(weight_count).chunks(criterion_count);
            let weights = chunks
                .into_iter()
                .map(|chunk| chunk.map(|f| f as i64).collect())
                .collect();
            let weights = weight::Array::Integers(weights);
            coupe_tools::Problem::<0>::without_mesh(weights)
        } else {
            let chunks = (&mut samples).take(weight_count).chunks(criterion_count);
            let weights = chunks.into_iter().map(|chunk| chunk.collect()).collect();
            let weights = weight::Array::Floats(weights);
            coupe_tools::Problem::<0>::without_mesh(weights)
        };
        let mut algo_iterations = None;
        for (algorithm_spec, algorithm) in algorithm_specs.iter().zip(&mut algorithms) {
            let mut algorithm = algorithm.to_runner(&problem);
            match algorithm(&mut partition) {
                Ok(Some(metadata)) => {
                    // TODO rework coupe api again i guess
                    if let Ok(v) = format!("{metadata:?}").parse() {
                        algo_iterations = Some(v);
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    eprintln!("Warning: algorithm {algorithm_spec} failed to run: {err}");
                    break;
                }
            }
        }
        let part_count = *partition.iter().max().unwrap() + 1;
        let imbalance = match problem.weights() {
            weight::Array::Integers(is) => {
                coupe::imbalance::imbalance(part_count, &partition, is.par_iter().map(|i| i[0]))
            }
            weight::Array::Floats(fs) => {
                coupe::imbalance::imbalance(part_count, &partition, fs.par_iter().map(|i| i[0]))
            }
        };
        db.insert_experiment(database::Experiment {
            algorithm: &algorithm_specs.join(";"),
            seed_id,
            distribution_id,
            weight_count,
            iteration: iter_no,
            case_type: use_integers,
            imbalance,
            algo_iterations,
        })
        .context("failed to save experiment")?;
        if iter_no < iteration_count {
            partition.fill(0);
        }
    }

    Ok(())
}
