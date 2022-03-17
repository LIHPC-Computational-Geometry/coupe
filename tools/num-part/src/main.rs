use anyhow::Context as _;
use anyhow::Result;
use rand::SeedableRng as _;
use std::env;
use std::path::PathBuf;
use std::process;

mod database;

pub fn parse_distribution<'a, R>(s: &str, rng: R) -> Result<Box<dyn Iterator<Item = f64> + 'a>>
where
    R: rand::Rng + 'a,
{
    use rand_distr::Distribution as _;

    let mut code = s.split(',');
    let name = code.next().context("empty definition")?;
    let mut args = code.zip(1..).map(|(part, i)| {
        let arg = part
            .parse::<f64>()
            .with_context(|| format!("arg #{} is invalid", i))?;
        if arg.is_finite() {
            anyhow::bail!("arg #{} is not finite", i);
        }
        Ok(arg)
    });
    let require = |maybe_arg: Option<Result<f64>>| maybe_arg.context("not enough arguments")?;
    Ok(match name {
        "uniform" => {
            let low = require(args.next())?;
            let high = require(args.next())?;
            Box::new(rand_distr::Uniform::new(low, high).sample_iter(rng))
        }
        "normal" => {
            let mean = require(args.next())?;
            let std_dev = require(args.next())?;
            Box::new(rand_distr::Normal::new(mean, std_dev)?.sample_iter(rng))
        }
        "exp" => {
            let lambda = require(args.next())?;
            Box::new(rand_distr::Exp::new(lambda)?.sample_iter(rng))
        }
        "pareto" => {
            let scale = require(args.next())?;
            let shape = require(args.next())?;
            Box::new(rand_distr::Pareto::new(scale, shape)?.sample_iter(rng))
        }
        "beta" => {
            let alpha = require(args.next())?;
            let beta = require(args.next())?;
            let scale = args.next().transpose()?.unwrap_or(1.0);
            Box::new(
                rand_distr::Beta::new(alpha, beta)?
                    .sample_iter(rng)
                    .map(move |n| n * scale),
            )
        }
        _ => anyhow::bail!("unknown distribution {:?}", name),
    })
}

/// A general and simple type for all implemented algorithms.
///
/// A partitioning algorithm is represented as a function that takes, in order:
/// - the partition to update,
/// - a list of weights,
/// - the number of parts in the partition,
/// - the number of criteria (TODO, won't work).
///
/// The function returns either:
/// - `Ok(())`, when the partition has been updated successfully,
/// - `Err(msg)`, where `msg` is the reason the algorithm cannot be run with the given arguments
///   (e.g. it's a bi-partitioning algorithm but the caller passed a number of parts that is
///   different than 2).
type Algorithm = Box<dyn Fn(&mut [usize], &[f64], usize, usize) -> Result<()>>;

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt(
        "a",
        "algorithm",
        "name of the algorithm to run, see ALGORITHMS",
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
    options.optopt("k", "parts", "number of parts", "INTEGER");
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
    options.optflagopt(
        "",
        "dry-run",
        "print the input of some or all iterations instead of runing the algorithm",
        "ITERATION",
    );

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: num-part [options]"));
        eprint!(include_str!("help_after.txt"));
        process::exit(1);
    }

    let algorithm_description = matches
        .opt_str("a")
        .context("missing required option 'algorithm'")?;
    let _algorithm_chain: Vec<Algorithm> = algorithm_description
        .split(',')
        .map(|_algorithm_name| {
            // TODO
            todo!()
        })
        .collect();

    let seed: [u8; 32] = {
        let mut bytes = matches.opt_str("s").unwrap_or_default().into_bytes();
        bytes.resize(32_usize, 0_u8);
        bytes.try_into().unwrap()
    };
    let _rng = rand_pcg::Pcg64::from_seed(seed);
    // TODO generate sample_iter
    //  -> one distribution per element
    //  -> cannot just make a function to parse distributions into distribution objects
    //  -> maybe have a shared rng? with RefCell

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
    let _criterion_count: usize = matches.opt_count("d");
    let part_count: usize = matches.opt_get("k")?.unwrap_or(2);
    if part_count == 0 {
        anyhow::bail!("-k, --parts  must be greater than zero");
    }

    let db_path = match matches.opt_str("o") {
        Some(s) => PathBuf::from(s),
        None => database::default_path()?,
    };
    let mut db = rusqlite::Connection::open(db_path).context("failed to open the database")?;
    database::upgrade_schema(&mut db).context("failed to upgrade the database schema")?;

    Ok(())
}
