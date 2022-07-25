use anyhow::Context as _;
use anyhow::Result;
use coupe::num_traits::FromPrimitive;
use coupe::num_traits::ToPrimitive;
use coupe::num_traits::Zero;
use mesh_io::medit::Mesh;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::env;
use std::fs;
use std::io;

const USAGE: &str = "Usage: part-info [options]";

fn imbalance<T>(part_count: usize, part_ids: &[usize], weights: &[Vec<T>]) -> Vec<f64>
where
    T: Copy + Send + Sync,
    T: FromPrimitive + ToPrimitive + Zero,
    T: std::ops::Div<Output = T> + std::ops::Sub<Output = T> + std::iter::Sum,
{
    let criterion_count = match weights.first() {
        Some(w) => w.len(),
        None => return Vec::new(),
    };
    (0..criterion_count)
        .into_par_iter()
        .map(|criterion| {
            let total_weight: T = weights.par_iter().map(|weight| weight[criterion]).sum();
            let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
            (0..part_count)
                .into_par_iter()
                .map(|part| {
                    let part_weight: T = part_ids
                        .iter()
                        .zip(weights)
                        .filter(|(p, _w)| **p == part)
                        .map(|(_p, w)| w[criterion])
                        .sum();
                    let part_weight = part_weight.to_f64().unwrap();
                    (part_weight - ideal_part_weight) / ideal_part_weight
                })
                .max_by(|part_weight0, part_weight1| {
                    f64::partial_cmp(part_weight0, part_weight1).unwrap()
                })
                .unwrap()
        })
        .collect()
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("m", "mesh", "mesh file", "FILE");
    options.optopt("n", "parts", "number of expected parts", "COUNT");
    options.optopt("p", "partition", "partition file", "FILE");
    options.optopt("w", "weights", "path to a weight file", "FILE");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: part-info [options]"));
        return Ok(());
    }
    if !matches.free.is_empty() {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mesh_file = fs::File::open(mesh_file).context("failed to open mesh file")?;
    let mesh_file = io::BufReader::new(mesh_file);

    let partition_file = matches
        .opt_str("p")
        .context("missing required option 'partition'")?;
    let partition_file = fs::File::open(partition_file).context("failed to open partition file")?;
    let partition_file = io::BufReader::new(partition_file);

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weight'")?;
    let weight_file = fs::File::open(weight_file).context("failed to open weight file")?;
    let weight_file = io::BufReader::new(weight_file);

    let (adjacency, parts) = rayon::join(
        || -> Result<_> {
            let mesh = Mesh::from_reader(mesh_file).context("failed to read mesh file")?;
            let adjacency = coupe_tools::dual(&mesh);
            Ok(adjacency)
        },
        || -> Result<_> {
            let (parts, weights) = rayon::join(
                || {
                    mesh_io::partition::read(partition_file)
                        .context("failed to read partition file")
                },
                || mesh_io::weight::read(weight_file).context("failed to read weight file"),
            );
            let parts = parts?;
            let weights = weights?;

            let part_count = matches
                .opt_get("n")?
                .unwrap_or_else(|| 1 + *parts.iter().max().unwrap_or(&0));

            let imbs = match &weights {
                mesh_io::weight::Array::Integers(v) => imbalance(part_count, &parts, v),
                mesh_io::weight::Array::Floats(v) => imbalance(part_count, &parts, v),
            };
            println!("imbalances: {:?}", imbs);
            Ok(parts)
        },
    );
    let adjacency = adjacency?;
    let parts = parts?;

    println!(
        "edge cut: {}",
        coupe::topology::edge_cut(adjacency.view(), &parts),
    );
    println!(
        "lambda cut: {}",
        coupe::topology::lambda_cut(adjacency.view(), &parts),
    );

    Ok(())
}
