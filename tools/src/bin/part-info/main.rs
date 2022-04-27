use anyhow::Context as _;
use anyhow::Result;
use mesh_io::medit::Mesh;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::env;
use std::fs;
use std::io;

fn imbalance<T>(part_count: usize, part_ids: &[usize], weights: &[Vec<T>]) -> Vec<f64>
where
    T: Copy + Send + Sync,
    T: num::FromPrimitive + num::ToPrimitive + num::Zero,
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

    let mesh_file = matches
        .opt_str("m")
        .context("missing required option 'mesh'")?;
    let mesh = Mesh::from_file(mesh_file).context("failed to read mesh file")?;

    let partition_file = matches
        .opt_str("p")
        .context("missing required option 'partition'")?;
    let partition_file = fs::File::open(partition_file).context("failed to open partition file")?;
    let partition_file = io::BufReader::new(partition_file);
    let parts =
        mesh_io::partition::read(partition_file).context("failed to read partition file")?;

    let weight_file = matches
        .opt_str("w")
        .context("missing required option 'weight'")?;
    let weight_file = fs::File::open(weight_file).context("failed to open weight file")?;
    let weight_file = io::BufReader::new(weight_file);
    let weights = mesh_io::weight::read(weight_file).context("failed to read weight file")?;

    let part_count = matches
        .opt_get("n")?
        .unwrap_or_else(|| 1 + *parts.iter().max().unwrap_or(&0));

    let adjacency = coupe_tools::dual(&mesh);

    let imbs = match &weights {
        mesh_io::weight::Array::Integers(v) => imbalance(part_count, &parts, v),
        mesh_io::weight::Array::Floats(v) => imbalance(part_count, &parts, v),
    };
    println!("imbalances: {:?}", imbs);
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
