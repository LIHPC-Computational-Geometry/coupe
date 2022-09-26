use anyhow::Context as _;
use anyhow::Result;
use coupe::num_traits::AsPrimitive;
use coupe::num_traits::FromPrimitive;
use coupe::num_traits::ToPrimitive;
use coupe::num_traits::Zero;
use coupe::sprs::CsMatView;
use coupe_tools::set_edge_weights;
use coupe_tools::EdgeWeightDistribution;
use mesh_io::Mesh;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs;
use std::io;
use std::iter::Sum;
use std::ops::Mul;

const USAGE: &str = "Usage: part-info [options]";

struct CriterionStats<T> {
    total_weight: T,
    imbalance: f64,
    max_part_weight: T,
    max_part_weight_count: usize,
    min_part_weight: T,
    min_part_weight_count: usize,
}

impl<T> fmt::Display for CriterionStats<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, " - Total weight: {:12.2}", self.total_weight)?;
        writeln!(f, " - Imbalance:    {:12.2}%", self.imbalance * 100.0)?;
        let count_str = |count: usize| {
            if count <= 1 {
                String::new()
            } else {
                format!(" ({} duplicates)", count)
            }
        };
        writeln!(
            f,
            " - Max part:     {:12.2}{}",
            self.max_part_weight,
            count_str(self.max_part_weight_count),
        )?;
        writeln!(
            f,
            " - Min part:     {:12.2}{}",
            self.min_part_weight,
            count_str(self.min_part_weight_count),
        )?;
        Ok(())
    }
}

fn weight_stats<T>(
    part_count: usize,
    part_ids: &[usize],
    weights: &[Vec<T>],
) -> Vec<Box<dyn fmt::Display + Send>>
where
    T: Copy + fmt::Display + Send + Sync + 'static,
    T: FromPrimitive + ToPrimitive + Zero + PartialOrd,
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
            let (min_part_weight, min_part_weight_count, max_part_weight, max_part_weight_count) =
                (0..part_count)
                    .into_par_iter()
                    .map(|part| {
                        let part_weight: T = part_ids
                            .par_iter()
                            .zip(weights)
                            .filter(|(p, _w)| **p == part)
                            .map(|(_p, w)| w[criterion])
                            .sum();
                        (part_weight, 1, part_weight, 1)
                    })
                    .reduce_with(
                        |(min1, min_count1, max1, max_count1),
                         (min2, min_count2, max2, max_count2)| {
                            let min;
                            let min_count;
                            let max;
                            let max_count;
                            if min1 < min2 {
                                min = min1;
                                min_count = min_count1;
                            } else if min2 < min1 {
                                min = min2;
                                min_count = min_count2;
                            } else {
                                min = min1;
                                min_count = min_count1 + min_count2;
                            }
                            if max1 < max2 {
                                max = max2;
                                max_count = max_count2;
                            } else if max2 < max1 {
                                max = max1;
                                max_count = max_count1;
                            } else {
                                max = max1;
                                max_count = max_count1 + max_count2;
                            }
                            (min, min_count, max, max_count)
                        },
                    )
                    .unwrap();

            let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
            let imbalance = if ideal_part_weight != 0.0 {
                (max_part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight
            } else {
                0.0
            };
            // Floating-point sums are not accurate enough and in some cases
            // the sum of the weights of the largest part ends up smaller
            // than the total weights divided by the number of parts.
            let imbalance = f64::max(0.0, imbalance);

            let res: Box<dyn fmt::Display + Send> = Box::new(CriterionStats {
                total_weight,
                imbalance,
                max_part_weight,
                max_part_weight_count,
                min_part_weight,
                min_part_weight_count,
            });

            res
        })
        .collect()
}

fn empty_part_count(part_ids: &[usize], part_count: usize) -> usize {
    part_count
        - part_ids
            .par_iter()
            .fold(HashSet::<usize>::default, |mut uniq, part_id| {
                uniq.insert(*part_id);
                uniq
            })
            .reduce(HashSet::<usize>::default, |uniq1, uniq2| {
                HashSet::union(&uniq1, &uniq2).cloned().collect()
            })
            .len()
}

/// Wrapper around coupe's [coupe::topology::lambda_cut] that applies the edge
/// weight distribution and sums the criterions.
fn lambda_cut<T>(
    adjacency: CsMatView<f64>,
    parts: &[usize],
    edge_weights: EdgeWeightDistribution,
    weights: &[Vec<T>],
) -> T
where
    T: Sum + Mul<Output = T> + AsPrimitive<f64> + Send + Sync + FromPrimitive,
{
    let weights = weights.par_iter().map(|weight| match edge_weights {
        EdgeWeightDistribution::Uniform => T::from_usize(weight.len()).unwrap(),
        EdgeWeightDistribution::Linear => weight.iter().cloned().sum(),
        EdgeWeightDistribution::Sqrt => {
            let sqrt_sum: f64 = weight
                .iter()
                .map(|criterion| f64::sqrt(criterion.as_()))
                .sum();
            T::from_f64(sqrt_sum).unwrap()
        }
    });
    coupe::topology::lambda_cut(adjacency, parts, weights)
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt(
        "E",
        "edge-weights",
        "Change how edge weights are set",
        "VARIANT",
    );
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

    let edge_weights = matches
        .opt_get("E")
        .context("invalid value for -E, --edge-weights")?
        .unwrap_or(coupe_tools::EdgeWeightDistribution::Uniform);

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

    let weights = mesh_io::weight::read(weight_file).context("failed to read weight file")?;

    let (adjacency, parts) = rayon::join(
        || -> Result<_> {
            let mesh = Mesh::from_reader(mesh_file).context("failed to read mesh file")?;
            let mut adjacency = coupe_tools::dual(&mesh);
            if edge_weights != EdgeWeightDistribution::Uniform {
                set_edge_weights(&mut adjacency, &weights, edge_weights);
            }
            Ok(adjacency)
        },
        || -> Result<_> {
            let parts = mesh_io::partition::read(partition_file)
                .context("failed to read partition file")?;

            let part_count = matches
                .opt_get("n")?
                .unwrap_or_else(|| 1 + *parts.iter().max().unwrap_or(&0));

            println!("Parts: {part_count}");
            println!("Empty parts: {}", empty_part_count(&parts, part_count));

            let stats = match &weights {
                mesh_io::weight::Array::Integers(v) => weight_stats(part_count, &parts, v),
                mesh_io::weight::Array::Floats(v) => weight_stats(part_count, &parts, v),
            };
            for (i, stat) in stats.into_iter().enumerate() {
                print!("Criterion #{i}:\n{stat}");
            }
            Ok(parts)
        },
    );
    let adjacency = adjacency?;
    let parts = parts?;

    println!(
        "Edge cut size: {}",
        coupe::topology::edge_cut(adjacency.view(), &parts),
    );
    let lambda_cut: Box<dyn fmt::Display> = match &weights {
        mesh_io::weight::Array::Integers(v) => {
            Box::new(lambda_cut(adjacency.view(), &parts, edge_weights, v))
        }
        mesh_io::weight::Array::Floats(v) => {
            Box::new(lambda_cut(adjacency.view(), &parts, edge_weights, v))
        }
    };
    println!("Lambda cut size: {}", lambda_cut);

    Ok(())
}
