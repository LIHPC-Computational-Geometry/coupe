use anyhow::Context;
use anyhow::Result;
use std::env;
use std::fs;
use std::io;
use std::str;

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "name")]
enum Distribution {
    #[serde(rename = "uniform")]
    Uniform { low: f64, high: f64 },

    #[serde(rename = "normal")]
    Normal { mean: f64, std_dev: f64 },

    #[serde(rename = "exp")]
    Exp { lambda: f64 },

    #[serde(rename = "pareto")]
    Pareto { scale: f64, shape: f64 },

    #[serde(rename = "beta")]
    Beta {
        alpha: f64,
        beta: f64,
        scale: Option<f64>,
    },
}

impl Distribution {
    pub fn samples<R>(
        &self,
        rng: R,
        num_cases: usize,
        case_size: usize,
        case_type: CaseType,
    ) -> Box<dyn Iterator<Item = Case>>
    where
        R: rand::Rng + 'static,
    {
        match self {
            DistInfo::Uniform { low, high } => {
                let distribution = rand_distr::Uniform::new(*low, *high);
                Box::new(CaseIter {
                    iter: distribution.sample_iter(rng),
                    num_cases,
                    case_size,
                    case_type,
                })
            }
            DistInfo::Normal { mean, std_dev } => {
                let distribution = rand_distr::Normal::new(*mean, *std_dev).unwrap();
                Box::new(CaseIter {
                    iter: distribution.sample_iter(rng),
                    num_cases,
                    case_size,
                    case_type,
                })
            }
            DistInfo::Exp { lambda } => {
                let distribution = rand_distr::Exp::new(*lambda).unwrap();
                Box::new(CaseIter {
                    iter: distribution.sample_iter(rng),
                    num_cases,
                    case_size,
                    case_type,
                })
            }
            DistInfo::Pareto { scale, shape } => {
                let distribution = rand_distr::Pareto::new(*scale, *shape).unwrap();
                Box::new(CaseIter {
                    iter: distribution.sample_iter(rng),
                    num_cases,
                    case_size,
                    case_type,
                })
            }
            DistInfo::Beta { alpha, beta, scale } => {
                let distribution = rand_distr::Beta::new(*alpha, *beta).unwrap();
                if let Some(scale) = *scale {
                    Box::new(CaseIter {
                        iter: distribution.sample_iter(rng).map(move |n| n * scale),
                        num_cases,
                        case_size,
                        case_type,
                    })
                } else {
                    Box::new(CaseIter {
                        iter: distribution.sample_iter(rng),
                        num_cases,
                        case_size,
                        case_type,
                    })
                }
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DistInfo::Uniform { .. } => "uniform",
            DistInfo::Normal { .. } => "normal",
            DistInfo::Exp { .. } => "exp",
            DistInfo::Pareto { .. } => "pareto",
            DistInfo::Beta { .. } => "beta",
        }
    }

    pub fn param1(&self) -> f64 {
        match self {
            DistInfo::Uniform { low: param1, .. }
            | DistInfo::Normal { mean: param1, .. }
            | DistInfo::Exp { lambda: param1 }
            | DistInfo::Pareto { scale: param1, .. }
            | DistInfo::Beta { alpha: param1, .. } => *param1,
        }
    }

    pub fn param2(&self) -> Option<f64> {
        match self {
            DistInfo::Uniform { high: param2, .. }
            | DistInfo::Normal {
                std_dev: param2, ..
            }
            | DistInfo::Pareto { shape: param2, .. }
            | DistInfo::Beta { beta: param2, .. } => Some(*param2),
            DistInfo::Exp { .. } => None,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "name")]
enum AlgorithmSpec {
    #[serde(rename = "greedy-num")]
    GreedyNum { parts: usize },

    #[serde(rename = "hilbert-curve")]
    HilbertCurve { parts: usize, order: u32 },

    #[serde(rename = "kk")]
    KarmarkarKarp { parts: usize },

    #[serde(rename = "ckk")]
    KarmarkarKarpComplete { tolerance: f64 },

    #[serde(rename = "kmeans")]
    KMeans {
        parts: usize,
        tolerance: f64,
        delta_threshold: f64,
        iterations_max: usize,
        max_balance_iter: usize,
        erode: bool,
        hilbert: bool,
        mbr_early_break: bool,
    },

    #[serde(rename = "multi-jagged")]
    MultiJagged { parts: usize, iterations_max: usize },

    #[serde(rename = "random")]
    Random,

    #[serde(rename = "rcb")]
    Rcb { iterations: usize },

    #[serde(rename = "rib")]
    Rib { iterations: usize },

    #[serde(rename = "vnbest")]
    VNBest { parts: usize },

    #[serde(rename = "zcurve")]
    ZCurve { parts: usize, order: u32 },
}

fn one() -> usize {
    1
}
fn default_path() -> String {
    String::from("results.db")
}

#[derive(Debug, serde::Deserialize)]
struct Input {
    algorithm: Vec<AlgorithmSpec>,
    distribution: Distribution,
    mesh_file: String,

    #[serde(default = "one")]
    num_criteria: usize,

    #[serde(default = "one")]
    num_iterations: usize,

    #[serde(default = "default_path")]
    output: String,

    #[serde(default)]
    seed: String,

    #[serde(default)]
    use_integers: bool,
}

fn main() -> Result<()> {
    use io::Read as _;

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read input")?;
    let input: Input = toml::from_str(&input).context("invalid input format")?;

    println!("{:?}", input);
    // TODO

    Ok(())
}
