use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::thread_rng;
use std::collections::HashMap;
use std::iter::Chain;

const NB_PARTS: usize = 2;
const NB_CRITERIA: usize = 3;
const NB_WEIGHTS: usize = 100;
const MIN_WEIGHT: usize = 1;
const MAX_WEIGHT: usize = 1000;

type CWeight = [usize; NB_CRITERIA];

pub struct Instance<const NB_CRITERIA: usize> {
    pub weights: Vec<CWeight>,
}
pub struct PartitionMetadata<const NB_CRITERIA: usize> {
    // Weight load for each criterion
    pub parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA],
    // Imbalance for each criterion
    pub parts_imbalances_per_crit: [[usize; NB_PARTS]; NB_CRITERIA],
}

fn eval_partition_metadata(
    partition: Vec<usize>,
    c_weights: [CWeight; NB_WEIGHTS],
) -> [[usize; NB_PARTS]; NB_CRITERIA] {
    let mut res = [[0; NB_PARTS]; NB_CRITERIA];

    for (part, c_weight) in partition.iter().zip(c_weights) {
        for (criterion, weight) in c_weight.iter().enumerate() {
            res[criterion][*part] += *weight
        }
    }

    res
}

fn main() {
    // Instance related data
    let uniform = Uniform::new(MIN_WEIGHT, MAX_WEIGHT);
    let c_weights: [CWeight; NB_WEIGHTS] =
        core::array::from_fn(|_| core::array::from_fn(|_| uniform.sample(&mut thread_rng())));

    let mut sum_c_weights: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    for c_weight in c_weights.iter() {
        for (refsum, val) in sum_c_weights.iter_mut().zip(c_weight) {
            *refsum += val;
        }
    }
    let mut parts_target_load: [[usize; NB_PARTS]; NB_CRITERIA] = [[0; NB_PARTS]; NB_CRITERIA];
    for (criterion, sum) in sum_c_weights.iter().enumerate() {
        if sum % 2 == 0 {
            parts_target_load[criterion] = [sum / 2, sum / 2];
        } else {
            parts_target_load[criterion] = [usize::from(sum / 2), usize::from(sum / 2) + 1];
        }
    }

    let mut partition: [usize; NB_WEIGHTS] = [0; NB_WEIGHTS];

    // for sum in sum_c_weights.iter() {
    // let mut parts_target_load: [[usize; NB_PARTS]; NB_CRITERIA] = sum_c_weights
    //     .iter()
    //     .map(|&sum| match sum % 2 {
    //         0 => core::array::from_fn(|_| [sum / 2, sum / 2]),
    //         1 => core::array::from_fn(|_| [usize::from(sum / 2), usize::from(sum / 2) + 1]),
    //         _ => todo!(),
    //     })
    //     .collect();

    //  = sum_c_weights
    //     .iter()
    //     .map(|&sum| match sum % 2 {
    //         0 => [sum / 2, sum / 2],
    //         1 => [usize::from(sum / 2), usize::from(sum / 2) + 1],
    //         _ => todo!(),
    //     })
    //     .collect();
    println!("{:?}", parts_target_load);
    // for c_weight in c_weights.iter() {
    //     for
    // }
    // let mut sum_weights = c_weights.map(
    //     |weight| {
    //         weight.iter().fold([0:usize; NB_CRITERIA], new_weight_from(acc.iter().zip(&x).map(|(a, b)| a + b)))
    //     }
    // )
    // new_coord_from(a.iter().zip(&b).map(|(a, b)| a + b))
    // new_coord_from(a.iter().zip(&b).map(|(a, b)| a + b))
    // let max_possible_gain = (0..partition.len())
    //     .map(|vertex| {
    //         adjacency
    //             .neighbors(vertex)
    //             .fold(0, |acc, (_, edge_weight)| acc + edge_weight)
    //     })
    //     .max()
    //     .unwrap();

    // let mut z: Coord = [0, 0, 0];
    // for (i, (aval, bval)) in a.iter().zip(&b).enumerate() {
    //     z[i] = aval + bval;
    // }
    // z
    // for ((iter_load, iter_value), bval) in parts_target_load.iter_mut().zip(&a).zip(&b) {
    //     *zval = aval + bval;
    // }

    // for c_weight in c_weights.iter() {
    //     for (criterion, weight) in c_weight.iter().enumerate() {
    //         res[criterion][*part] += *weight
    //     }
    // }

    // [[7, 8], [7, 8], [7, 8]];
    let parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA] =
        eval_partition_metadata(partition, c_weights);
    println!("{:?}", parts_loads_per_crit);

    // let x = std::env::args().nth(1).unwrap().parse().unwrap();
    // let y = std::env::args().nth(2).unwrap().parse().unwrap();
    // let iter = std::env::args()
    //     .nth(3)
    //     .unwrap_or_else(|| String::from("12"))
    //     .parse()
    //     .unwrap();
    // eprintln!("grid size: ({x},{y}); rcb iters: {iter}");
    // let grid = coupe::Grid::new_2d(x, y);
    // let n = usize::from(x) * usize::from(y);
    // let weights: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // let mut partition = vec![0; n];

    // let domain = ittapi::Domain::new("MyIncredibleDomain");
    // let before = std::time::Instant::now();
    // let task = ittapi::Task::begin(&domain, "MyIncredibleTask");
    // grid.rcb(&mut partition, &weights, iter);
    // std::mem::drop(task);
    // eprintln!("time: {:?}", before.elapsed());

    // let i = usize::from(x);
    // eprint!("partition[{}] = {}\r", i, partition[i]);
}
