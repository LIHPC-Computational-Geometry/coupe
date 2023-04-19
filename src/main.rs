use num_traits::Float;
use rand::distributions::Distribution;
// use rand::distributions::Uniform;
use rand::thread_rng;
use rand::SeedableRng as _;
use std::cmp;
use std::collections::HashMap;
use std::iter::Chain;
// use rand_distr::Distribution as _;
// use array_init;
use itertools::iproduct;
use itertools::Itertools;
use itertools::MultiProduct;
use itertools::Product;
use rand::{distributions::Uniform, Rng};
use std::iter::Filter;
use std::iter::Map; // 0.6.5

const NB_PARTS: usize = 2;
const NB_CRITERIA: usize = 3;
const NB_WEIGHTS: usize = 100;
const MIN_WEIGHT: usize = 1;
const MIN_WEIGHTS: [usize; NB_CRITERIA] = [MIN_WEIGHT; NB_CRITERIA];
const MAX_WEIGHT: usize = 1000;
const MAX_WEIGHTS: [usize; NB_CRITERIA] = [MAX_WEIGHT; NB_CRITERIA];
const NB_INTERVALS: usize = 10;

// TODO : See CKK

type CWeight = [usize; NB_CRITERIA];

pub struct Instance<const NB_CRITERIA: usize> {
    pub c_weights: Vec<CWeight>,
}

pub struct PartitionMetadata<const NB_CRITERIA: usize> {
    // Weight load for each criterion
    pub parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA],
    // Imbalance for each criterion
    pub parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA],
}

#[derive(Clone, Copy)]
pub struct TargetorParameters<const NB_CRITERIA: usize> {
    // Number of intervals used to discretize the solution space over each
    // criterion
    pub nb_intervals: [usize; NB_CRITERIA],
    // Deltas used to discretize the solution space over each criterion.
    // Current implementation supposed that delta is constant over each
    // criterion
    pub deltas: [f64; NB_CRITERIA],
    // Target Load for each part over all criterion
    pub parts_target_load: [[usize; NB_PARTS]; NB_CRITERIA],
}

fn eval_parts_loads_per_crit(
    partition: Vec<usize>,
    c_weights: Vec<CWeight>,
) -> [[usize; NB_PARTS]; NB_CRITERIA] {
    let mut res = [[0; NB_PARTS]; NB_CRITERIA];

    for (part, c_weight) in partition.iter().zip(c_weights) {
        for (criterion, weight) in c_weight.iter().enumerate() {
            res[criterion][*part] += *weight
        }
    }

    res
}

fn eval_parts_imbalances_per_crit(
    parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA],
    parts_target_load: [[usize; NB_PARTS]; NB_CRITERIA],
) -> [[isize; NB_PARTS]; NB_CRITERIA] {
    let mut res = [[0; NB_PARTS]; NB_CRITERIA];

    let iter = parts_loads_per_crit.iter().zip(parts_target_load.iter());
    for (criterion, (curr_loads, target_loads)) in iter.enumerate() {
        let mut imbalances = [0; NB_PARTS];
        for (i, (curr_load, target_load)) in curr_loads.iter().zip(target_loads).enumerate() {
            let (conv_curr, conv_target) =
                (isize::try_from(*curr_load), isize::try_from(*target_load));
            match (conv_curr, conv_target) {
                (Ok(conv_curr), Ok(conv_target)) => imbalances[i] = conv_curr - conv_target,
                _ => panic!("type conversion error."),
            }
        }
        res[criterion] = imbalances;
    }

    res
}

fn compute_box_indices(
    c_weight: CWeight,
    parameters: TargetorParameters<NB_CRITERIA>,
) -> [usize; NB_CRITERIA] {
    let mut res: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    for (criterion, weight) in c_weight.iter().enumerate() {
        let diff = (weight - MIN_WEIGHTS[criterion]) as f64;
        let f_index = diff / parameters.deltas[criterion];
        res[criterion] = f_index.floor() as usize;
    }

    res
}

// fn compute_boxes(instance: Instance<NB_CRITERIA>, parameters: TargetorParameters<NB_CRITERIA>) {
//  -> HashMap<[usize; NB_CRITERIA], Vec<usize>>
fn compute_boxes(
    c_weights: Vec<CWeight>,
    parameters: TargetorParameters<NB_CRITERIA>,
) -> HashMap<[usize; NB_CRITERIA], Vec<usize>> {
    let mut res: HashMap<[usize; NB_CRITERIA], Vec<usize>> = HashMap::new();

    // let ins = instance.clone();
    // let c_weights = ins.c_weights;

    for (pos, c_weight) in c_weights.iter().enumerate() {
        let mut box_indices: [usize; NB_CRITERIA] = compute_box_indices(*c_weight, parameters);

        // Handle max bounds
        box_indices
            .iter_mut()
            .enumerate()
            .for_each(|(criterion, index)| {
                if *index > parameters.nb_intervals[criterion] {
                    *index = parameters.nb_intervals[criterion] - 1;
                }
            });

        match res.get_mut(&box_indices) {
            Some(vect_box_indices) => vect_box_indices.push(pos),
            None => {
                let vect_box_indices: Vec<usize> = vec![pos];
                res.insert(box_indices, vect_box_indices);
            }
        }
        // if boxes.contains(box_indices) {
        //     boxes.get(box_indices).push(box_indices);
        // } else {
        //     let vec
        //     boxes.insert(box_indices, )
        // }
        // if box_indices in res:
        //     res[box_indices].append(pos)
        // else:
        //     res[box_indices] = [pos]
    }

    res
}

fn generate_indices(
    origin: [usize; NB_CRITERIA],
    dist: usize,
    parameters: TargetorParameters<NB_CRITERIA>,
    // ) -> impl Iterator<Item = [isize; NB_CRITERIA]> {
) -> impl Iterator<Item = [isize; 3]> {
    // Generator<Yield = [usize; NB_CRITERIA], Return = ()>{
    let mut min_bounds: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    let mut max_bounds: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    for criterion in 0..NB_CRITERIA {
        min_bounds[criterion] = origin[criterion];
        max_bounds[criterion] = parameters.nb_intervals[criterion] - origin[criterion];
    }

    let mut rngs = Vec::new();
    for criterion in 0..NB_CRITERIA as usize {
        let rng = -(cmp::min(dist, min_bounds[criterion]) as isize)
            ..(cmp::min(dist + 1, max_bounds[criterion]) as isize);
        rngs.push(rng);
    }

    let indices_generator = rngs
        .into_iter()
        .multi_cartesian_product()
        .filter(move |indices| indices.iter().map(|i| i.abs() as usize).sum::<usize>() == dist)
        .map(move |indices| {
            let mut add = [0; NB_CRITERIA];
            for ((addref, aval), bval) in add.iter_mut().zip(indices).zip(origin) {
                *addref = aval + bval as isize;
            }
            add
        });

    indices_generator
    // for val in indices_generator {
    //     println! {".... {:?}", val};
    // }

    // return indices_generator;
    //     add
    //     // indices
    //     //     .iter()
    //     //     .zip(origin.iter())
    //     //     .map(|(&i, &o)| (i + o as isize) as usize)
    //     //     .collect()
    // })

    // let t = rngs.iter().collect_tuple();
    // let it = iproduct!(t.iter());
    // for (a, b, c) in it {
    // println!("it {:?}", (a, b, c));
    // }

    // let min_bounds = core::array::from_fn(|_| )
    // let min_bounds: [usize; NB_CRITERIA] = origin
    //     .iter()
    //     .map(|val| *val)
    //     .collect::<Vec<usize>>()
    //     .try_into()
    //     .unwrap();
    // let max_bounds: [usize; NB_CRITERIA] = parameters
    //     .nb_intervals
    //     .iter()
    //     .zip(origin)
    //     .map(|(nb_interval, val)| *nb_interval - val)
    //     .collect::<Vec<usize>>()
    //     .try_into()
    //     .unwrap();

    // for criterion in 0..NB_CRITERIA {
    //     min_bounds[criterion] = origin[criterion];
    //     max_bounds[criterion] = parameters.nb_intervals[criterion] - origin[criterion];
    // }

    // let max_bounds: [usize; NB_CRITERIA];
    // // let bounds: [[usize; 2]; NB_CRITERIA] =

    // bounds = [
    //         [center[criterion], self.__nb_intervals[criterion] - center[criterion]]
    //         for criterion in range(NB_CRITERIA)
    //     ]
}

fn main() {
    // Instance related data
    let range = Uniform::new(MIN_WEIGHT, MAX_WEIGHT);
    let c_weights: Vec<CWeight> = (0..NB_WEIGHTS)
        .map(|_| core::array::from_fn(|_| range.sample(&mut thread_rng())))
        .collect();
    let instance: Instance<NB_CRITERIA> = Instance {
        c_weights: c_weights,
    };

    let mut sum_c_weights: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    for c_weight in instance.c_weights.iter() {
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
    let mut partition: Vec<usize> = vec![0; NB_WEIGHTS];

    // PartitionMetadata
    let curr_parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA] =
        eval_parts_loads_per_crit(partition, instance.c_weights.clone());
    // let curr_parts_imbalances_per_crit: [[i64; NB_PARTS]; NB_CRITERIA] =
    //     eval_parts_imbalances_per_crit(curr_parts_loads_per_crit, parts_target_load);
    // let partition_metadata: PartitionMetadata<NB_CRITERIA> = PartitionMetadata {
    //     parts_loads_per_crit: curr_parts_loads_per_crit,
    //     parts_imbalances_per_crit: curr_parts_imbalances_per_crit,
    // };

    // TargetorParameters
    let nb_intervals: [usize; NB_CRITERIA] = [NB_INTERVALS; NB_CRITERIA];
    let mut deltas: [f64; NB_CRITERIA] = [0.; NB_CRITERIA];
    for (criterion, (max_weight, min_weight)) in MAX_WEIGHTS.iter().zip(MIN_WEIGHTS).enumerate() {
        let f_max_weight = *max_weight as f64;
        let f_min_weight = min_weight as f64;
        let f_nb_intervals = NB_INTERVALS as f64;
        deltas[criterion] = (f_max_weight - f_min_weight) / f_nb_intervals;
    }
    let tp: TargetorParameters<NB_CRITERIA> = TargetorParameters {
        deltas: deltas,
        nb_intervals: nb_intervals,
        parts_target_load: parts_target_load,
    };

    let dist = 1;
    let boxes = compute_boxes(instance.c_weights.clone(), tp);
    let indices_generator = generate_indices([0; 3], dist, tp);
    for val in indices_generator {
        println! {".... {:?}", val};
    }

    // let map_2 = compute_boxes(instance, tp);
    // println!("~~~ {:?}", res);
    // for (criterion, (max, min)) in MAX_WEIGHTS.iter().zip(&MIN_WEIGHTS).enumerate() {
    //     deltas[criterion] = (max - min) / NB_INTERVALS[criterion];
    // }
    // let deltas [f64; NB_CRITERIA] = MAX_WEIGHTS.iter().zip(&MIN_WEIGHTS).map(
    //     |max, min| (max - min) /
    // )

    // (MAX_WEIGHT - MIN_WEIGHT) / nb_intervals[crit]
    // targetor_parameters = TargetorParameters {
    //     nb_intervals,
    //     deltas,
    //     parts_target_load,
    // }

    // println!("parts_target_load {:?}", parts_target_load);
    // println!("parts_loads_per_crit {:?}", curr_parts_loads_per_crit);
    // println!(
    //     "parts_imbalances_per_crit {:?}",
    //     curr_parts_imbalances_per_crit
    // );

    // println!("parts_imbalances_per_crit {:?}", partition_metadata);
    // println!(" {:?}", partition_metadata);

    // self.parts_imbalances_per_crit: List[List[Union[int, float]]] = (
    //     self.parts_loads_per_crit - self.parameters.parts_target_load
    // )

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
    // println!("{:?}", parts_target_load);
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
    // let parts_loads_per_crit: [[usize; NB_PARTS]; NB_CRITERIA] =
    //     eval_partition_metadata(partition, c_weights);
    // println!("{:?}", parts_loads_per_crit);

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
