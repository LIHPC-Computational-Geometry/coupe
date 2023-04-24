use coupe::imbalance;
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
use std::cmp::max;
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

// pub struct NotImplementedError;
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

fn compute_candidate_moves(
    // fn filtered_boxed_moves<'a>(
    // origin: &'a [usize; NB_CRITERIA],
    part_source: usize,
    partition_imbalance: usize,
    // boxes: &'a HashMap<[usize; NB_CRITERIA], Vec<usize>>,
    box_content: &Vec<usize>,
    partition: &Vec<usize>,
    parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA],
    c_weights: Vec<CWeight>,
    // ) -> Box<dyn Iterator<Item = (&'a usize, usize)> + 'a> {
) -> Vec<(usize, usize)> {
    // ) {
    println!("ICIIIIIIIIIIIIIIIIII {:?}", box_content);
    // println!("ICIIIIIIIIIIIIIIIIII ",);
    let part_target = 1 - part_source;
    let max_imbalance_per_crit: [usize; NB_CRITERIA] = parts_imbalances_per_crit
        .iter()
        .map(|imbalances| *imbalances.iter().max().unwrap() as usize)
        .collect::<Vec<usize>>()
        .try_into()
        .unwrap();
    // println!("~~~ {:?} {:?}", boxes, origin);

    let strict_positive_gain = |c_weight_index: usize| {
        !(0..NB_CRITERIA).any(|criterion| {
            max_imbalance_per_crit[criterion] == partition_imbalance
                && c_weights[c_weight_index][criterion] >= 2 * partition_imbalance
                || max_imbalance_per_crit[criterion] != partition_imbalance
                    && c_weights[c_weight_index][criterion] as isize
                        >= partition_imbalance as isize
                            - parts_imbalances_per_crit[criterion][part_target]
        })
    };

    // let mut box_indices: Vec<usize> = boxes.get(origin).unwrap().clone();
    // println!("{}", my_list.iter().any(|&i| i == 4));
    // println!("box indices {:?}", box_indices);
    // box_indices = box_indices
    //     .iter()
    //     .filter(|&&c_weight_index| partition[c_weight_index] == part_source)
    //     .collect::<Vec<usize>>();
    // let mut box_indices: Vec<usize> = boxes
    let mut box_indices: Vec<usize> = box_content
        .iter()
        .cloned()
        // .into_iter()
        // Filter c_weight indices that are not assigned to part_source
        .filter(|c_weight_index| partition[*c_weight_index] == part_source)
        // Filter moves with settled wieghts, i.e. weights whose move would
        // leave to a higher partition imbalance
        .filter(|c_weight_index| strict_positive_gain(*c_weight_index))
        .collect();

    println!("box indices {:?}", box_indices);

    let moves = box_indices
        .iter()
        .map(move |c_weight_index| (*c_weight_index, part_target))
        .collect();

    println!("moves {:?}", moves);
    moves
}

fn filtered_boxed_moves_bis<'a>(
    origin: &'a [usize; NB_CRITERIA],
    part_source: usize,
    partition_imbalance: usize,
    boxes: &'a HashMap<[usize; NB_CRITERIA], Vec<usize>>,
    partition: &'a Vec<usize>,
    parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA],
    c_weights: Vec<CWeight>,
    // ) -> Box<dyn Iterator<Item = (&'a &'a usize, usize)> + 'a> {
) {
    let part_target = 1 - part_source;
    let max_imbalance_per_crit: [usize; NB_CRITERIA] = parts_imbalances_per_crit
        .iter()
        .map(|imbalances| *imbalances.iter().max().unwrap() as usize)
        .collect::<Vec<usize>>()
        .try_into()
        .unwrap();
    println!("~~~ {:?} {:?}", boxes, origin);

    let strict_positive_gain = |&&c_weight_index: &&usize| {
        !(0..NB_CRITERIA).any(|criterion| {
            max_imbalance_per_crit[criterion] == partition_imbalance
                && c_weights[criterion][c_weight_index] >= 2 * partition_imbalance
                || max_imbalance_per_crit[criterion] != partition_imbalance
                    && c_weights[criterion][c_weight_index] as isize
                        >= partition_imbalance as isize
                            - parts_imbalances_per_crit[criterion][part_target]
        })
    };

    // let mut box_indices: Vec<usize> = boxes.get(origin).unwrap().clone();
    // println!("{}", my_list.iter().any(|&i| i == 4));
    // println!("box indices {:?}", box_indices);
    // box_indices = box_indices
    //     .iter()
    //     .filter(|&&c_weight_index| partition[c_weight_index] == part_source)
    //     .collect::<Vec<usize>>();
    // let mut box_indices: Vec<usize> = boxes
    let mut box_indices: Vec<&usize> = boxes
        .get(origin)
        .unwrap()
        .iter()
        // Filter c_weight indices that are not assigned to part_source
        .filter(|&&c_weight_index| partition[c_weight_index] == part_source)
        // Filter moves with settled wieghts, i.e. weights whose move would
        // leave to a higher partition imbalance
        .filter(|c_weight_index| strict_positive_gain(c_weight_index))
        .collect();

    let moves = Box::new(
        box_indices
            .iter()
            .map(move |c_weight_index| (c_weight_index, part_target)),
    );

    println!("box indices {:?}", box_indices);
    println!("moves {:?}", moves);
    // moves
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
            let mut add: [isize; NB_CRITERIA] = [0; NB_CRITERIA];
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

fn process_imbalance(
    parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA],
) -> (Vec<usize>, Vec<usize>) {
    let max_imbalance_per_crit: [usize; NB_CRITERIA] = parts_imbalances_per_crit
        .iter()
        .map(|imbalances| *imbalances.iter().max().unwrap() as usize)
        .collect::<Vec<usize>>()
        .try_into()
        .unwrap();
    println!("[MAX IMB PER CRIT] {:?}", max_imbalance_per_crit,);

    let partition_imbalance: usize = *max_imbalance_per_crit.iter().max().unwrap();
    let mut criteria_most_imbalanced = Vec::new();
    for (criterion, &imbalance) in max_imbalance_per_crit.iter().enumerate() {
        if imbalance == partition_imbalance {
            criteria_most_imbalanced.push(criterion)
        }
    }
    // println!("yes{:?}", most_imbalanced_criteria);
    // let res = max_imbalance_per_crit.iter().enumerate().fold(
    //     Vec::new(),
    //     |mut mut_res, (criterion, &imbalance)| {
    //         if imbalance == partition_imbalance {
    //             mut_res.push(criterion);
    //         }
    //         mut_res
    //     },
    // );
    // println!("yes{:?}", res);

    let mut parts_source = Vec::new();
    for imbalance in parts_imbalances_per_crit.iter() {
        for (part, &imbalance) in imbalance.iter().enumerate() {
            if imbalance as usize == partition_imbalance {
                parts_source.push(part)
            }
        }
    }
    println!("criteria_most_imbalanced{:?}", criteria_most_imbalanced);
    println!("part sources{:?}", parts_source);

    (criteria_most_imbalanced, parts_source)
    // let part_sources = parts_imbalances_per_crit.iter().enumerate().map(
    //     |(i, (min, center))| {
    //         if (region >> i) & 1 == 0 {
    //             *min
    //         } else {
    //             *center
    //         }
    //     },
    // ));
    // .fold(
    //     Vec::new(),
    //     [mut ret, ()]
    // );

    // let mut parts_source = Vec::new();
    // for imbalance in parts_imbalances_per_crit.iter() {

    // }

    // // for (criterion, value) in max_imbalance_per_crit.iter().enumerate(){
    // // }
    // max_imbalance_per_crit
    //     .iter()
    //     .enumerate()
    //     .max_by_key(|(_, &value)| value)
    //     .map(|(idx, _)| v.push(idx));

    // let mut r = Vec::new();
    // // Put each weight in the lightweightest part.
    // for (weight, weight_id) in weights.into_iter().rev() {
    //     let (min_part_weight_idx, _min_part_weight) = r
    //         .iter()
    //         .enumerate()
    //         .min_by(|(_, part_weight0), (_, part_weight1)| {
    //             crate::partial_cmp(part_weight0, part_weight1)
    //         })
    //         .unwrap(); // Will not panic because !part_weights.is_empty()
    //     partition[weight_id] = min_part_weight_idx;
    //     part_weights[min_part_weight_idx] += weight;
    // }
    // let partition_imbalance = max_imbalance_per_crit[most_imbalanced_criterion];

    // let found = false;
    // for imbalances in parts_imbalances_per_crit {
    //     for (part, imbalance) in imbalances.iter().enumerate() {
    //         if imbalance == partition_imbalance {
    //             part_source = part
    //         }
    //     }
    // }

    // println!(
    //     "res {:?}, {}",
    //     max_imbalance_per_crit, most_imbalanced_criterion
    // );
}

fn find_move(
    parameters: TargetorParameters<NB_CRITERIA>,
    c_weights: Vec<CWeight>,
    parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA],
    boxes: HashMap<[usize; NB_CRITERIA], Vec<usize>>,
    partition: &Vec<usize>,
) -> Option<(usize, usize)> {
    let (criteria_most_imbalanced, parts_source) = process_imbalance(parts_imbalances_per_crit);

    // No more strictly positive gain can be achieved through a move
    if parts_source.len() > 1 {
        return None;
    }
    let part_source: usize = parts_source[0];

    // Setup target gain
    let mut target_gain: [usize; NB_CRITERIA] = [0; NB_CRITERIA];
    target_gain = core::array::from_fn::<usize, NB_CRITERIA, _>(|criterion| {
        let mut gain = parts_imbalances_per_crit[criterion][part_source].abs() as usize;
        if gain > MAX_WEIGHTS[criterion] {
            gain = MAX_WEIGHTS[criterion];
        }
        gain
    });

    // Retrieve candidates
    // let target_box_index = compute_box_indices(target_gain, parameters).clone();
    let target_box_index = compute_box_indices(target_gain, parameters);
    // let target_box_index = compute_box_indices(c_weights[0], parameters);
    let partition_imbalance: usize =
        parts_imbalances_per_crit[criteria_most_imbalanced[0]][part_source] as usize;

    println!("target gain {:?}", target_gain);

    println!(
        "boxes {:?} and target_box_index {:?} contains {:?}",
        boxes,
        &target_box_index,
        boxes.contains_key(&target_box_index)
    );

    if boxes.contains_key(&target_box_index) {
        let box_content = boxes.get(&target_box_index as &[usize; 3]).unwrap();
        let candidate_moves = compute_candidate_moves(
            // &target_box_index,
            part_source,
            partition_imbalance,
            box_content,
            partition,
            parts_imbalances_per_crit,
            c_weights.clone(),
        );
        if !candidate_moves.is_empty() {
            return Some(candidate_moves[0]);
        }
        // return candidate_moves.next();
    }

    let mut dist: usize = 1;
    loop {
        let iter_indices = generate_indices(target_box_index, dist, parameters);
        for box_index in iter_indices {
            if boxes.contains_key(&box_index) {
                let box_content = boxes.get(&box_index).unwrap();
                let candidate_moves = compute_candidate_moves(
                    // &target_box_index,
                    part_source,
                    partition_imbalance,
                    box_content,
                    partition,
                    parts_imbalances_per_crit,
                    c_weights.clone(),
                );
                if !candidate_moves.is_empty() {
                    return Some(candidate_moves[0]);
                }
                // return candidate_moves.next();
            }
        }

        dist += 1;
        let bound_indices = compute_box_indices([partition_imbalance; NB_CRITERIA], parameters);
        let increase_offset = (0..NB_CRITERIA).any(|criterion| {
            target_box_index[criterion] >= dist
                || target_box_index[criterion] + dist <= bound_indices[criterion]
        });
        if !increase_offset {
            return None;
        }
        // bound_indices = self.__box_indices([partition_imbalance] * NB_CRITERIA)
        // increase_offset = any(
        //     map(
        //         lambda criterion: target_box_index[criterion] - offset >= 0
        //         or target_box_index[criterion] + offset
        //         <= bound_indices[criterion],
        //         range(NB_CRITERIA),
        //     )
        // )
        // if not increase_offset:
        //     print("move not found with offset", offset)
        //     break
    }

    // return None;

    // return Some(target_gain);
    // target_gain.iter().enumerate().map(
    //     |(criterion, _)| parts_imbalances_per_crit[]
    // )
    // let a = core::array::from_fn<usize,NB_CRITERIA,_>()
    // return [12, 0];
}

// print("[[MAX IMB PER CRIT]]", max_imbalance_per_crit)
// most_imbalanced_criterion = np.argmax(max_imbalance_per_crit)
// partition_imbalance = max_imbalance_per_crit[most_imbalanced_criterion]

// found = False
// for imbalances in self.parts_imbalances_per_crit:
//     for part, imbalance in enumerate(imbalances):
//         if imbalance == partition_imbalance:
//             part_source = part
//             found = True
//             break
//     if found:
//         break

// return most_imbalanced_criterion, part_source

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
        eval_parts_loads_per_crit(partition.clone(), instance.c_weights.clone());
    let curr_parts_imbalances_per_crit: [[isize; NB_PARTS]; NB_CRITERIA] =
        eval_parts_imbalances_per_crit(curr_parts_loads_per_crit, parts_target_load);
    let partition_metadata: PartitionMetadata<NB_CRITERIA> = PartitionMetadata {
        parts_loads_per_crit: curr_parts_loads_per_crit,
        parts_imbalances_per_crit: curr_parts_imbalances_per_crit,
    };

    // process_imbalance(curr_parts_imbalances_per_crit);
    let (criteria_most_imbalanced, parts_source) = process_imbalance([[12, 0], [0, 12], [5, 5]]);

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

    let res = find_move(
        tp,
        instance.c_weights.clone(),
        curr_parts_imbalances_per_crit,
        boxes,
        &partition,
    );

    // let res = find_move(
    //     tp,
    //     instance.c_weights.clone(),
    //     curr_parts_imbalances_per_crit,
    //     boxes.clone(),
    // );

    println!("RESULT {:?}", res);

    // // let : usize = parts_imbalances_per_crit.iter().max().unwrap();
    // let partition_imbalance: usize = curr_parts_imbalances_per_crit
    //     .iter()
    //     .map(|imbalances| *imbalances.iter().max().unwrap() as usize)
    //     // .collect::<Vec<usize>>()
    //     // .try_into()
    //     // .unwrap()
    //     // .iter()
    //     .max()
    //     .unwrap();
    // println!("[IMBALANCE] {:?}", partition_imbalance);

    // let part_source = parts_source[0];
    // let origin = compute_box_indices(instance.c_weights.clone()[0], tp);
    // // let partition_imbalance: usize = *max_imbalance_per_crit.iter().max().unwrap();
    // filtered_boxed_moves(
    //     &origin,
    //     part_source,
    //     partition_imbalance,
    //     &boxes,
    //     &partition,
    //     curr_parts_imbalances_per_crit,
    //     instance.c_weights.clone(),
    // )
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
