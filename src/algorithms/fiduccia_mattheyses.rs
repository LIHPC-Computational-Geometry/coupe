use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};
use rayon::prelude::*;
use sprs::CsMatView;

use crate::partition::Partition;
use crate::PointND;
use crate::ProcessUniqueId;
use std::collections::{HashMap, LinkedList};

pub fn fiduccia_mattheyses<'a, D>(
    initial_partition: &mut Partition<'a, PointND<D>, f64>,
    adjacency: CsMatView<f64>,
    max_passes: impl Into<Option<usize>>,
    max_flips_per_pass: impl Into<Option<usize>>,
    max_imbalance_per_flip: impl Into<Option<f64>>,
    max_bad_move_in_a_row: usize,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    let max_passes = max_passes.into();
    let max_flips_per_pass = max_flips_per_pass.into();
    let max_imbalance_per_flip = max_imbalance_per_flip.into();

    // To adapt Kernighan-Lin to a partition of more than 2 parts,
    // we apply the algorithm to each pair of adjacent parts (two parts
    // are adjacent if there exists an element in one part that is linked to
    // an element in the other part).

    let adjacent_parts = initial_partition
        .adjacent_parts(adjacency.view())
        .into_iter()
        .map(|(p, q)| (p.into_indices(), q.into_indices()))
        .collect::<Vec<_>>();

    let (_points, weights, ids) = initial_partition.as_raw_mut();

    fmn(
        weights,
        adjacency.view(),
        ids,
        max_passes,
        max_flips_per_pass,
        max_imbalance_per_flip,
        max_bad_move_in_a_row,
    );
    return;

    for (mut p, mut q) in adjacent_parts {
        fm2(
            weights,
            &mut p,
            &mut q,
            adjacency.view(),
            ids,
            max_passes,
            max_flips_per_pass,
            max_imbalance_per_flip,
            max_bad_move_in_a_row,
        );
    }
}

pub fn fiduccia_mattheyses_2(
    weights: &[f64],
    idx1: &mut [usize], // indices of elements in first part
    idx2: &mut [usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
    max_imbalance_per_iter: f64,
) {
    // let possible_ids = initial_partition
    //     .iter()
    //     .cloned()
    //     .unique()
    //     .collect::<Vec<_>>();
    // assert_eq!(possible_ids.len(), 2);

    let first_id = initial_partition[idx1[0]];
    let second_id = initial_partition[idx2[0]];
    let swap_id = move |id: ProcessUniqueId| {
        if id == first_id {
            second_id
        } else {
            first_id
        }
    };

    for _ in 0..num_iter {
        let gains1 = idx1
            .par_iter()
            .cloned()
            .map(|i| {
                adjacency
                    .outer_view(i)
                    .and_then(|row| {
                        Some(
                            idx2.iter()
                                .chain(idx1.iter())
                                .filter_map(|j| {
                                    row.nnz_index(*j)
                                        .and_then(|nnz_idx| Some((j, row[nnz_idx])))
                                })
                                .fold(0., |acc, (j, w)| {
                                    if initial_partition[i] != initial_partition[*j] {
                                        acc + w
                                    } else {
                                        acc - w
                                    }
                                }),
                        )
                    })
                    .unwrap_or(0.)
            })
            .collect::<Vec<_>>();

        let gains2 = idx2
            .par_iter()
            .cloned()
            .map(|i| {
                adjacency
                    .outer_view(i)
                    .and_then(|row| {
                        Some(
                            idx2.iter()
                                .chain(idx1.iter())
                                .filter_map(|j| {
                                    row.nnz_index(*j)
                                        .and_then(|nnz_idx| Some((j, row[nnz_idx])))
                                })
                                .fold(0., |acc, (j, w)| {
                                    if initial_partition[i] != initial_partition[*j] {
                                        acc + w
                                    } else {
                                        acc - w
                                    }
                                }),
                        )
                    })
                    .unwrap_or(0.)
            })
            .collect::<Vec<_>>();

        // let (max_pos, max_gain) = gains
        //     .par_iter()
        //     .enumerate()
        //     .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
        //     .unwrap();

        let (max_pos_1, max_gain_1) = gains1
            .par_iter()
            .cloned()
            .enumerate()
            .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
            .unwrap();

        let (max_pos_2, max_gain_2) = gains2
            .par_iter()
            .cloned()
            .enumerate()
            .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
            .unwrap();

        if max_gain_1 > max_gain_2 {
            println!("swapping pos: {} for gain {}", max_pos_1, max_gain_1);
            initial_partition[idx1[max_pos_1]] = swap_id(initial_partition[idx1[max_pos_1]]);
        } else {
            println!("swapping pos: {} for gain {}", max_pos_2, max_gain_2);
            initial_partition[idx2[max_pos_2]] = swap_id(initial_partition[idx2[max_pos_2]]);
        }
    }
}

// todo take
fn fm2(
    weights: &[f64],
    idx1: &mut [usize], // indices of elements in first part
    idx2: &mut [usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) {
    // utility function to easily swap nodes between two parts
    let first_id = initial_partition[idx1[0]];
    let second_id = initial_partition[idx2[0]];
    let swap_id = move |id: ProcessUniqueId| {
        if id == first_id {
            second_id
        } else {
            first_id
        }
    };

    // we group indices into a single array so that
    // it's easier to work with
    let indices = idx1
        .iter()
        .cloned()
        .chain(idx2.iter().cloned())
        .collect::<Vec<_>>();

    let mut cut_size = crate::topology::cut_size(adjacency.view(), initial_partition);
    println!("Initial cut size: {}", cut_size);
    let mut new_cut_size = cut_size;

    // monitor imbalance generated by flipping elements between parts
    let mut imbalance = 0.;

    // set a pass limit but the algorithm can also
    // exit on a cut_size condition
    for iter in 0.. {
        // break if pass limit is reached
        if let Some(max_passes) = max_passes {
            if iter >= max_passes {
                break;
            }
        }

        // update cut size
        cut_size = new_cut_size;

        let mut gains = gains(&indices, adjacency.view(), initial_partition);
        let mut saves = Vec::new();
        let mut cut_saves = Vec::new();
        let mut locks = vec![false; indices.len()];
        let mut num_bad_move = 0;

        // pass loop
        for _ in 0..indices
            .len()
            .min(max_flips_per_pass.unwrap_or(std::usize::MAX))
        {
            // find max gain that is not locked
            let (max_pos, max_gain) = indices
                .iter()
                .cloned()
                .zip(gains.iter().cloned())
                .filter(|(max_pos, _)| {
                    if let Some(max_imbalance_per_flip) = max_imbalance_per_flip {
                        !locks[*max_pos] && weights[*max_pos] <= max_imbalance_per_flip
                    } else {
                        !locks[*max_pos]
                    }
                })
                .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
                .unwrap();

            if max_gain < 0. {
                if num_bad_move >= max_bad_move_in_a_row {
                    println!("reached max bad move in a row");
                    break;
                }

                num_bad_move += 1;
            }

            // save movement
            saves.push((max_pos, max_gain));
            locks[max_pos] = true;

            // update neighbors gains
            let row = adjacency.outer_view(max_pos).unwrap();
            for j in indices.iter().cloned() {
                if let Some(w) = row.get(j) {
                    if j != max_pos {
                        // update gain of j
                        if initial_partition[max_pos] != initial_partition[j] {
                            gains[j] -= 2. * w;
                        } else {
                            gains[j] += 2. * w;
                        }
                    }
                }
            }

            // update imbalance
            if initial_partition[max_pos] == first_id {
                imbalance += weights[max_pos];
            } else {
                imbalance -= weights[max_pos];
            }

            // flip node and save new cutsize
            initial_partition[max_pos] = swap_id(initial_partition[max_pos]);

            cut_saves.push(crate::topology::cut_size(
                adjacency.view(),
                initial_partition,
            ));
        }
        eprintln!("cut saves = {:?}", &cut_saves);
        // after pass lookup for best cutsize
        let (best_pos, best_cut) = cut_saves
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        // rewind flips until reaching best cut
        println!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            cut_saves.len()
        );
        for i in best_pos + 1..cut_saves.len() {
            let idx = saves[i].0;
            initial_partition[idx] = swap_id(initial_partition[idx]);

            // undo imbalance changes
            if initial_partition[idx] == first_id {
                imbalance -= weights[idx];
            } else {
                imbalance += weights[idx];
            }
        }

        new_cut_size = best_cut;

        if new_cut_size >= cut_size {
            break;
        }
    }

    println!("final cut size: {}", new_cut_size);
    println!("total imbalance introduced: {}", imbalance.abs());
}

fn gains(
    idx: &[usize],
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
) -> Vec<f64> {
    idx.par_iter()
        .cloned()
        .map(|i| {
            adjacency
                .outer_view(i)
                .and_then(|row| {
                    Some(
                        idx.iter()
                            .filter_map(|j| {
                                row.nnz_index(*j)
                                    .and_then(|nnz_idx| Some((j, row[nnz_idx])))
                            })
                            .fold(0., |acc, (j, w)| {
                                if initial_partition[i] != initial_partition[*j] {
                                    acc + w
                                } else {
                                    acc - w
                                }
                            }),
                    )
                })
                .unwrap_or(0.)
        })
        .collect::<Vec<_>>()
}

pub fn fmn(
    weights: &[f64],
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) {
    let unique_ids = initial_partition
        .iter()
        .cloned()
        .unique()
        .collect::<Vec<_>>();

    // store weights of each part to update imbalance easily
    let mut parts_weights: HashMap<ProcessUniqueId, f64> = unique_ids
        .iter()
        .cloned()
        .map(|id| {
            (
                id,
                weights
                    .iter()
                    .zip(initial_partition.iter().cloned())
                    .filter(|(_w, other_id)| id == *other_id)
                    .map(|(w, _)| *w)
                    .sum::<f64>(),
            )
        })
        .collect();

    let mut cut_size = crate::topology::cut_size(adjacency.view(), initial_partition);
    println!("Initial cut size: {}", cut_size);
    let mut new_cut_size = cut_size;

    // Outer loop: each iteration makes a "pass" which can flip several nodes
    // at a time. Repeat passes until passes no longer decrease the cut size.
    for iter in 0.. {
        // check user defined iteration limit
        if let Some(max_passes) = max_passes {
            if iter >= max_passes {
                break;
            }
        }

        // save old cut size
        cut_size = new_cut_size;

        // imbalance introduced by flipping nodes around parts
        let imbalance;

        // monitors for each pass the number of subsequent flips
        // that increase cut size. It may be beneficial in some
        // situations to allow a certain amount of them. Performing bad flips can open
        // up new sequences of good flips.
        let mut num_bad_move = 0;

        // We save each flip data during a pass so that they can be reverted easily
        // afterwards. For instance if performing wrong flips did not open up any
        // good flip sequence, those bad flips must be reverted at the end of the pass
        // so that cut size remains optimal
        let mut saves = Vec::new(); // flip save
        let mut cut_saves = Vec::new(); // cut size save
        let mut ids_before_flip = Vec::new(); // the target id for reverting a flip

        // create gain data structure
        // for each node, a gain is associated to each possible target part.
        // It is currently implemented wit an array of vectors:
        // [
        //  node_1: [(target_part_1, gain_1), ..., (target_part_n, gain_n)],
        //  ...,
        //  node_n: [(target_part_1, gain_1), ..., (target_part_n, gain_n)]
        // ]
        //
        // note that the current part in wich a node is is still considered as a potential target part
        // with a gain 0.
        let mut gains: Vec<Vec<(ProcessUniqueId, f64)>> = (0..initial_partition.len())
            .map(|_idx| unique_ids.iter().cloned().map(|id2| (id2, 0.)).collect())
            .collect();

        // lock array
        // during a loop iteration, if a node is flipped during a pass,
        // it becomes locked and can't be flipped again during the following passes,
        // and is unlocked at next loop iteration.
        // locks are per node and do not depend on target partition.
        let mut locks = vec![false; initial_partition.len()];

        // enter pass loop
        // The number of iteration of the pas loop is at most the
        // number of nodes in the mesh. However, if too many subsequent
        // bad flips are performed, the loop will break early
        for _ in 0..initial_partition
            .len()
            .min(max_flips_per_pass.unwrap_or(std::usize::MAX))
        {
            // construct gains
            // Right now all of the gains are recomputed at each new pass
            // a possible optimization would be to use a different gain data structure
            // to modify only some of the gains instead of recomputing everything.
            //
            // for each node (assigned to part p), and for each target part q (with p != q),
            // gain contributiuon comes from each node neighbor:
            //   - if the neighbor is in part p, then the flip will increase cut size
            //   - if the neighbor is in part q, then the flip will decrease cut size
            //   - if the neighbor is not in part p nor in q, then the flip won't affect the cut size
            for (idx, other_ids) in gains.iter_mut().enumerate() {
                for (id2, ref mut gain) in other_ids.iter_mut() {
                    if initial_partition[idx] == *id2 {
                        // target part is current part, no gain
                        *gain = 0.;
                    } else {
                        for (j, w) in adjacency.outer_view(idx).unwrap().iter() {
                            if initial_partition[idx] == initial_partition[j] {
                                *gain -= w;
                            } else if initial_partition[j] == *id2 {
                                *gain += w;
                            }
                        }
                    }
                }
            }

            // find max gain and target part
            let (max_pos, (target_part, max_gain)) = gains
                .iter()
                .zip(locks.iter())
                .zip(weights.iter())
                .enumerate()
                .filter(|(_, ((_, locked), weight))| {
                    if let Some(max_imbalance_per_flip) = max_imbalance_per_flip {
                        !*locked && **weight <= max_imbalance_per_flip
                    } else {
                        !*locked
                    }
                })
                .map(|(idx, ((vec, _), _))| (idx, vec))
                .map(|(idx, vec)| {
                    // (index of node, (target part of max gain, max gain))
                    (
                        idx,
                        *vec.iter()
                            .max_by(|(_idx1, gain1), (_id2, gain2)| {
                                gain1.partial_cmp(&gain2).unwrap()
                            })
                            .unwrap(),
                    )
                })
                // get max gain of max gains computed for each node
                .max_by(|(_, (_, gain1)), (_, (_, gain2))| gain1.partial_cmp(&gain2).unwrap())
                .unwrap();

            if max_gain <= 0. {
                if num_bad_move >= max_bad_move_in_a_row {
                    println!("reached max bad move in a row");
                    break;
                }
                num_bad_move += 1;
            } else {
                // a good move breaks the bad moves sequence
                num_bad_move = 0;
            }

            // lock node
            locks[max_pos] = true;

            // save flip
            saves.push((max_pos, target_part, max_gain));

            // eprintln!(
            //     "save: {:?}",
            //     saves.iter().map(|(_, _, g)| g).collect::<Vec<_>>()
            // );
            // eprintln!(
            //     "indices: {:?}",
            //     saves.iter().map(|(pos, _, _)| pos).collect::<Vec<_>>()
            // );

            // update imbalance
            *parts_weights.get_mut(&initial_partition[max_pos]).unwrap() -= weights[max_pos];
            *parts_weights.get_mut(&target_part).unwrap() += weights[max_pos];

            // flip node
            ids_before_flip.push(initial_partition[max_pos]);
            initial_partition[max_pos] = target_part;

            // save cut_size
            cut_saves.push(crate::topology::cut_size(
                adjacency.view(),
                initial_partition,
            ));

            // end of pass
        }

        // lookup for best cutsize
        let (best_pos, best_cut) = cut_saves
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        // rewind flips
        println!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            cut_saves.len()
        );
        for i in best_pos + 1..cut_saves.len() {
            let idx = saves[i].0;
            initial_partition[idx] = ids_before_flip[i];

            // revert weight change
            *parts_weights.get_mut(&ids_before_flip[i]).unwrap() += weights[idx];
            *parts_weights.get_mut(&saves[i].1).unwrap() += weights[idx];
        }

        new_cut_size = best_cut;

        if new_cut_size >= cut_size {
            break;
        }

        let (min_w, max_w) = parts_weights.values().minmax().into_option().unwrap();
        imbalance = max_w - min_w;
        dbg!(imbalance);
    }

    println!("final cut size: {}", new_cut_size);
}
