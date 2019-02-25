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
    num_iter: usize,
    max_imbalance_per_iter: f64,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
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

    for (mut p, mut q) in adjacent_parts {
        fm2(
            weights,
            &mut p,
            &mut q,
            adjacency.view(),
            ids,
            num_iter,
            max_imbalance_per_iter,
            10,
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

fn fm2(
    weights: &[f64],
    idx1: &mut [usize], // indices of elements in first part
    idx2: &mut [usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    mut num_iter: usize,
    max_imbalance_per_iter: f64,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) {
    let first_id = initial_partition[idx1[0]];
    let second_id = initial_partition[idx2[0]];
    let swap_id = move |id: ProcessUniqueId| {
        if id == first_id {
            second_id
        } else {
            first_id
        }
    };

    let num_gains = 8;

    let indices = idx1
        .iter()
        .cloned()
        .chain(idx2.iter().cloned())
        .collect::<Vec<_>>();

    let mut cut_size = crate::topology::cut_size(adjacency.view(), initial_partition);
    let mut new_cut_size = cut_size;

    loop {
        num_iter -= 1;
        dbg!(cut_size);
        dbg!(new_cut_size);

        // update cut size
        cut_size = new_cut_size;

        let mut gains = gains_single(&indices, adjacency.view(), initial_partition);
        let mut saves = Vec::new();
        let mut cut_saves = Vec::new();
        let mut locks = vec![false; indices.len()];
        let mut num_bad_move = 0;

        // pass loop
        for _ in 0..indices.len() {
            // find max gain that is not locked
            let (max_pos, max_gain) = indices
                .iter()
                .cloned()
                .zip(gains.iter().cloned())
                .filter(|(max_pos, _)| !locks[*max_pos])
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
        }
        dbg!(&saves[0..=best_pos]);
        // dbg!(&gains);
        new_cut_size = best_cut;

        if new_cut_size >= cut_size {
            break;
        }
    }

    println!("final cut size: {}", new_cut_size);
}

fn compute_descrete_gain_range(gains: &[f64], num_gains: usize) -> Vec<f64> {
    let max_gain = gains
        .iter()
        .max_by(|w1, w2| w1.partial_cmp(&w2).unwrap())
        .unwrap();
    (1..num_gains / 2)
        .rev()
        .map(|n| -1. * n as f64 * max_gain / (num_gains / 2) as f64)
        .chain((0..num_gains / 2).map(|n| n as f64 * max_gain / (num_gains / 2) as f64))
        .collect()
}

fn gains(
    idx1: &[usize], // indices of elements in current part
    idx2: &[usize], // indices of elements in other part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
) -> Vec<f64> {
    idx1.par_iter()
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
        .collect::<Vec<_>>()
}

fn gains_single(
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
