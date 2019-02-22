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

pub fn fiduccia_mattheyses_2_ll(
    weights: &[f64],
    idx1: &mut [usize], // indices of elements in first part
    idx2: &mut [usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
    max_imbalance_per_iter: f64,
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

    for _ in 0..num_iter {
        let mut locks_left = vec![false; idx1.len()];
        let mut locks_right = vec![false; idx2.len()];
        let num_elem = idx1.len() + idx2.len();

        let mut weight_left = idx1.iter().cloned().map(|i| weights[i]).sum::<f64>();
        let mut weight_right = idx2.iter().cloned().map(|i| weights[i]).sum::<f64>();

        let neigbors_left = idx1
            .iter()
            .cloned()
            .map(|i| {
                adjacency
                    .outer_view(i)
                    .expect("isolated node")
                    .iter()
                    .map(|(j, _w)| j)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let neigbors_right = idx2
            .iter()
            .cloned()
            .map(|i| {
                adjacency
                    .outer_view(i)
                    .expect("isolated node")
                    .iter()
                    .map(|(j, _w)| j)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut balance = weight_right - weight_left;

        let mut gains1 = idx1
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

        let mut gains2 = idx2
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

        if balance < 0. {
            for _ in 0..idx1.len() {
                // flip a node from underweighted part

                let (pos, max_gain_pos) = idx1
                    .iter()
                    .cloned()
                    .zip(locks_left.iter())
                    .enumerate()
                    .filter(|(i, (idx, locked))| !**locked)
                    .map(|(i, (idx, locked))| (i, idx))
                    .max_by(|(i1, _), (i2, _)| gains1[*i1].partial_cmp(&gains1[*i2]).unwrap())
                    .unwrap();

                for neighbor_idx in neigbors_left[pos].iter() {
                    if locks_left[pos] {
                        // update gain
                        gains1[pos] += if initial_partition[*neighbor_idx]
                            == initial_partition[max_gain_pos]
                        {
                            -2. * adjacency.get(max_gain_pos, *neighbor_idx).unwrap()
                        } else {
                            -2. * adjacency.get(max_gain_pos, *neighbor_idx).unwrap()
                        };

                        // lock node
                    }
                }
                locks_left[pos] = true;
            }
        } else {
            for _ in 0..idx2.len() {
                // flip a node from underweighted part

                let (pos, max_gain_pos) = idx2
                    .iter()
                    .cloned()
                    .zip(locks_right.iter())
                    .enumerate()
                    .filter(|(i, (idx, locked))| !**locked)
                    .map(|(i, (idx, locked))| (i, idx))
                    .max_by(|(i1, _), (i2, _)| gains1[*i1].partial_cmp(&gains1[*i2]).unwrap())
                    .unwrap();

                for neighbor_idx in neigbors_right[pos].iter() {
                    if locks_right[pos] {
                        // update gain
                        gains2[pos] += if initial_partition[*neighbor_idx]
                            == initial_partition[max_gain_pos]
                        {
                            -2. * adjacency.get(max_gain_pos, *neighbor_idx).unwrap()
                        } else {
                            -2. * adjacency.get(max_gain_pos, *neighbor_idx).unwrap()
                        };

                        // lock node
                    }
                }
                locks_right[pos] = true;
            }
        }

        let (max_pos_1, max_gain_1) = gains1
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(i1, _), (i2, _)| gains1[*i1].partial_cmp(&gains1[*i2]).unwrap())
            .unwrap();

        let (max_pos_2, max_gain_2) = gains2
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(i1, _), (i2, _)| gains2[*i1].partial_cmp(&gains2[*i2]).unwrap())
            .unwrap();

        dbg!(max_gain_1);
        dbg!(max_gain_2);
        if max_gain_1 > 0. || max_gain_2 > 0. {
            if max_gain_1 > max_gain_2 {
                // flip node
                println!("flipping in left {}", max_pos_1);
                initial_partition[max_pos_1] = swap_id(initial_partition[max_pos_1]);
            } else {
                println!("flipping in right {}", max_pos_2);
                initial_partition[max_pos_2] = swap_id(initial_partition[max_pos_2]);
            }
            // break;
        }
    }
}

fn fm2(
    weights: &[f64],
    idx1: &mut [usize], // indices of elements in first part
    idx2: &mut [usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
    max_imbalance_per_iter: f64,
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

    let mut gains = gains(&indices, &indices, adjacency.view(), initial_partition);

    let mut cut_size = crate::topology::cut_size(adjacency.view(), initial_partition);
    let mut new_cut_size = cut_size;

    while new_cut_size <= cut_size {
        dbg!(new_cut_size);
        // update cut size
        cut_size = new_cut_size;

        let mut saves = Vec::new();
        let mut cut_saves = Vec::new();
        let mut locks = vec![false; indices.len()];

        // pass loop
        for _ in 0..indices.len() {
            // find max gain that is not locked
            let ((max_pos, max_gain), _) = indices
                .iter()
                .cloned()
                .zip(gains.iter().cloned())
                .zip(locks.iter().cloned())
                .filter(|(_, locked)| !locked)
                .max_by(|((_, g1), _), ((_, g2), _)| g1.partial_cmp(&g2).unwrap())
                .unwrap();

            // save movement
            saves.push((max_pos, max_gain));
            locks[max_pos] = true;
            dbg!(&locks
                .iter()
                .enumerate()
                .filter(|(_, lock)| **lock)
                .collect::<Vec<_>>());

            // update neighbors gain
            let row = adjacency.outer_view(max_pos).unwrap();
            for j in indices.iter().cloned() {
                if let Some(w) = row.get(j) {
                    // update gain of j
                    if initial_partition[max_pos] != initial_partition[j] {
                        gains[j] -= 2. * w;
                    } else {
                        gains[j] += 2. * w;
                    }
                }
            }

            eprintln!("Flipping pos {}", max_pos);
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
        for i in best_pos + 1..cut_saves.len() {
            let idx = indices[i];
            initial_partition[idx] = swap_id(initial_partition[idx]);
        }
        dbg!(&gains[0..=best_pos]);
        // dbg!(&gains);
        new_cut_size = best_cut;
    }
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
