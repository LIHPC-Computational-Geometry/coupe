use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};
use rayon::prelude::*;
use sprs::CsMatView;

use crate::partition::Partition;
use crate::PointND;
use crate::ProcessUniqueId;

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
        fiduccia_mattheyses_2(
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
