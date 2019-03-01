//! Implementation of the Kernighan-Lin algorithm for graph partitioning improvement.
//!
//! At each iteration, two nodes of different partition will be swapped, decreasing the overall cutsize
//! of the partition. The swap is performed in such a way that the added partition imbalanced is controlled.

use crate::geometry::PointND;
use crate::partition::Partition;
use crate::ProcessUniqueId;
use sprs::CsMatView;

use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
use nalgebra::DimName;

use rayon::prelude::*;

pub(crate) fn kernighan_lin<'a, D>(
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

    for (p, q) in adjacent_parts {
        kernighan_lin_2(
            weights,
            &p,
            &q,
            adjacency.view(),
            ids,
            num_iter,
            max_imbalance_per_iter,
        );
    }
}

// kernighan-lin with only a partition of two parts
fn kernighan_lin_2<D>(
    weights: &[f64],
    idx1: &[usize], // indices of elements in first part
    idx2: &[usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
    max_imbalance_per_iter: f64,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    for _ in 0..num_iter {
        // compute gain associated to each node of the graph
        // the gain is defined as the cut_size decrease if the node
        // is assigned to the other partition

        // first compute gains for elements in first part
        let mut gains1 = idx1
            .par_iter()
            .map(|i| {
                // the gain makes sense only if there exists an edge between
                // the current node and an other one.
                // if no such edge exists, then flipping the node won't affect
                // the cutsize; the gain is thus 0
                adjacency
                    .outer_view(*i)
                    .and_then(|row| {
                        Some(
                            idx2.iter()
                                .chain(idx1.iter())
                                .filter_map(|j| {
                                    row.nnz_index(*j)
                                        .and_then(|nnz_idx| Some((j, row[nnz_idx])))
                                })
                                .fold(0., |acc, (j, w)| {
                                    if initial_partition[*i] != initial_partition[*j] {
                                        // here an edge linking two nodes of different partitions
                                        // will then link two nodes of same partitions
                                        // hence the gain increase
                                        acc + w
                                    } else {
                                        // here an edge linking two nodes of same partitions
                                        // will then link two nodes of different partitions
                                        // hence the gain increase
                                        acc - w
                                    }
                                }),
                        )
                    })
                    .unwrap_or(0.)
            })
            .collect::<Vec<_>>();

        // compute gains for elements in second partition
        let gains2 = idx2
            .par_iter()
            .map(|i| {
                // the gain makes sense only if there exists an edge between
                // the current node and an other one.
                // if no such edge exists, then flipping the node won't affect
                // the cutsize; the gain is thus 0
                adjacency
                    .outer_view(*i)
                    .and_then(|row| {
                        Some(
                            idx1.iter()
                                .chain(idx2.iter())
                                .filter_map(|j| {
                                    row.nnz_index(*j)
                                        .and_then(|nnz_idx| Some((j, row[nnz_idx])))
                                })
                                .fold(0., |acc, (j, w)| {
                                    if initial_partition[*i] != initial_partition[*j] {
                                        // here an edge linking two nodes of different partitions
                                        // will then link two nodes of same partitions
                                        // hence the gain increase
                                        acc + w
                                    } else {
                                        // here an edge linking two nodes of same partitions
                                        // will then link two nodes of different partitions
                                        // hence the gain increase
                                        acc - w
                                    }
                                }),
                        )
                    })
                    .unwrap_or(0.)
            })
            .collect::<Vec<_>>();

        let (max_pos_1, max_gain_1) = gains1
            .par_iter()
            .cloned()
            .enumerate()
            .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
            .unwrap();

        // remove max gain
        // we cannot set it to 0 because the second
        // best gain may be below 0
        gains1[max_pos_1] = std::f64::MIN;

        //get second max gain
        let (max_pos_2, _max_gain_2) = gains2
            .par_iter()
            .cloned()
            .enumerate()
            .max_by(|(i1, g1), (i2, g2)| {
                // if the second best gain is linked to the first one,
                // then the computation is wrong because the gain associated with
                // the edge linking the two nodes of best gain will be counted twice
                // whereas it should be null
                let g1 = if initial_partition[idx1[max_pos_1]] == initial_partition[idx2[*i1]] {
                    std::f64::MIN
                } else if let Some(w) = adjacency.get(max_pos_1, *i1) {
                    *g1 - 2. * *w
                } else {
                    *g1
                };

                let g2 = if initial_partition[idx1[max_pos_1]] == initial_partition[idx2[*i2]] {
                    std::f64::MIN
                } else if let Some(w) = adjacency.get(max_pos_1, *i2) {
                    *g2 - 2. * *w
                } else {
                    *g2
                };

                g1.partial_cmp(&g2).unwrap()
            })
            .unwrap();

        // if max_gain <= 0 then any partition modification will
        // increase the cut size
        if max_gain_1 <= 0. {
            break;
        }

        let imbalance = (weights[max_pos_1] - weights[max_pos_2]).abs();

        if imbalance > max_imbalance_per_iter {
            // we should handle this differently
            // e.g. by looking for other possible gains that would fit within
            // the the imbalance limit
            break;
        }

        // check they are in different partitions
        // assert_ne!(initial_partition[max_pos_1], initial_partition[max_pos_2]);
        initial_partition.swap(idx1[max_pos_1], idx2[max_pos_2]);
    }
}

fn kl2<D>(
    weights: &[f64],
    idx1: &[usize], // indices of elements in first part
    idx2: &[usize], // indices of elements in second part
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
    max_imbalance_per_iter: f64,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
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
