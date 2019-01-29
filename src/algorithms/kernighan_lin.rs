use crate::geometry::PointND;
use crate::ProcessUniqueId;
use sprs::CsMatView;

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::U1;

use itertools::Itertools;
use rayon::prelude::*;

pub fn kernighan_lin<D>(
    points: &[PointND<D>],
    weights: &[f64],
    adjacency: CsMatView<f64>,
    initial_partition: &mut [ProcessUniqueId],
    num_iter: usize,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
{
    // check there are only 2 parts in the partition
    assert_eq!(num_unique_elements(initial_partition), 2);
    let initial_cut = cut_size(adjacency.clone(), initial_partition);

    for _ in 0..num_iter {
        // compute gain associated to each node of the graph
        // the gain is defined as the cut_size decrease if the node
        // is assigned to the other partition
        let mut gains = points
            .iter()
            .enumerate()
            .map(|(i, _)| {
                // the gain makes sense only if there exists an edge between
                // the current node and an other one.
                // if no such edge exists, then flipping the node won't affect
                // the cutsize; the gain is thus 0
                adjacency
                    .outer_view(i)
                    .and_then(|row| {
                        Some(row.iter().fold(0., |acc, (j, w)| {
                            if initial_partition[i] != initial_partition[j] {
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
                        }))
                    })
                    .unwrap_or(0.)
            })
            .collect::<Vec<_>>();

        let (max_pos, max_gain) = gains
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, g1), (_, g2)| g1.partial_cmp(&g2).unwrap())
            .unwrap();

        // remove max gain
        // we cannot set it to 0 because the second
        // best gain may be below 0
        gains[max_pos] = std::f64::MIN;

        //get second max gain
        let (max_pos_2, _max_gain_2) = gains
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(i1, g1), (i2, g2)| {
                // if the second best gain is linked to the first one,
                // then the computation is wrong because the gain associated with
                // the edge linking the two nodes of best gain will be counted twice
                // whereas it should be null
                let g1 = if let Some(w) = adjacency.get(max_pos, *i1) {
                    // check if partition is different
                    if initial_partition[max_pos] != initial_partition[*i1] {
                        *g1 - 2. * *w
                    } else {
                        *g1 + 2. * *w
                    }
                } else {
                    *g1
                };
                let g2 = if let Some(w) = adjacency.get(max_pos, *i2) {
                    // check if partition is different
                    if initial_partition[max_pos] != initial_partition[*i2] {
                        *g2 - 2. * *w
                    } else {
                        *g2 + 2. * *w
                    }
                } else {
                    *g2
                };
                g1.partial_cmp(&g2).unwrap()
            })
            .unwrap();

        // if max_gain <= 0 then any partition modification will
        // increase the cut size
        if max_gain <= 0. {
            break;
        }

        // check they are in different partitions
        assert_ne!(initial_partition[max_pos], initial_partition[max_pos_2]);
        initial_partition.swap(max_pos, max_pos_2);
    }
    let final_cut = cut_size(adjacency.clone(), initial_partition);
    println!("final cut = {}", final_cut);
    println!("overall gain: {}", initial_cut - final_cut);
}

fn cut_size(adjacency: CsMatView<f64>, partition: &[ProcessUniqueId]) -> f64 {
    let mut cut_size = 0.;
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, w) in row.iter() {
            // graph edge are present twice in the matrix be cause of symetry
            if j >= i {
                continue;
            }
            if partition[i] != partition[j] {
                cut_size += w;
            }
        }
    }
    cut_size
}

fn num_unique_elements<T>(slice: &[T]) -> usize
where
    T: std::hash::Hash + std::cmp::Eq,
{
    slice.iter().unique().count()
}
