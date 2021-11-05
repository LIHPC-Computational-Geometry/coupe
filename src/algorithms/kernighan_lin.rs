//! Implementation of the Kernighan-Lin algorithm for graph partitioning improvement.
//!
//! At each iteration, two nodes of different partition will be swapped, decreasing the overall cutsize
//! of the partition. The swap is performed in such a way that the added partition imbalanced is controlled.

use crate::geometry::PointND;
use crate::partition::Partition;

use itertools::Itertools;
use sprs::CsMatView;

pub(crate) fn kernighan_lin<'a, const D: usize>(
    initial_partition: &mut Partition<'a, PointND<D>, f64>,
    adjacency: CsMatView<f64>,
    max_passes: impl Into<Option<usize>>,
    max_flips_per_pass: impl Into<Option<usize>>,
    max_imbalance_per_flip: impl Into<Option<f64>>,
    max_bad_move_in_a_row: usize,
) {
    // To adapt Kernighan-Lin to a partition of more than 2 parts,
    // we apply the algorithm to each pair of adjacent parts (two parts
    // are adjacent if there exists an element in one part that is linked to
    // an element in the other part).

    let max_passes = max_passes.into();
    let max_flips_per_pass = max_flips_per_pass.into();
    let max_imbalance_per_flip = max_imbalance_per_flip.into();
    let (_points, weights, ids) = initial_partition.as_raw_mut();

    kernighan_lin_2_impl::<D>(
        weights,
        adjacency.view(),
        ids,
        max_passes,
        max_flips_per_pass,
        max_imbalance_per_flip,
        max_bad_move_in_a_row,
    );
}

fn kernighan_lin_2_impl<const D: usize>(
    weights: &[f64],
    adjacency: CsMatView<f64>,
    initial_partition: &mut [usize],
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    _max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize,
) {
    let unique_ids = initial_partition
        .iter()
        .cloned()
        .unique()
        .collect::<Vec<_>>();

    if unique_ids.len() != 2 {
        unimplemented!();
    }

    let mut cut_size = crate::topology::cut_size(adjacency.view(), initial_partition);
    println!("Initial cut size: {}", cut_size);
    let mut new_cut_size = cut_size;

    for iter in 0.. {
        if let Some(max_passes) = max_passes {
            if iter >= max_passes {
                break;
            }
        }

        cut_size = new_cut_size;

        // let imbalance;
        let num_bad_move = 0;

        let mut saves = Vec::new(); // flip save
        let mut cut_saves = Vec::new(); // cut size save
                                        // let mut ids_before_flip = Vec::new(); // the target id for reverting a flip

        let mut gains: Vec<f64> = vec![0.; initial_partition.len()];
        let mut locks = vec![false; initial_partition.len()];

        // pass loop
        for _ in 0..(initial_partition.len() / 2).min(max_flips_per_pass.unwrap_or(std::usize::MAX))
        {
            // construct gains
            for (idx, gain) in gains.iter_mut().enumerate() {
                for (j, w) in adjacency.outer_view(idx).unwrap().iter() {
                    if initial_partition[idx] == initial_partition[j] {
                        *gain -= w;
                    } else {
                        *gain += w;
                    }
                }
            }

            // find max gain for first part
            let (max_pos_1, max_gain_1) = gains
                .iter()
                .zip(locks.iter())
                .zip(weights.iter())
                .enumerate()
                .filter(|(idx, ((_, locked), _weight))| {
                    initial_partition[*idx] == unique_ids[0] && !**locked
                })
                .map(|(idx, ((gain, _), _))| (idx, *gain))
                .max_by(|(_, g1), (_, g2)| g1.partial_cmp(g2).unwrap())
                .unwrap();

            // update gain of neighbors
            for (j, w) in adjacency.outer_view(max_pos_1).unwrap().iter() {
                if initial_partition[max_pos_1] == initial_partition[j] {
                    gains[j] += 2. * w;
                } else {
                    gains[j] -= 2. * w;
                }
            }

            // find max gain for second part
            let (max_pos_2, max_gain_2) = gains
                .iter()
                .zip(locks.iter())
                .zip(weights.iter())
                .enumerate()
                .filter(|(idx, ((_, locked), _weight))| {
                    initial_partition[*idx] == unique_ids[1] && !**locked
                })
                .map(|(idx, ((gain, _), _))| (idx, *gain))
                .max_by(|(_, g1), (_, g2)| g1.partial_cmp(g2).unwrap())
                .unwrap();

            let total_gain = max_gain_1 + max_gain_2;

            if total_gain <= 0. && num_bad_move >= max_bad_move_in_a_row {
                println!("readched max bad move in a row");
                break;
            }

            locks[max_pos_1] = true;
            locks[max_pos_2] = true;

            // save flip
            saves.push(((max_pos_1, max_gain_1), (max_pos_2, max_gain_2)));

            // swap nodes
            initial_partition.swap(max_pos_1, max_pos_2);

            // save cut size
            cut_saves.push(crate::topology::cut_size(
                adjacency.view(),
                initial_partition,
            ));
        }

        // lookup for best cutsize
        let (best_pos, best_cut) = cut_saves
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // rewind swaps
        println!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            cut_saves.len()
        );

        for save in saves[best_pos + 1..cut_saves.len()].iter() {
            let ((idx_1, _), (idx_2, _)) = *save;
            initial_partition.swap(idx_1, idx_2);
        }

        new_cut_size = best_cut;

        if new_cut_size >= cut_size {
            break;
        }
    }
    println!("final cut size: {}", new_cut_size);
}
