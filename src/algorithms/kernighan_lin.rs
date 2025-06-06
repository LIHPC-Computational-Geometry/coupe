//! Implementation of the Kernighan-Lin algorithm for graph partitioning improvement.
//!
//! At each iteration, two nodes of different partition will be swapped, decreasing the overall cutsize
//! of the partition. The swap is performed in such a way that the added partition imbalanced is controlled.

use crate::topology::Topology;
use itertools::Itertools;

fn kernighan_lin<T>(
    part_ids: &mut [usize],
    weights: &[f64],
    adjacency: T,
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize,
) where
    T: Topology<f64> + Sync,
{
    // To adapt Kernighan-Lin to a partition of more than 2 parts,
    // we apply the algorithm to each pair of adjacent parts (two parts
    // are adjacent if there exists an element in one part that is linked to
    // an element in the other part).

    kernighan_lin_2_impl(
        part_ids,
        weights,
        adjacency,
        max_passes,
        max_flips_per_pass,
        max_imbalance_per_flip,
        max_bad_move_in_a_row,
    );
}

fn kernighan_lin_2_impl<T>(
    initial_partition: &mut [usize],
    weights: &[f64],
    adjacency: T,
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    _max_imbalance_per_flip: Option<f64>,
    max_bad_move_in_a_row: usize,
) where
    T: Topology<f64> + Sync,
{
    let unique_ids = initial_partition
        .iter()
        .cloned()
        .unique()
        .collect::<Vec<_>>();

    if unique_ids.len() != 2 {
        unimplemented!();
    }

    let mut cut_size = adjacency.edge_cut(initial_partition);
    tracing::info!("Initial cut size: {}", cut_size);
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
        for _ in 0..(initial_partition.len() / 2).min(max_flips_per_pass.unwrap_or(usize::MAX))
        {
            // construct gains
            for (idx, gain) in gains.iter_mut().enumerate() {
                for (j, w) in adjacency.neighbors(idx) {
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
            for (j, w) in adjacency.neighbors(max_pos_1) {
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
                tracing::info!("readched max bad move in a row");
                break;
            }

            locks[max_pos_1] = true;
            locks[max_pos_2] = true;

            // save flip
            saves.push(((max_pos_1, max_gain_1), (max_pos_2, max_gain_2)));

            // swap nodes
            initial_partition.swap(max_pos_1, max_pos_2);

            // save cut size
            cut_saves.push(adjacency.edge_cut(initial_partition));
        }

        // lookup for best cutsize
        let (best_pos, best_cut) = cut_saves
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // rewind swaps
        tracing::info!(
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
    tracing::info!("final cut size: {}", new_cut_size);
}

/// KernighanLin algorithm
///
/// An implementation of the Kernighan Lin topologic algorithm
/// for graph partitioning. The current implementation currently only handles
/// partitioning a graph into two parts, as described in the original algorithm in
/// "An efficient heuristic procedure for partitioning graphs" by W. Kernighan and S. Lin.
///
/// The algorithms repeats an iterative pass during which several pairs of nodes have
/// their part assignment swapped in order to reduce the cutsize of the partition.
/// If all the nodes are equally weighted, the algorithm preserves the partition balance.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), std::convert::Infallible> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
/// use sprs::CsMat;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
/// let points = [
///      Point2D::new(0., 0.),
///      Point2D::new(1., 0.),
///      Point2D::new(2., 0.),
///      Point2D::new(3., 0.),
///      Point2D::new(0., 1.),
///      Point2D::new(1., 1.),
///      Point2D::new(2., 1.),
///      Point2D::new(3., 1.),
///  ];
///  let weights = [1.; 8];
///  let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// // 1 iteration
/// coupe::KernighanLin {
///     max_passes: Some(1),
///     max_flips_per_pass: Some(1),
///     max_imbalance_per_flip: None,
///     max_bad_move_in_a_row: 1,
/// }
/// .partition(&mut partition, (adjacency.view(), &weights))?;
///
/// assert_eq!(partition[5], 0);
/// assert_eq!(partition[6], 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct KernighanLin {
    pub max_passes: Option<usize>,
    pub max_flips_per_pass: Option<usize>,
    pub max_imbalance_per_flip: Option<f64>,
    pub max_bad_move_in_a_row: usize,
}

impl<'a, T> crate::Partition<(T, &'a [f64])> for KernighanLin
where
    T: Topology<f64> + Sync,
{
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (T, &'a [f64]),
    ) -> Result<Self::Metadata, Self::Error> {
        kernighan_lin(
            part_ids,
            weights,
            adjacency,
            self.max_passes,
            self.max_flips_per_pass,
            self.max_imbalance_per_flip,
            self.max_bad_move_in_a_row,
        );
        Ok(())
    }
}
