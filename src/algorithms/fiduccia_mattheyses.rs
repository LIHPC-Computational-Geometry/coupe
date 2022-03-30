use super::Error;
use itertools::Itertools;
use sprs::CsMatView;

fn fiduccia_mattheyses<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<f64>,
    max_passes: usize,
    max_flips_per_pass: usize,
    max_imbalance: Option<f64>,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) where
    W: std::fmt::Debug + Copy + PartialOrd,
    W: std::iter::Sum + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W> + num::Zero,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());

    let part_count = 1 + *partition.iter().max().unwrap();
    let mut parts_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());
    let total_weight: W = parts_weights.iter().cloned().sum();
    let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
    let max_imbalance = max_imbalance.unwrap_or_else(|| {
        parts_weights
            .iter()
            .map(|part_weight| {
                (part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight
            })
            .minmax()
            .into_option()
            .unwrap()
            .1
    });
    tracing::info!(?max_imbalance);

    let mut best_cut_size = crate::topology::cut_size(adjacency, partition);
    tracing::info!("Initial cut size: {}", best_cut_size);

    for _ in 0..max_passes {
        // monitors for each pass the number of subsequent flips
        // that increase cut size. It may be beneficial in some
        // situations to allow a certain amount of them. Performing bad flips can open
        // up new sequences of good flips.
        let mut num_bad_move = 0;

        // Avoid copying partition arrays around and instead record an history
        // of flips.
        let mut flip_history = Vec::new();
        let mut cut_imb_history = Vec::new();

        use std::collections::BinaryHeap;

        #[derive(Clone, Copy)]
        struct Flip {
            target_part: usize,
            gain: f64,
        }

        impl PartialEq for Flip {
            fn eq(&self, other: &Self) -> bool {
                self.gain == other.gain
            }
        }
        impl Eq for Flip {}

        impl PartialOrd for Flip {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                f64::partial_cmp(&self.gain, &other.gain)
            }
        }

        impl Ord for Flip {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                Flip::partial_cmp(self, other).unwrap()
            }
        }

        /// Gain table element.
        enum NodeGains {
            /// The edge-cut gain should the node move to part P, for each P.
            Available(BinaryHeap<Flip>),
            /// The node is locked and will not be visited again this pass.
            Locked,
        }

        impl NodeGains {
            pub fn unlock(&self) -> Option<&BinaryHeap<Flip>> {
                match self {
                    NodeGains::Available(gains) => Some(gains),
                    NodeGains::Locked => None,
                }
            }

            pub fn max_gain(&self) -> Option<Flip> {
                self.unlock().and_then(|gains| gains.peek().cloned())
            }
        }

        // Gain table, stored as a heap of node gains for each node.
        let mut gains: Vec<NodeGains> = partition
            .iter()
            .enumerate()
            .map(|(node, &initial_part)| {
                let gains = (0..part_count)
                    .map(|target_part| {
                        let gain = if target_part == initial_part {
                            0.0
                        } else {
                            adjacency
                                .outer_view(node)
                                .unwrap()
                                .iter()
                                .map(|(neighbor, &edge_weight)| {
                                    if partition[neighbor] == initial_part {
                                        -edge_weight
                                    } else if partition[neighbor] == target_part {
                                        edge_weight
                                    } else {
                                        0.0
                                    }
                                })
                                .sum()
                        };
                        Flip { target_part, gain }
                    })
                    .collect();
                NodeGains::Available(gains)
            })
            .collect();

        // enter pass loop
        // The number of iteration of the pas loop is at most the
        // number of nodes in the mesh. However, if too many subsequent
        // bad flips are performed, the loop will break early
        for _ in 0..max_flips_per_pass {
            // find max gain and target part
            let (max_pos, flip) = gains
                .iter()
                .enumerate()
                .filter_map(|(node, gains)| gains.max_gain().map(|flip| (node, flip)))
                // get max gain of max gains computed for each node
                .max_by(|(_, flip1), (_, flip2)| {
                    f64::partial_cmp(&flip1.gain, &flip2.gain).unwrap()
                })
                .unwrap();
            let max_gain = flip.gain;
            let target_part = flip.target_part;

            if max_gain <= 0. {
                if num_bad_move >= max_bad_move_in_a_row {
                    tracing::info!("reached max bad move in a row");
                    break;
                }
                num_bad_move += 1;
            } else {
                // a good move breaks the bad moves sequence
                num_bad_move = 0;
            }

            // flip node
            let old_part = std::mem::replace(&mut partition[max_pos], target_part);
            flip_history.push((max_pos, old_part, target_part));
            parts_weights[old_part] -= weights[max_pos];
            parts_weights[target_part] += weights[max_pos];
            gains[max_pos] = NodeGains::Locked;
            let imbalance = parts_weights
                .iter()
                .map(|part_weight| {
                    (part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight
                })
                .minmax()
                .into_option()
                .unwrap()
                .1;

            // save cut_size
            tracing::info!(?imbalance, max_pos, old_part, target_part, "flip");
            let mut new_cut_size = match cut_imb_history.last() {
                Some((cut_size, _imbalance)) => *cut_size,
                None => best_cut_size,
            };
            for (neighbors, edge_weight) in adjacency.outer_view(max_pos).unwrap().iter() {
                if partition[neighbors] == old_part {
                    new_cut_size += edge_weight;
                } else if partition[neighbors] == target_part {
                    new_cut_size -= edge_weight;
                }
            }
            debug_assert_eq!(
                new_cut_size,
                crate::topology::cut_size(adjacency, partition),
            );
            cut_imb_history.push((new_cut_size, imbalance));

            for (neighbor, _) in adjacency.outer_view(max_pos).unwrap().iter() {
                let mut update_gains_for = |node: usize| {
                    let gains = match &mut gains[node] {
                        NodeGains::Available(gains) => gains,
                        NodeGains::Locked => return,
                    };
                    let initial_part = partition[node];
                    *gains = (0..part_count)
                        .map(|target_part| {
                            let gain = if target_part == initial_part {
                                0.0
                            } else {
                                adjacency
                                    .outer_view(node)
                                    .unwrap()
                                    .iter()
                                    .map(|(neighbor, &edge_weight)| {
                                        if partition[neighbor] == initial_part {
                                            -edge_weight
                                        } else if partition[neighbor] == target_part {
                                            edge_weight
                                        } else {
                                            0.0
                                        }
                                    })
                                    .sum()
                            };
                            Flip { target_part, gain }
                        })
                        .collect();
                };
                update_gains_for(neighbor);
            }
        }

        let old_cut_size = best_cut_size;

        // lookup for best cut_size that respects max_imbalance
        let rewind_to = cut_imb_history
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, (_cut_size, imbalance))| *imbalance <= max_imbalance)
            .min_by(|(_, (cut_size1, _imb1)), (_, (cut_size2, _imb2))| {
                cut_size1.partial_cmp(cut_size2).unwrap()
            })
            .map_or(0, |(best_pos, (best_cut, _best_imb))| {
                best_cut_size = best_cut;
                best_pos + 1
            });

        tracing::info!(
            "rewinding flips from pos {} to pos {}",
            rewind_to,
            flip_history.len()
        );
        for (idx, old_part, target_part) in flip_history.drain(rewind_to..) {
            partition[idx] = old_part;
            parts_weights[old_part] += weights[idx];
            parts_weights[target_part] += weights[idx];
        }

        if old_cut_size <= best_cut_size {
            break;
        }
    }

    tracing::info!("final cut size: {}", best_cut_size);
}

/// FiducciaMattheyses
///
/// An implementation of the Fiduccia Mattheyses topologic algorithm
/// for graph partitioning. This implementation is an extension of the
/// original algorithm to handle partitioning into more than two parts.
///
/// This algorithm repeats an iterative pass during which a set of graph nodes are assigned to
/// a new part, reducing the overall cutsize of the partition. As opposed to the
/// Kernighan-Lin algorithm, during each pass iteration, only one node is flipped at a time.
/// The algorithm thus does not preserve partition weights balance and may produce an unbalanced
/// partition.
///
/// Original algorithm from "A Linear-Time Heuristic for Improving Network Partitions"
/// by C.M. Fiduccia and R.M. Mattheyses.
///
/// # Example
///
/// ```rust
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
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),
///     Point2D::new(3., 0.),
///     Point2D::new(0., 1.),
///     Point2D::new(1., 1.),
///     Point2D::new(2., 1.),
///     Point2D::new(3., 1.),
/// ];
/// let weights = [1.0; 8];
/// let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
/// let mut adjacency = CsMat::empty(sprs::CSR, 8);
/// adjacency.reserve_outer_dim(8);
/// eprintln!("shape: {:?}", adjacency.shape());
/// adjacency.insert(0, 1, 1.);
/// adjacency.insert(1, 2, 1.);
/// adjacency.insert(2, 3, 1.);
/// adjacency.insert(4, 5, 1.);
/// adjacency.insert(5, 6, 1.);
/// adjacency.insert(6, 7, 1.);
/// adjacency.insert(0, 4, 1.);
/// adjacency.insert(1, 5, 1.);
/// adjacency.insert(2, 6, 1.);
/// adjacency.insert(3, 7, 1.);
///
/// // symmetry
/// adjacency.insert(1, 0, 1.);
/// adjacency.insert(2, 1, 1.);
/// adjacency.insert(3, 2, 1.);
/// adjacency.insert(5, 4, 1.);
/// adjacency.insert(6, 5, 1.);
/// adjacency.insert(7, 6, 1.);
/// adjacency.insert(4, 0, 1.);
/// adjacency.insert(5, 1, 1.);
/// adjacency.insert(6, 2, 1.);
/// adjacency.insert(7, 3, 1.);
///
/// coupe::FiducciaMattheyses::default()
///     .partition(&mut partition, (adjacency.view(), &weights))
///     .unwrap();
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FiducciaMattheyses {
    pub max_passes: Option<usize>,
    pub max_flips_per_pass: Option<usize>,
    pub max_imbalance: Option<f64>,
    pub max_bad_move_in_a_row: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, f64>, &'a [W])> for FiducciaMattheyses
where
    W: std::fmt::Debug + Copy + PartialOrd + num::Zero,
    W: std::iter::Sum + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W>,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<f64>, &'a [W]),
    ) -> Result<Self::Metadata, Self::Error> {
        if part_ids.is_empty() {
            return Ok(());
        }
        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.rows() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.rows(),
            });
        }
        if part_ids.len() != adjacency.cols() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.cols(),
            });
        }
        fiduccia_mattheyses(
            part_ids,
            weights,
            adjacency,
            self.max_passes.unwrap_or(usize::MAX),
            self.max_flips_per_pass.unwrap_or(usize::MAX),
            self.max_imbalance,
            self.max_bad_move_in_a_row,
        );
        Ok(())
    }
}
