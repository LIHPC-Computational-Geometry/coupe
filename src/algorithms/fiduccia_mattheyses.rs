use super::Error;
use itertools::Itertools as _;
use sprs::CsMatView;
use std::collections::HashSet;

fn fiduccia_mattheyses<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<i64>,
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
    debug_assert!(part_count <= 2);

    let mut parts_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());
    let total_weight: W = parts_weights.iter().cloned().sum();
    let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
    let part_imbalance =
        |part_weight: &W| (part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight;
    let max_imbalance = max_imbalance.unwrap_or_else(|| {
        parts_weights
            .iter()
            .map(part_imbalance)
            .minmax()
            .into_option()
            .unwrap()
            .1
    });
    tracing::info!(?max_imbalance);

    struct GainTable {
        gain_to_node: Box<[HashSet<usize>]>,
        node_to_gain: Box<[Option<i64>]>,
    }

    let pmax = (0..partition.len())
        .map(|node| {
            adjacency
                .outer_view(node)
                .unwrap()
                .iter()
                .fold(0, |acc, (_, edge_weight)| acc + edge_weight)
        })
        .max()
        .unwrap();

    let gain_count = (2 * pmax + 1) as usize;
    let gain_to_node = vec![HashSet::new(); gain_count];
    let node_to_gain = vec![None; adjacency.outer_dims()];

    let mut gain_table = GainTable {
        gain_to_node: gain_to_node.into_boxed_slice(),
        node_to_gain: node_to_gain.into_boxed_slice(),
    };
    let gain_table_idx = |gain: i64| (gain + pmax) as usize;

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
        let mut cut_size_history = Vec::new();

        for set in &mut *gain_table.gain_to_node {
            set.clear();
        }
        for (node, initial_part) in partition.iter().enumerate() {
            let target_part = 1 - *initial_part;
            let gain = adjacency
                .outer_view(node)
                .unwrap()
                .iter()
                .map(|(neighbor, edge_weight)| {
                    if partition[neighbor] == *initial_part {
                        -*edge_weight
                    } else if partition[neighbor] == target_part {
                        *edge_weight
                    } else {
                        0
                    }
                })
                .sum();
            gain_table.node_to_gain[node] = Some(gain);
            gain_table.gain_to_node[gain_table_idx(gain)].insert(node);
        }

        // enter pass loop
        // The number of iteration of the pas loop is at most the
        // number of nodes in the mesh. However, if too many subsequent
        // bad flips are performed, the loop will break early
        for _ in 0..max_flips_per_pass {
            let (max_pos, max_gain) = match gain_table
                .gain_to_node
                .iter()
                .rev()
                .zip((-pmax..=pmax).rev())
                .find_map(|(nodes, gain)| {
                    let (best_node, _) = nodes
                        .iter()
                        .filter_map(|node| {
                            let weight = weights[*node];
                            let initial_part = partition[*node];
                            let target_part = 1 - initial_part;
                            let initial_part_weight = parts_weights[initial_part] - weight;
                            let target_part_weight = parts_weights[target_part] + weight;
                            let imbalance = [initial_part_weight, target_part_weight]
                                .iter()
                                .map(part_imbalance)
                                .minmax()
                                .into_option()
                                .unwrap()
                                .1;
                            if max_imbalance < imbalance {
                                return None;
                            }
                            Some((*node, imbalance))
                        })
                        .min_by(|(_, imbalance0), (_, imbalance1)| {
                            f64::partial_cmp(imbalance0, imbalance1).unwrap()
                        })?;
                    Some((best_node, gain))
                }) {
                Some(v) => v,
                None => break,
            };

            gain_table.node_to_gain[max_pos] = None;
            gain_table.gain_to_node[gain_table_idx(max_gain)].remove(&max_pos);
            let target_part = 1 - partition[max_pos];

            if max_gain <= 0 {
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

            // save cut_size
            tracing::info!(max_pos, old_part, target_part, "flip");
            let mut new_cut_size = *cut_size_history.last().unwrap_or(&best_cut_size);
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
            cut_size_history.push(new_cut_size);

            for (neighbor, _) in adjacency.outer_view(max_pos).unwrap().iter() {
                let initial_part = partition[neighbor];
                let target_part = 1 - initial_part;
                let outdated_gain = match gain_table.node_to_gain[neighbor] {
                    Some(v) => v,
                    None => continue,
                };
                let updated_gain = adjacency
                    .outer_view(neighbor)
                    .unwrap()
                    .iter()
                    .map(|(neighbor, edge_weight)| {
                        if partition[neighbor] == initial_part {
                            -*edge_weight
                        } else if partition[neighbor] == target_part {
                            *edge_weight
                        } else {
                            0
                        }
                    })
                    .sum();
                if outdated_gain != updated_gain {
                    gain_table.gain_to_node[gain_table_idx(outdated_gain)].remove(&neighbor);
                    gain_table.gain_to_node[gain_table_idx(updated_gain)].insert(neighbor);
                }
            }
        }

        let old_cut_size = best_cut_size;

        // lookup for best cutsize
        let (best_pos, best_cut) = match cut_size_history
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some(v) => v,
            None => break,
        };

        tracing::info!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            flip_history.len()
        );
        for (idx, old_part, target_part) in flip_history.drain(best_pos + 1..) {
            partition[idx] = old_part;
            parts_weights[old_part] += weights[idx];
            parts_weights[target_part] += weights[idx];
        }

        best_cut_size = best_cut;

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
/// adjacency.insert(0, 1, 1);
/// adjacency.insert(1, 2, 1);
/// adjacency.insert(2, 3, 1);
/// adjacency.insert(4, 5, 1);
/// adjacency.insert(5, 6, 1);
/// adjacency.insert(6, 7, 1);
/// adjacency.insert(0, 4, 1);
/// adjacency.insert(1, 5, 1);
/// adjacency.insert(2, 6, 1);
/// adjacency.insert(3, 7, 1);
///
/// // symmetry
/// adjacency.insert(1, 0, 1);
/// adjacency.insert(2, 1, 1);
/// adjacency.insert(3, 2, 1);
/// adjacency.insert(5, 4, 1);
/// adjacency.insert(6, 5, 1);
/// adjacency.insert(7, 6, 1);
/// adjacency.insert(4, 0, 1);
/// adjacency.insert(5, 1, 1);
/// adjacency.insert(6, 2, 1);
/// adjacency.insert(7, 3, 1);
///
/// // Set the imbalance tolerance to 25% to provide enough room for FM to do
/// // the swap.
/// coupe::FiducciaMattheyses { max_imbalance: Some(0.25), ..Default::default() }
///     .partition(&mut partition, (adjacency.view(), &weights))
///     .unwrap();
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// ```
///
/// # Reference
///
/// Fiduccia, C. M., Mattheyses, R. M. (1982). A linear-time heuristic for
/// improving network partitions. *DAC'82: Proceeding of the 19th Design
/// Automation Conference*.
#[derive(Debug, Clone, Copy, Default)]
pub struct FiducciaMattheyses {
    pub max_passes: Option<usize>,
    pub max_flips_per_pass: Option<usize>,
    pub max_imbalance: Option<f64>,
    pub max_bad_move_in_a_row: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, i64>, &'a [W])> for FiducciaMattheyses
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
        (adjacency, weights): (CsMatView<i64>, &'a [W]),
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
        if 1 < *part_ids.iter().max().unwrap_or(&0) {
            return Err(Error::BiPartitioningOnly);
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
