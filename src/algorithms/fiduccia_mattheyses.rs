use super::Error;
use sprs::CsMatView;
use std::cmp::Ordering;
use std::collections::HashSet;

fn partial_cmp<W>(a: &W, b: &W) -> Ordering
where
    W: PartialOrd,
{
    if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// Some data used to rewind the partition array to a previous state, should the
/// edge cut in said state is better.
struct Move {
    /// The index of the vertex.
    vertex: usize,

    /// The index of the part the vertex was in before the move.
    ///
    /// The target part is 1 - initial_part, because there are only two parts.
    initial_part: usize,
}

fn fiduccia_mattheyses<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<i64>,
    max_passes: usize,
    max_moves_per_pass: usize,
    max_imbalance: Option<f64>,
    max_bad_moves_in_a_row: usize,
) where
    W: std::fmt::Debug + Copy + PartialOrd,
    W: std::iter::Sum + num::FromPrimitive + num::ToPrimitive + num::Zero,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W>,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());

    let part_count = 1 + *partition.iter().max().unwrap();
    debug_assert!(part_count <= 2);

    let mut part_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());

    // Enforce part weights to be below this value.
    let max_part_weight = match max_imbalance {
        Some(max_imbalance) => {
            let total_weight: W = part_weights.iter().cloned().sum();
            let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
            W::from_f64(ideal_part_weight + max_imbalance * ideal_part_weight).unwrap()
        }
        None => *part_weights.iter().max_by(partial_cmp).unwrap(),
    };

    let mut best_edge_cut = crate::topology::cut_size(adjacency, partition);
    tracing::info!("Initial edge cut: {}", best_edge_cut);

    let max_possible_gain = (0..partition.len())
        .map(|vertex| {
            adjacency
                .outer_view(vertex)
                .unwrap()
                .iter()
                .fold(0, |acc, (_, edge_weight)| acc + edge_weight)
        })
        .max()
        .unwrap();

    // Maps (-max_possible_gain..=max_possible_gain) to nodes that have that gain.
    let mut gain_to_vertex = {
        let possible_gain_count = (2 * max_possible_gain + 1) as usize;
        vec![HashSet::new(); possible_gain_count].into_boxed_slice()
    };
    // Maps a gain value to its index in gain_to_vertex.
    let gain_table_idx = |gain: i64| (gain + max_possible_gain) as usize;
    // Either Some(gain) or None if the vertex is locked.
    let mut vertex_to_gain = vec![None; adjacency.outer_dims()].into_boxed_slice();

    for _ in 0..max_passes {
        // monitors for each pass the number of subsequent moves that increase
        // the edge cut. It may be beneficial in some situations to allow a
        // certain amount of them. Performing bad moves can open up new
        // sequences of good moves.
        let mut num_bad_move = 0;

        // Avoid copying partition arrays around and instead record an history
        // of moves, so that even if bad moves are performed during the pass, we
        // can look back and pick the best partition.
        let mut move_history: Vec<Move> = Vec::new();
        let mut edge_cut_history: Vec<i64> = Vec::new();

        for set in &mut *gain_to_vertex {
            set.clear();
        }
        for (vertex, initial_part) in partition.iter().enumerate() {
            let gain = adjacency
                .outer_view(vertex)
                .unwrap()
                .iter()
                .map(|(neighbor, edge_weight)| {
                    if partition[neighbor] == *initial_part {
                        -*edge_weight
                    } else {
                        *edge_weight
                    }
                })
                .sum();
            vertex_to_gain[vertex] = Some(gain);
            gain_to_vertex[gain_table_idx(gain)].insert(vertex);
        }

        // enter pass loop
        // The number of iteration of the pas loop is at most the
        // number of vertices in the mesh. However, if too many subsequent
        // bad flips are performed, the loop will break early
        for _ in 0..max_moves_per_pass {
            let (moved_vertex, move_gain) = match gain_to_vertex
                .iter()
                .rev()
                .zip((-max_possible_gain..=max_possible_gain).rev())
                .find_map(|(vertices, gain)| {
                    let (best_vertex, _) = vertices
                        .iter()
                        .filter_map(|vertex| {
                            let weight = weights[*vertex];
                            let initial_part = partition[*vertex];
                            let target_part = 1 - initial_part;
                            let target_part_weight = part_weights[target_part] + weight;
                            if max_part_weight < target_part_weight {
                                return None;
                            }
                            Some((*vertex, target_part_weight))
                        })
                        .min_by(|(_, max_part_weight0), (_, max_part_weight1)| {
                            partial_cmp(max_part_weight0, max_part_weight1)
                        })?;
                    Some((best_vertex, gain))
                }) {
                Some(v) => v,
                None => break,
            };

            if move_gain <= 0 {
                if num_bad_move >= max_bad_moves_in_a_row {
                    tracing::info!("reached max bad move in a row");
                    break;
                }
                num_bad_move += 1;
            } else {
                // A good move breaks the bad moves sequence.
                num_bad_move = 0;
            }

            vertex_to_gain[moved_vertex] = None;
            gain_to_vertex[gain_table_idx(move_gain)].remove(&moved_vertex);

            let initial_part = partition[moved_vertex];
            let target_part = 1 - initial_part;
            partition[moved_vertex] = target_part;
            part_weights[initial_part] -= weights[moved_vertex];
            part_weights[target_part] += weights[moved_vertex];
            move_history.push(Move {
                vertex: moved_vertex,
                initial_part,
            });
            tracing::info!(moved_vertex, initial_part, target_part, "moved vertex");

            let mut new_edge_cut = *edge_cut_history.last().unwrap_or(&best_edge_cut);
            for (neighbor, edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
                if partition[neighbor] == initial_part {
                    new_edge_cut += edge_weight;
                } else if partition[neighbor] == target_part {
                    new_edge_cut -= edge_weight;
                }
            }
            debug_assert_eq!(
                new_edge_cut,
                crate::topology::cut_size(adjacency, partition),
            );
            edge_cut_history.push(new_edge_cut);

            for (neighbor, edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
                let outdated_gain = match vertex_to_gain[neighbor] {
                    Some(v) => v,
                    None => continue,
                };
                let updated_gain = if partition[neighbor] == initial_part {
                    outdated_gain + edge_weight
                } else {
                    outdated_gain - edge_weight
                };
                vertex_to_gain[neighbor] = Some(updated_gain);
                gain_to_vertex[gain_table_idx(outdated_gain)].remove(&neighbor);
                gain_to_vertex[gain_table_idx(updated_gain)].insert(neighbor);
            }
        }

        let old_edge_cut = best_edge_cut;

        // Rewind history of moves to the best edge cut found in the pass.
        let (best_pos, best_cut) = match edge_cut_history
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| i64::cmp(a, b))
        {
            Some(v) => v,
            None => break,
        };

        tracing::info!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            move_history.len()
        );
        for Move {
            vertex,
            initial_part,
        } in move_history.drain(best_pos + 1..)
        {
            partition[vertex] = initial_part;
            part_weights[initial_part] += weights[vertex];
            part_weights[1 - initial_part] += weights[vertex];
        }

        best_edge_cut = best_cut;

        if old_edge_cut <= best_edge_cut {
            break;
        }
    }

    tracing::info!("final edge cut: {}", best_edge_cut);
}

/// FiducciaMattheyses
///
/// An implementation of the Fiduccia Mattheyses topologic algorithm
/// for graph partitioning. This implementation is an extension of the
/// original algorithm to handle partitioning into more than two parts.
///
/// This algorithm repeats an iterative pass during which a set of graph
/// vertices are assigned to a new part, reducing the overall cutsize of the
/// partition. As opposed to the Kernighan-Lin algorithm, during each pass
/// iteration, only one vertex is flipped at a time. The algorithm thus does not
/// preserve partition weights balance and may produce an unbalanced partition.
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
    W: std::iter::Sum + num::FromPrimitive + num::ToPrimitive,
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
