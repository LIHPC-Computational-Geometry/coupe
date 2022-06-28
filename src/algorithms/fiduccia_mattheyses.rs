use super::try_from_f64;
use super::try_to_f64;
use super::Error;
use crate::vec::VecExt as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMatView;
use std::collections::HashSet;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

/// Diagnostic data for a Fiduccia-Mattheyses run.
#[non_exhaustive]
#[derive(Debug, Default)]
pub struct Metadata {
    /// Move count for each pass, included discarded moves by history rewinds.
    pub moves_per_pass: Vec<usize>,

    /// Number of moves that have been discarded for each pass.
    pub rewinded_moves_per_pass: Vec<usize>,
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
    adjacency: CsMatView<'_, i64>,
    max_passes: usize,
    max_moves_per_pass: usize,
    max_imbalance: Option<f64>,
    max_bad_moves_in_a_row: usize,
) -> Result<Metadata, Error>
where
    W: FmWeight,
{
    if partition.is_empty() {
        return Ok(Metadata::default());
    }
    if partition.len() != weights.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    if partition.len() != adjacency.rows() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: adjacency.rows(),
        });
    }
    if partition.len() != adjacency.cols() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: adjacency.cols(),
        });
    }

    let part_count = 1 + *partition.iter().max().unwrap();
    if 2 < part_count {
        return Err(Error::BiPartitioningOnly);
    }

    let mut part_weights =
        crate::imbalance::compute_parts_load(partition, 2, weights.par_iter().cloned());

    // Enforce part weights to be below this value.
    let max_part_weight = match max_imbalance {
        Some(max_imbalance) => {
            let total_weight: W = part_weights.iter().cloned().sum();
            let ideal_part_weight = try_to_f64(&total_weight)? / 2.0;
            try_from_f64(ideal_part_weight + max_imbalance * ideal_part_weight)?
        }
        None => *part_weights.iter().max_by(crate::partial_cmp).unwrap(),
    };

    let mut best_edge_cut = crate::topology::edge_cut(adjacency, partition);
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
        Vec::try_filled(HashSet::new(), possible_gain_count)?.into_boxed_slice()
    };
    // Maps a gain value to its index in gain_to_vertex.
    let gain_table_idx = |gain: i64| (gain + max_possible_gain) as usize;
    // Either Some(gain) or None if the vertex is locked.
    let mut vertex_to_gain = Vec::try_filled(None, partition.len())?.into_boxed_slice();

    let mut moves_per_pass = Vec::new();
    let mut rewinded_moves_per_pass = Vec::new();

    for _ in 0..max_passes {
        let old_edge_cut = best_edge_cut;
        let mut current_edge_cut = best_edge_cut;
        let mut move_with_best_edge_cut = None;

        // monitors for each pass the number of subsequent moves that increase
        // the edge cut. It may be beneficial in some situations to allow a
        // certain amount of them. Performing bad moves can open up new
        // sequences of good moves.
        let mut num_bad_move = 0;

        // Avoid copying partition arrays around and instead record an history
        // of moves, so that even if bad moves are performed during the pass, we
        // can look back and pick the best partition.
        let mut move_history: Vec<Move> = Vec::new();

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
        // bad moves are performed, the loop will break early
        for move_num in 0..max_moves_per_pass {
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
                            crate::partial_cmp(max_part_weight0, max_part_weight1)
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
            move_history.try_push(Move {
                vertex: moved_vertex,
                initial_part,
            })?;
            tracing::info!(moved_vertex, initial_part, target_part, "moved vertex");

            current_edge_cut -= move_gain;
            debug_assert_eq!(
                current_edge_cut,
                crate::topology::edge_cut(adjacency, partition),
            );
            if current_edge_cut < best_edge_cut {
                best_edge_cut = current_edge_cut;
                move_with_best_edge_cut = Some(move_num);
            }

            for (neighbor, edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
                let outdated_gain = match vertex_to_gain[neighbor] {
                    Some(v) => v,
                    None => continue,
                };
                let updated_gain = if partition[neighbor] == initial_part {
                    outdated_gain + 2 * edge_weight
                } else {
                    outdated_gain - 2 * edge_weight
                };
                vertex_to_gain[neighbor] = Some(updated_gain);
                gain_to_vertex[gain_table_idx(outdated_gain)].remove(&neighbor);
                gain_to_vertex[gain_table_idx(updated_gain)].insert(neighbor);
            }
        }

        let rewind_to = match move_with_best_edge_cut {
            Some(v) => v + 1,
            None => 0,
        };

        moves_per_pass.try_push(move_history.len())?;
        rewinded_moves_per_pass.try_push(move_history.len() - rewind_to)?;

        tracing::info!("rewinding {} moves", move_history.len() - rewind_to);
        for Move {
            vertex,
            initial_part,
        } in move_history.drain(rewind_to..)
        {
            partition[vertex] = initial_part;
            part_weights[initial_part] += weights[vertex];
            part_weights[1 - initial_part] -= weights[vertex];
        }

        if old_edge_cut <= best_edge_cut {
            break;
        }
    }

    tracing::info!("final edge cut: {}", best_edge_cut);

    Ok(Metadata {
        moves_per_pass,
        rewinded_moves_per_pass,
    })
}

/// Trait alias for values accepted as weights by [FiducciaMattheyses].
pub trait FmWeight
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + num::FromPrimitive + num::ToPrimitive + num::Zero,
    Self: Sub<Output = Self> + AddAssign + SubAssign,
{
}

impl<T> FmWeight for T
where
    Self: Copy + std::fmt::Debug + Send + Sync,
    Self: Sum + PartialOrd + num::FromPrimitive + num::ToPrimitive + num::Zero,
    Self: Sub<Output = Self> + AddAssign + SubAssign,
{
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
/// iteration, only one vertex is moved at a time. The algorithm thus does not
/// preserve partition weights balance and may produce an unbalanced partition.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
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
/// let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
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
///     .partition(&mut partition, (adjacency.view(), &weights))?;
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Fiduccia, C. M., Mattheyses, R. M. (1982). A linear-time heuristic for
/// improving network partitions. *DAC'82: Proceeding of the 19th Design
/// Automation Conference*.
#[derive(Debug, Clone, Copy, Default)]
pub struct FiducciaMattheyses {
    /// If `Some(max)` then the algorithm will not do more than `max` passes.
    /// If `None` then it will stop on the first non-fruitful pass.
    pub max_passes: Option<usize>,

    /// If `Some(max)` then the algorithm will not do more than `max` moves per
    /// pass.  If `None` then passes will stop when no more vertices yield a
    /// positive gain, and no more bad moves can be made.
    pub max_moves_per_pass: Option<usize>,

    /// If `Some(max)` then the algorithm will not move vertices in ways that
    /// the imbalance goes over `max`.  If `None`, then it will default to the
    /// imbalance of the input partition.
    pub max_imbalance: Option<f64>,

    /// How many moves that yield negative gains can be made before a pass ends.
    pub max_bad_move_in_a_row: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, i64>, &'a [W])> for FiducciaMattheyses
where
    W: FmWeight,
{
    type Metadata = Metadata;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<'_, i64>, &'a [W]),
    ) -> Result<Self::Metadata, Self::Error> {
        fiduccia_mattheyses(
            part_ids,
            weights,
            adjacency,
            self.max_passes.unwrap_or(usize::MAX),
            self.max_moves_per_pass.unwrap_or(usize::MAX),
            self.max_imbalance,
            self.max_bad_move_in_a_row,
        )
    }
}
