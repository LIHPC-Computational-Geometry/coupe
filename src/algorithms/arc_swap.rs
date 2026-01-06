use super::Error;
use crate::Partition;
use crate::defer::defer;
use crate::partial_cmp;
use crate::topology::Topology;
use crate::work_share::work_share;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use num_traits::Zero;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::slice::ParallelSlice;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// Diagnostic data for a [ArcSwap] run.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy)]
pub struct Metadata {
    /// By how much the edge cut has been reduced by the algorithm.
    /// Positive values mean reduced edge cut.
    pub edge_cut_gain: i64,
    /// Number of passes.
    pub pass_count: usize,
    /// Number of attempts to move vertices.
    pub move_attempts: usize,
    /// Number of times vertices has been moved.
    pub move_count: usize,
    /// Number of times threads found out the vertex they picked had a locked
    /// neighbor.
    pub race_count: usize,
    /// Number of times threads have picked a locked vertex.
    pub locked_count: usize,
    /// Number of times threads have picked a vertex with negative gain.
    pub no_gain_count: usize,
    /// Number of times threads have picked a vertex that would disrupt balance.
    pub bad_balance_count: usize,
    /// Number of vertices distributed to each thread at the start of the run.
    pub vertices_per_thread: usize,
}

impl Metadata {
    fn merge(self, other: Self) -> Self {
        Self {
            edge_cut_gain: self.edge_cut_gain + other.edge_cut_gain,
            pass_count: self.pass_count + other.pass_count,
            move_attempts: self.move_attempts + other.move_attempts,
            move_count: self.move_count + other.move_count,
            race_count: self.race_count + other.race_count,
            locked_count: self.locked_count + other.locked_count,
            no_gain_count: self.no_gain_count + other.no_gain_count,
            bad_balance_count: self.bad_balance_count + other.bad_balance_count,
            vertices_per_thread: self.vertices_per_thread + other.vertices_per_thread,
        }
    }
}

fn arc_swap<W, T>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: T,
    part_count: usize,
    max_imbalance: Option<f64>,
) -> Metadata
where
    W: AsWeight,
    T: Topology<i64> + Sync,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.len());

    let compute_part_weights = || {
        let span = tracing::info_span!("compute part_weights and max_part_weight");
        let _enter = span.enter();

        let part_weights = crate::imbalance::compute_parts_load(
            partition,
            part_count,
            weights.par_iter().cloned(),
        );

        let total_weight: W = part_weights.iter().cloned().sum();

        // Enforce part weights to be below this value.
        let max_part_weight = match max_imbalance {
            Some(max_imbalance) => {
                let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
                W::from_f64(ideal_part_weight + max_imbalance * ideal_part_weight).unwrap()
            }
            None => *part_weights.iter().max_by(partial_cmp).unwrap(),
        };
        (part_weights, max_part_weight)
    };
    let compute_locks = || {
        let span = tracing::info_span!("compute locks");
        let _enter = span.enter();

        partition
            .par_iter()
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<AtomicBool>>()
    };

    let (items_per_thread, thread_count) =
        work_share(partition.len(), rayon::current_num_threads());
    let (locks, (mut part_weights, max_part_weight)) =
        rayon::join(compute_locks, compute_part_weights);

    let span = tracing::info_span!("doing moves");
    let _enter = span.enter();

    let partition = crate::as_atomic(partition);

    // This function makes move attempts until either
    // - `cut` is empty, or
    // - a move is successfully done.
    // In the later case, it adds relevant vertices to `cut`.
    // Finally it returns whether a move has been made.
    let make_move = |cut: &mut Vec<usize>,
                     part_weights: &mut [W],
                     metadata: &mut Metadata,
                     max_part_weights: &[W]|
     -> bool {
        let moved_vertex = loop {
            let vertex = match cut.pop() {
                Some(v) => v,
                None => return false,
            };
            metadata.move_attempts += 1;

            let locked = locks[vertex]
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err();
            if locked {
                metadata.locked_count += 1;
                continue;
            }
            let _lock_guard = defer({
                let locks = &locks;
                move || locks[vertex].store(false, Ordering::Release)
            });
            let raced = adjacency
                .neighbors(vertex)
                .any(|(neighbor, _edge_weight)| locks[neighbor].load(Ordering::Acquire));
            if raced {
                metadata.race_count += 1;
                continue;
            }

            let initial_part = partition[vertex].load(Ordering::Relaxed);

            let (target_part, gain) = (0..part_count)
                .filter(|target_part| *target_part != initial_part)
                .map(|target_part| {
                    let gain: i64 = adjacency
                        .neighbors(vertex)
                        .map(|(neighbor, edge_weight)| {
                            let part = partition[neighbor].load(Ordering::Relaxed);
                            if part == initial_part {
                                -edge_weight
                            } else if part == target_part {
                                edge_weight
                            } else {
                                0
                            }
                        })
                        .sum();
                    (target_part, gain)
                })
                .max_by(|(_, g1), (_, g2)| i64::cmp(g1, g2))
                .unwrap();
            if gain <= 0 {
                // Don't push it back now, it will when its gain is updated
                // positively (due to a neighbor moving).
                metadata.no_gain_count += 1;
                continue;
            }

            let weight = weights[vertex];
            let target_part_weight = weight + part_weights[target_part];
            if max_part_weights[target_part] < target_part_weight {
                metadata.bad_balance_count += 1;
                continue;
            }

            // Make the move.
            metadata.move_count += 1;
            metadata.edge_cut_gain += gain;
            part_weights[initial_part] -= weight;
            part_weights[target_part] += weight;
            partition[vertex].store(target_part, Ordering::Relaxed);

            break vertex;
        };

        for (neighbor, _edge_weight) in adjacency.neighbors(moved_vertex) {
            let neighbor_part = partition[neighbor].load(Ordering::Relaxed);
            let neighbor_gain: i64 = (0..part_count)
                .filter(|target_part| *target_part != neighbor_part)
                .map(|target_part| {
                    adjacency
                        .neighbors(neighbor)
                        .map(|(neighbor2, edge_weight)| {
                            let part = partition[neighbor2].load(Ordering::Relaxed);
                            if part == neighbor_part {
                                -edge_weight
                            } else if part == target_part {
                                edge_weight
                            } else {
                                0
                            }
                        })
                        .sum()
                })
                .max()
                .unwrap();
            if 0 < neighbor_gain {
                cut.push(neighbor);
            }
        }

        true
    };

    let mut metadata = Metadata::default();
    // Thread-local maximum of the weight of each part.
    // Each thread can move vertices as they wish as long as their local
    // `part_weights` remains lower than this array, element-by-element.
    // While this can be over-restrictive, it removes a RW lock and still
    // ensures the imbalance stays within bounds.
    let mut thread_max_pws = vec![W::zero(); part_count];
    loop {
        metadata.pass_count += 1;

        // `part_weights` changes at the end of each pass, so `thread_max_pws`
        // needs to be updated here.
        for (max_pw, pw) in thread_max_pws.iter_mut().zip(&part_weights) {
            *max_pw = *pw
                + W::from_f64((max_part_weight - *pw).to_f64().unwrap() / thread_count as f64)
                    .unwrap();
        }

        // The actual pass.
        let (pass_metadata, part_weights_sum) = partition
            .par_chunks(items_per_thread)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut cut = Vec::new();
                let mut part_weights = part_weights.clone();
                let mut metadata = Metadata::default();
                for (initial_part, vertex) in chunk.iter().zip(items_per_thread * chunk_idx..) {
                    let initial_part = initial_part.load(Ordering::Relaxed);
                    let on_cut = adjacency.neighbors(vertex).any(|(neighbor, _edge_weight)| {
                        partition[neighbor].load(Ordering::Relaxed) != initial_part
                    });
                    if !on_cut {
                        continue;
                    }
                    cut.push(vertex);
                    while make_move(&mut cut, &mut part_weights, &mut metadata, &thread_max_pws) {
                        // blank
                    }
                }
                (metadata, part_weights)
            })
            .reduce(
                || (Metadata::default(), vec![W::zero(); part_count]),
                |(metadata1, mut part_weights1), (metadata2, part_weights2)| {
                    let metadata = Metadata::merge(metadata1, metadata2);
                    // part weights are summed for the `part_weights` update
                    // further below.
                    for (pw1, pw2) in part_weights1.iter_mut().zip(part_weights2) {
                        *pw1 += pw2;
                    }
                    (metadata, part_weights1)
                },
            );

        // Update `part_weights` using the following formula:
        //
        //     PW <- PW + (tPW0 - PW) + ... + (tPWn - PW)
        // simplified to
        //     PW <- (sum_i tPWi) - (thread_count - 1) * PW
        //
        // where tPWi is the thread_local part-weights array.
        // I think the whole thing is correct because vertices are locked and no
        // two threads can do the same move at the same time.
        for (pw, pw_sum) in part_weights.iter_mut().zip(part_weights_sum) {
            *pw = pw_sum - W::from_usize(thread_count - 1).unwrap() * *pw;
        }

        metadata = metadata.merge(pass_metadata);
        if pass_metadata.edge_cut_gain == 0 {
            break;
        }
    }

    metadata.vertices_per_thread = items_per_thread;
    metadata
}

/// Trait alias for values accepted as weights by [ArcSwap].
pub trait AsWeight
where
    Self: Copy + PartialOrd + Send + Sync + Zero,
    Self: std::iter::Sum + FromPrimitive + ToPrimitive,
    Self: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = Self>,
    Self: std::ops::Mul<Output = Self> + std::ops::Div<Output = Self>,
{
}

impl<T> AsWeight for T
where
    Self: Copy + PartialOrd + Send + Sync + Zero,
    Self: std::iter::Sum + FromPrimitive + ToPrimitive,
    Self: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = Self>,
    Self: std::ops::Mul<Output = Self> + std::ops::Div<Output = Self>,
{
}

/// # Arc-swap
///
/// A multi-threaded variant of [Fiduccia-Mattheyses][fm], which moves
/// vertices from one part to the other as long as the move reduces the cutset.
///
/// In contrast to the original algorithm, it is not greedy and thus does more,
/// smaller moves. It also does not need to build up a gain table, which greatly
/// speeds up the execution time.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Grid;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
///
/// let width = std::num::NonZeroUsize::new(4).unwrap();
/// let height = std::num::NonZeroUsize::new(2).unwrap();
/// let grid = Grid::new_2d(width, height);
///
/// let weights = [1.0; 8];
/// let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
/// coupe::ArcSwap { max_imbalance: Some(0.25) }
///     .partition(&mut partition, (grid, &weights))?;
///
/// # Ok(())
/// # }
/// ```
///
/// [fm]: crate::FiducciaMattheyses
#[derive(Debug, Clone, Copy, Default)]
pub struct ArcSwap {
    /// Restrict moves made by ArcSwap such that the imbalance of the output
    /// partition is smaller than this value.
    ///
    /// - if `Some(value)`, ArcSwap reduces the edge cut as much as possible
    ///   while keeping imbalance below `value`.
    /// - if `None`, ArcSwap does not increase the imbalance of the input
    ///   partition (same behavior as `Some(imbalance(input_partition))`).
    pub max_imbalance: Option<f64>,
}

impl<'a, T, W> Partition<(T, &'a [W])> for ArcSwap
where
    W: AsWeight,
    T: Topology<i64> + Sync,
{
    type Metadata = Metadata;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (T, &'a [W]),
    ) -> Result<Self::Metadata, Self::Error> {
        if part_ids.is_empty() {
            return Ok(Metadata::default());
        }
        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.len(),
            });
        }
        let part_count = 1 + *part_ids.par_iter().max().unwrap_or(&0);
        let part_count = usize::max(2, part_count);
        Ok(arc_swap(
            part_ids,
            weights,
            adjacency,
            part_count,
            self.max_imbalance,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    proptest!(
        #[test]
        fn test_arcswap_no_max_imb(
            (width, height, part_count, mut partition, weights) in
                (2..40_usize, 2..40_usize, 2..10_usize)
                    .prop_flat_map(|(width, height, part_count)| (
                        Just(width),
                        Just(height),
                        Just(part_count),
                        prop::collection::vec(0..part_count, width * height),
                        prop::collection::vec(0.0..10.0_f64, width * height),
                    ))
        ) {
            prop_assume!({
                // Ensure the input partition is valid.
                let mut partition = partition.clone();
                partition.sort();
                partition.dedup();
                partition.len() == part_count
            });

            let width = NonZeroUsize::new(width).unwrap();
            let height = NonZeroUsize::new(height).unwrap();
            let grid = Grid::new_2d(width, height);

            let old_edge_cut: i64 = grid.edge_cut(&partition);
            let old_imbalance =
                crate::imbalance::imbalance(part_count, &partition, weights.par_iter().cloned());

            let metadata = ArcSwap { max_imbalance: None }
                .partition(&mut partition, (grid, &weights));
            prop_assert!(metadata.is_ok());
            let metadata = metadata.unwrap();

            let new_edge_cut: i64 = grid.edge_cut(&partition);
            let new_imbalance =
                crate::imbalance::imbalance(part_count, &partition, weights.par_iter().cloned());

            // Decreased edge cut.
            prop_assert_eq!(
                metadata.edge_cut_gain,
                old_edge_cut - new_edge_cut,
                "incorrect metadata\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );
            prop_assert!(
                metadata.edge_cut_gain >= 0,
                "worsened edge cut\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );

            // Imbalance is preserved (TODO fix 1.001* errors).
            prop_assert!(
                new_imbalance <= 1.000_000_1 * old_imbalance,
                "worsened imbalance\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );
        }
    );

    proptest!(
        #[test]
        fn test_arcswap_with_max_imb(
            (width, height, part_count, mut partition, weights, max_imbalance) in
                (2..40_usize, 2..40_usize, 2..10_usize, 0.0..1.0_f64)
                    .prop_flat_map(|(width, height, part_count, max_imbalance)| (
                        Just(width),
                        Just(height),
                        Just(part_count),
                        prop::collection::vec(0..part_count, width * height),
                        prop::collection::vec(0.0..10.0_f64, width * height),
                        Just(max_imbalance),
                    ))
        ) {
            prop_assume!({
                // Ensure the input partition is valid.
                let mut partition = partition.clone();
                partition.sort();
                partition.dedup();
                partition.len() == part_count
            });

            let width = NonZeroUsize::new(width).unwrap();
            let height = NonZeroUsize::new(height).unwrap();
            let grid = Grid::new_2d(width, height);

            let old_edge_cut: i64 = grid.edge_cut(&partition);
            let old_imbalance =
                crate::imbalance::imbalance(part_count, &partition, weights.par_iter().cloned());

            prop_assume!(old_imbalance < 1.0);

            let max_imbalance = old_imbalance + (1.0 - old_imbalance) * max_imbalance;
            let metadata = ArcSwap { max_imbalance: Some(max_imbalance) }
                .partition(&mut partition, (grid, &weights));
            prop_assert!(metadata.is_ok());
            let metadata = metadata.unwrap();

            let new_edge_cut: i64 = grid.edge_cut(&partition);
            let new_imbalance =
                crate::imbalance::imbalance(part_count, &partition, weights.par_iter().cloned());

            // Decreased edge cut.
            prop_assert_eq!(
                metadata.edge_cut_gain,
                old_edge_cut - new_edge_cut,
                "incorrect metadata\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );
            prop_assert!(
                metadata.edge_cut_gain >= 0,
                "worsened edge cut\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );

            // Imbalance is below threshold (TODO fix 1.001* errors).
            prop_assert!(
                new_imbalance <= 1.000_000_1 * max_imbalance,
                "worsened imbalance\nnew_partition: {:?}\nmetadata: {:?}\nold_imb: {}\nnew_imb: {}\nold_ec: {}\nnew_ec: {}",
                partition, metadata, old_imbalance, new_imbalance, old_edge_cut, new_edge_cut
            );
        }
    );
}
