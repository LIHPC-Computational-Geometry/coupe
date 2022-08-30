use super::Error;
use crate::defer::defer;
use crate::work_share::work_share;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMatView;
use std::cmp;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

fn partial_cmp<W>(a: &W, b: &W) -> cmp::Ordering
where
    W: PartialOrd,
{
    if a < b {
        cmp::Ordering::Less
    } else {
        cmp::Ordering::Greater
    }
}

/// Diagnostic data for a [ArcSwap] run.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy)]
pub struct Metadata {
    /// By how much the edge cut has been reduced by the algorithm.
    /// Positive values mean reduced edge cut.
    pub edge_cut_gain: i64,
    /// Number of times vertices has been moved.
    pub move_count: usize,
    /// Number of times threads found out the vertex they picked had a locked
    /// neighbor.
    pub race_count: usize,
    /// Number of attempts to move vertices.
    pub move_attempts: usize,
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
            move_count: self.move_count + other.move_count,
            race_count: self.race_count + other.race_count,
            move_attempts: self.move_attempts + other.move_attempts,
            locked_count: self.locked_count + other.locked_count,
            no_gain_count: self.no_gain_count + other.no_gain_count,
            bad_balance_count: self.bad_balance_count + other.bad_balance_count,
            vertices_per_thread: self.vertices_per_thread + other.vertices_per_thread,
        }
    }
}

fn arc_swap<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<'_, i64>,
    max_imbalance: Option<f64>,
) -> Metadata
where
    W: AsWeight,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());

    let compute_part_weights = |thread_count: usize| {
        let span = tracing::info_span!("compute part_weights and max_part_weight");
        let _enter = span.enter();

        let part_weights =
            crate::imbalance::compute_parts_load(partition, 2, weights.par_iter().cloned());

        let total_weight: W = part_weights.iter().cloned().sum();

        // Enforce part weights to be below this value.
        let max_part_weight = match max_imbalance {
            Some(max_imbalance) => {
                let max_imbalance = max_imbalance / thread_count as f64;
                let ideal_part_weight = total_weight.to_f64().unwrap() / 2.0;
                W::from_f64(ideal_part_weight + max_imbalance * ideal_part_weight).unwrap()
            }
            None => {
                let max_part_weight = *part_weights.iter().max_by(partial_cmp).unwrap();
                max_part_weight
                    + (max_part_weight + max_part_weight - total_weight)
                        / W::from_usize(2 * thread_count).unwrap()
            }
        };
        (total_weight, part_weights[0], max_part_weight)
    };
    let compute_locks = || {
        let span = tracing::info_span!("compute locks");
        let _enter = span.enter();

        partition
            .par_iter()
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<AtomicBool>>()
    };

    let (locks, (items_per_thread, total_weight, part0_weight, max_part_weight)) =
        rayon::join(compute_locks, || {
            let cut_len = partition
                .par_iter()
                .enumerate()
                .filter(|(vertex, initial_part)| {
                    adjacency
                        .outer_view(*vertex)
                        .unwrap()
                        .iter()
                        .any(|(neighbor, _edge_weight)| partition[neighbor] != **initial_part)
                })
                .count();
            let (items_per_thread, thread_count) =
                work_share(cut_len, rayon::current_num_threads());
            let (total_weight, part0_weight, max_part_weight) = compute_part_weights(thread_count);
            (
                items_per_thread,
                total_weight,
                part0_weight,
                max_part_weight,
            )
        });

    let span = tracing::info_span!("doing moves");
    let _enter = span.enter();

    let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    let make_move = |cut: &mut Vec<usize>, part0_weight: &mut W, metadata: &mut Metadata| -> bool {
        let (moved_vertex, move_gain, initial_part, _lock) = loop {
            let vertex = match cut.pop() {
                Some(v) => {
                    metadata.move_attempts += 1;
                    v
                }
                None => {
                    return false;
                }
            };

            let initial_part = partition[vertex].load(Ordering::Relaxed);
            let target_part = 1 - initial_part;
            let neighbors = adjacency.outer_view(vertex).unwrap();

            let gain: i64 = neighbors
                .iter()
                .map(|(neighbor, edge_weight)| {
                    if partition[neighbor].load(Ordering::Relaxed) == initial_part {
                        -*edge_weight
                    } else {
                        *edge_weight
                    }
                })
                .sum();
            if gain <= 0 {
                // Don't push it back now, it will when its gain is updated
                // positively (due to a neighbor moving).
                metadata.no_gain_count += 1;
                continue;
            }

            let weight = weights[vertex];
            let target_part_weight = weight
                + if target_part == 0 {
                    *part0_weight
                } else {
                    total_weight - *part0_weight
                };
            if max_part_weight < target_part_weight {
                metadata.bad_balance_count += 1;
                continue;
            }

            let locked = locks[vertex]
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err();
            if locked {
                metadata.locked_count += 1;
                continue;
            }
            let lock_guard = defer({
                let locks = &locks;
                move || locks[vertex].store(false, Ordering::Release)
            });
            let raced = neighbors
                .iter()
                .any(|(neighbor, _edge_weight)| locks[neighbor].load(Ordering::Acquire));
            if raced {
                metadata.race_count += 1;
                continue;
            }

            *part0_weight = if initial_part == 0 {
                *part0_weight - weight
            } else {
                *part0_weight + weight
            };

            break (vertex, gain, initial_part, lock_guard);
        };

        metadata.move_count += 1;
        metadata.edge_cut_gain += move_gain;

        let target_part = 1 - initial_part;
        partition[moved_vertex].store(target_part, Ordering::Relaxed);

        for (neighbor, _edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
            let neighbor_part = partition[neighbor].load(Ordering::Relaxed);
            let neighbor_gain: i64 = adjacency
                .outer_view(neighbor)
                .unwrap()
                .iter()
                .map(|(neighbor2, edge_weight)| {
                    if partition[neighbor2].load(Ordering::Relaxed) == neighbor_part {
                        -*edge_weight
                    } else {
                        *edge_weight
                    }
                })
                .sum();
            if 0 < neighbor_gain {
                cut.push(neighbor);
            }
        }

        true
    };

    let mut metadata =
        partition
            .par_iter()
            .enumerate()
            .fold(
                || (Metadata::default(), Vec::new(), part0_weight),
                |(mut metadata, mut cut, mut part0_weight), (vertex, initial_part)| {
                    let initial_part = initial_part.load(Ordering::Relaxed);
                    let on_cut = adjacency.outer_view(vertex).unwrap().iter().any(
                        |(neighbor, _edge_weight)| {
                            partition[neighbor].load(Ordering::Relaxed) != initial_part
                        },
                    );
                    if !on_cut {
                        return (metadata, cut, part0_weight);
                    }
                    cut.push(vertex);
                    while make_move(&mut cut, &mut part0_weight, &mut metadata) {
                        // blank
                    }
                    (metadata, cut, part0_weight)
                },
            )
            .map(|(metadata, _, _)| metadata)
            .reduce(Metadata::default, Metadata::merge);

    metadata.vertices_per_thread = items_per_thread;
    metadata
}

/// Trait alias for values accepted as weights by [ArcSwap].
pub trait AsWeight
where
    Self: std::fmt::Debug + Copy + PartialOrd + Send + Sync + num::Zero,
    Self: std::iter::Sum + num::FromPrimitive + num::ToPrimitive,
    Self: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = Self>,
    Self: std::ops::Div<Output = Self>,
{
}

impl<T> AsWeight for T
where
    Self: std::fmt::Debug + Copy + PartialOrd + Send + Sync + num::Zero,
    Self: std::iter::Sum + num::FromPrimitive + num::ToPrimitive,
    Self: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = Self>,
    Self: std::ops::Div<Output = Self>,
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
/// See the documentation of [`FiducciaMattheyses`][fm] for an example.
///
/// [fm]: crate::FiducciaMattheyses
#[derive(Debug, Clone, Copy, Default)]
pub struct ArcSwap {
    pub max_imbalance: Option<f64>,
}

impl<'a, W> crate::Partition<(CsMatView<'a, i64>, &'a [W])> for ArcSwap
where
    W: AsWeight,
{
    type Metadata = Metadata;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<'_, i64>, &'a [W]),
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
        let part_count = 1 + *part_ids.par_iter().max().unwrap_or(&0);
        if 2 < part_count {
            return Err(Error::BiPartitioningOnly);
        }
        Ok(arc_swap(part_ids, weights, adjacency, self.max_imbalance))
    }
}
