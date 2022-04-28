use super::Error;
use crate::algorithms::recursive_bisection::work_share;
use crossbeam_queue::ArrayQueue;
use rand::Rng;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::slice::ParallelSlice;
use sprs::CsMatView;
use std::cell::UnsafeCell;
use std::cmp;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Debug)]
struct RaceCell<T> {
    value: UnsafeCell<T>,
}

impl<T> RaceCell<T> {
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
        }
    }
}

impl<T> RaceCell<T>
where
    T: Copy,
{
    pub fn get(&self) -> T {
        unsafe { self.value.get().read() }
    }

    pub fn set(&self, value: T) {
        unsafe {
            self.value.get().write(value);
        }
    }
}

unsafe impl<T> Send for RaceCell<T> {}
unsafe impl<T> Sync for RaceCell<T> {}

struct Defer<F>(Option<F>)
where
    F: FnOnce();

fn defer<F>(f: F) -> Defer<F>
where
    F: FnOnce(),
{
    Defer(Some(f))
}

impl<F> Drop for Defer<F>
where
    F: FnOnce(),
{
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

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

#[derive(Debug, Default, Clone, Copy)]
pub struct Metadata {
    /// By how much the edge cut has been reduced by the algorithm.
    /// Positive values mean reduced edge cut.
    edge_cut_gain: i64,
    /// Number of times vertices has been moved.
    move_count: usize,
    /// Number of times threads found out the vertex they picked had a locked
    /// neighbor.
    race_count: usize,
    /// Number of times threads have called `ArrayQueue::pop` from their own
    /// queue.
    pop_from_self_count: usize,
    /// Number of times threads have called `ArrayQueue::pop` from another's
    /// queue.
    pop_from_others_count: usize,
    /// Number of times threads have poped a locked vertex.
    locked_count: usize,
    /// Number of times threads have poped a vertex with negative gain.
    no_gain_count: usize,
    /// Number of times threads have poped a vertex that would disrupt balance.
    bad_balance_count: usize,
}

#[derive(Default)]
struct AtomicMetadata {
    edge_cut_gain: AtomicI64,
    move_count: AtomicUsize,
    race_count: AtomicUsize,
    pop_from_self_count: AtomicUsize,
    pop_from_others_count: AtomicUsize,
    locked_count: AtomicUsize,
    no_gain_count: AtomicUsize,
    bad_balance_count: AtomicUsize,
}

impl AtomicMetadata {
    pub fn add(&self, m: Metadata) {
        self.edge_cut_gain
            .fetch_add(m.edge_cut_gain, Ordering::Relaxed);
        self.move_count.fetch_add(m.move_count, Ordering::Relaxed);
        self.race_count.fetch_add(m.race_count, Ordering::Relaxed);
        self.pop_from_self_count
            .fetch_add(m.pop_from_self_count, Ordering::Relaxed);
        self.pop_from_others_count
            .fetch_add(m.pop_from_others_count, Ordering::Relaxed);
        self.locked_count
            .fetch_add(m.locked_count, Ordering::Relaxed);
        self.no_gain_count
            .fetch_add(m.no_gain_count, Ordering::Relaxed);
        self.bad_balance_count
            .fetch_add(m.bad_balance_count, Ordering::Relaxed);
    }
}

impl From<AtomicMetadata> for Metadata {
    fn from(a: AtomicMetadata) -> Metadata {
        Metadata {
            edge_cut_gain: a.edge_cut_gain.load(Ordering::Relaxed),
            move_count: a.move_count.load(Ordering::Relaxed),
            race_count: a.race_count.load(Ordering::Relaxed),
            pop_from_self_count: a.pop_from_self_count.load(Ordering::Relaxed),
            pop_from_others_count: a.pop_from_others_count.load(Ordering::Relaxed),
            locked_count: a.locked_count.load(Ordering::Relaxed),
            no_gain_count: a.no_gain_count.load(Ordering::Relaxed),
            bad_balance_count: a.bad_balance_count.load(Ordering::Relaxed),
        }
    }
}

fn arc_swap<W>(
    partition: &mut [usize],
    part_count: usize,
    weights: &[W],
    adjacency: sprs::CsMatBase<i64, usize, &[usize], &[usize], &[i64]>,
    max_moves: usize,
    max_imbalance: Option<f64>,
) -> Metadata
where
    W: std::fmt::Debug + Copy + PartialOrd + Send + Sync + num::Zero,
    W: std::iter::Sum + num::FromPrimitive + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W>,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());
    debug_assert!(part_count <= 2);

    let (items_per_thread, thread_count) =
        work_share(partition.len(), rayon::current_num_threads());
    let compute_cut =
        || {
            let span = tracing::info_span!("compute cut");
            let _enter = span.enter();

            partition
                .par_chunks(items_per_thread)
                .enumerate()
                .map(|(start_idx, vertex_chunk)| {
                    let cut_chunk = ArrayQueue::new(items_per_thread);
                    vertex_chunk
                        .iter()
                        .zip(start_idx * items_per_thread..)
                        .filter(|(initial_part, vertex)| {
                            adjacency.outer_view(*vertex).unwrap().iter().any(
                                |(neighbor, _edge_weight)| partition[neighbor] != **initial_part,
                            )
                        })
                        .for_each(|(_, vertex)| cut_chunk.push(vertex).unwrap());
                    cut_chunk
                })
                .collect::<Vec<_>>()
        };
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
        (total_weight, part_weights[0], max_part_weight)
    };
    let compute_gains = || {
        let span = tracing::info_span!("compute gains");
        let _enter = span.enter();

        partition
            .par_iter()
            .enumerate()
            .map(|(vertex, initial_part)| {
                let gain: i64 = adjacency
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
                AtomicI64::new(gain)
            })
            .collect::<Vec<AtomicI64>>()
    };
    let compute_locks = || {
        let span = tracing::info_span!("compute locks");
        let _enter = span.enter();

        partition
            .par_iter()
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<AtomicBool>>()
    };

    let (cuts, ((total_weight, part0_weight, max_part_weight), (gains, locks))) =
        rayon::join(compute_cut, || {
            rayon::join(compute_part_weights, || {
                rayon::join(compute_gains, compute_locks)
            })
        });

    let span = tracing::info_span!("compute cut, gains and locks");
    let enter = span.enter();

    mem::drop(enter);
    let span = tracing::info_span!("doing moves");
    let _enter = span.enter();

    let weight_contributions = (0..thread_count)
        .map(|_| RaceCell::new(W::zero()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    let metadata = AtomicMetadata::default();
    (0..max_moves)
        .into_par_iter()
        .try_fold(Metadata::default, |mut metadata, _| {
            let thread_idx = rayon::current_thread_index().unwrap() % thread_count;
            let mut rng = rand::thread_rng();

            let (moved_vertex, move_gain, initial_part, _lock) = loop {
                let vertex = match cuts[thread_idx].pop() {
                    Some(v) => {
                        metadata.pop_from_self_count += 1;
                        v
                    }
                    None => match cuts[rng.gen_range(0..thread_count)].pop() {
                        Some(v) => {
                            metadata.pop_from_others_count += 1;
                            v
                        }
                        None => {
                            return Err(metadata);
                        }
                    },
                };
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

                let initial_part = partition[vertex].load(Ordering::Relaxed);
                let target_part = 1 - initial_part;

                let gain = gains[vertex].load(Ordering::Relaxed);
                if gain <= 0 {
                    // Don't push it back now, it will when its gain is updated
                    // positively (due to a neighbor moving).
                    metadata.no_gain_count += 1;
                    continue;
                }

                let raced = adjacency
                    .outer_view(vertex)
                    .unwrap()
                    .iter()
                    .any(|(neighbor, _edge_weight)| locks[neighbor].load(Ordering::Acquire));
                if raced {
                    cuts[thread_idx].push(vertex).unwrap();
                    metadata.race_count += 1;
                    continue;
                }

                let weight = weights[vertex];
                let part0_weight =
                    part0_weight + weight_contributions.iter().map(|cell| cell.get()).sum();
                let target_part_weight = weight
                    + if target_part == 0 {
                        part0_weight
                    } else {
                        total_weight - part0_weight
                    };
                if max_part_weight < target_part_weight {
                    // TODO fix infinite loops
                    //cut.push(vertex).unwrap();
                    metadata.bad_balance_count += 1;
                    continue;
                }

                weight_contributions[thread_idx].set(if initial_part == 0 {
                    part0_weight - weight
                } else {
                    part0_weight + weight
                });

                break (vertex, gain, initial_part, lock_guard);
            };

            metadata.move_count += 1;
            metadata.edge_cut_gain += move_gain;

            let target_part = 1 - initial_part;
            partition[moved_vertex].store(target_part, Ordering::Relaxed);
            gains[moved_vertex].store(-move_gain, Ordering::Relaxed);

            for (neighbor, edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
                if partition[neighbor].load(Ordering::Relaxed) == initial_part {
                    let old_gain = gains[neighbor].fetch_add(2 * edge_weight, Ordering::Relaxed);
                    if 0 <= old_gain + 2 * edge_weight {
                        cuts[thread_idx].push(neighbor).unwrap();
                    }
                } else {
                    gains[neighbor].fetch_sub(2 * edge_weight, Ordering::Relaxed);
                }
            }

            Ok(metadata)
        })
        .try_for_each(|mdpart| {
            let err = mdpart.is_err();
            let (Ok(mdpart) | Err(mdpart)) = mdpart;
            metadata.add(mdpart);
            if err {
                Some(())
            } else {
                None
            }
        });

    metadata.into()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ArcSwap {
    pub max_moves: Option<usize>,
    pub max_imbalance: Option<f64>,
}

impl<'a, W> crate::Partition<(CsMatView<'a, i64>, &'a [W])> for ArcSwap
where
    W: std::fmt::Debug + Copy + PartialOrd + Send + Sync + num::Zero,
    W: std::iter::Sum + num::FromPrimitive + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W>,
{
    type Metadata = Metadata;
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<i64>, &'a [W]),
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
        Ok(arc_swap(
            part_ids,
            part_count,
            weights,
            adjacency,
            self.max_moves.unwrap_or(usize::MAX),
            self.max_imbalance,
        ))
    }
}
