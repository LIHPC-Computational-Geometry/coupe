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
use std::cmp;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::RwLock;

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

fn arc_swap<W>(
    partition: &mut [usize],
    part_count: usize,
    weights: &[W],
    adjacency: sprs::CsMatBase<i64, usize, &[usize], &[usize], &[i64]>,
    max_moves: usize,
    max_imbalance: Option<f64>,
) where
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
    let compute_part_weights_and_max_part_weight = || {
        let span = tracing::info_span!("compute part_weights and max_part_weight");
        let _enter = span.enter();

        let part_weights = crate::imbalance::compute_parts_load(
            partition,
            part_count,
            weights.par_iter().cloned(),
        );

        // Enforce part weights to be below this value.
        let max_part_weight = match max_imbalance {
            Some(max_imbalance) => {
                let total_weight: W = part_weights.iter().cloned().sum();
                let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
                W::from_f64(ideal_part_weight + max_imbalance * ideal_part_weight).unwrap()
            }
            None => *part_weights.iter().max_by(partial_cmp).unwrap(),
        };
        (part_weights, max_part_weight)
    };
    let compute_best_edge_cut = || {
        let span = tracing::info_span!("compute best_edge_cut");
        let _enter = span.enter();

        let best_edge_cut = crate::topology::edge_cut(adjacency, partition);
        tracing::info!("Initial edge cut: {}", best_edge_cut);
        AtomicI64::new(best_edge_cut)
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

    let (cuts, ((part_weights, max_part_weight), (best_edge_cut, (gains, locks)))) =
        rayon::join(compute_cut, || {
            rayon::join(compute_part_weights_and_max_part_weight, || {
                rayon::join(compute_best_edge_cut, || {
                    rayon::join(compute_gains, compute_locks)
                })
            })
        });

    let span = tracing::info_span!("compute cut, gains and locks");
    let enter = span.enter();

    mem::drop(enter);
    let span = tracing::info_span!("doing moves");
    let _enter = span.enter();

    let part_weights = RwLock::new(part_weights);
    let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    let move_count = AtomicUsize::new(0);
    let race_count = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);

    (0..max_moves)
        .into_par_iter()
        .try_for_each_init(rand::thread_rng, |rng, _| {
            let thread_idx = rayon::current_thread_index().unwrap() % thread_count;
            let part_weights_copy = part_weights.read().unwrap().to_vec();

            let (moved_vertex, move_gain, initial_part, weight, _lock) = loop {
                let vertex = match cuts[thread_idx].pop() {
                    Some(v) => v,
                    None => match cuts[rng.gen_range(0..thread_count)].pop() {
                        Some(v) => v,
                        None => {
                            return None;
                        }
                    },
                };
                let locked = locks[vertex]
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_err();
                if locked {
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
                    continue;
                }

                let weight = weights[vertex];
                let target_part_weight = part_weights_copy[target_part] + weight;
                if max_part_weight < target_part_weight {
                    // TODO fix infinite loops
                    //cut.push(vertex).unwrap();
                    continue;
                }

                let raced = adjacency
                    .outer_view(vertex)
                    .unwrap()
                    .iter()
                    .any(|(neighbor, _edge_weight)| locks[neighbor].load(Ordering::Acquire));
                if raced {
                    cuts[thread_idx].push(vertex).unwrap();
                    race_count.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                break (vertex, gain, initial_part, weight, lock_guard);
            };

            if max_moves <= 1 + move_count.fetch_add(1, Ordering::Release) {
                stop.store(true, Ordering::Release);
            }

            let target_part = 1 - initial_part;
            partition[moved_vertex].store(target_part, Ordering::Relaxed);
            gains[moved_vertex].store(-move_gain, Ordering::Relaxed);
            {
                let mut part_weights = part_weights.write().unwrap();
                part_weights[initial_part] -= weight;
                part_weights[target_part] += weight;
            }

            best_edge_cut.fetch_sub(move_gain, Ordering::Relaxed);
            for (neighbor, edge_weight) in adjacency.outer_view(moved_vertex).unwrap().iter() {
                if partition[neighbor].load(Ordering::Relaxed) == initial_part {
                    cuts[thread_idx].push(neighbor).unwrap();
                    gains[neighbor].fetch_add(2 * edge_weight, Ordering::Relaxed);
                } else {
                    gains[neighbor].fetch_sub(2 * edge_weight, Ordering::Relaxed);
                }
            }

            Some(())
        });

    tracing::info!(
        ?best_edge_cut,
        ?move_count,
        ?race_count,
        "cut.len()={}",
        cuts.len(),
    );
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
        let part_count = 1 + *part_ids.par_iter().max().unwrap_or(&0);
        if 2 < part_count {
            return Err(Error::BiPartitioningOnly);
        }
        arc_swap(
            part_ids,
            part_count,
            weights,
            adjacency,
            self.max_moves.unwrap_or(usize::MAX),
            self.max_imbalance,
        );
        Ok(())
    }
}
