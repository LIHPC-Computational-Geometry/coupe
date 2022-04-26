use super::Error;
use crossbeam_queue::ArrayQueue;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
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

fn compute_gains(adjacency: sprs::CsMatView<i64>, partition: &[usize]) -> Vec<AtomicI64> {
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
        .collect()
}

fn arc_swap<W>(
    partition: &mut [usize],
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

    let part_count = 1 + *partition.iter().max().unwrap();
    debug_assert!(part_count <= 2);

    let span = tracing::info_span!("compute part_weights");
    let enter = span.enter();

    let part_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.par_iter().cloned());

    mem::drop(enter);
    let span = tracing::info_span!("compute max_part_weight");
    let enter = span.enter();

    // Enforce part weights to be below this value.
    let max_part_weight = match max_imbalance {
        Some(max_imbalance) => {
            let total_weight: W = part_weights.iter().cloned().sum();
            let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
            W::from_f64(ideal_part_weight + max_imbalance * ideal_part_weight).unwrap()
        }
        None => *part_weights.iter().max_by(partial_cmp).unwrap(),
    };

    mem::drop(enter);
    let span = tracing::info_span!("compute best_edge_cut");
    let enter = span.enter();

    let best_edge_cut = crate::topology::edge_cut(adjacency, partition);
    let best_edge_cut = AtomicI64::new(best_edge_cut);
    tracing::info!("Initial edge cut: {:?}", best_edge_cut);

    mem::drop(enter);
    let span = tracing::info_span!("compute cut, gains and locks");
    let enter = span.enter();

    let (cut, (gains, locks)) = rayon::join(
        || {
            let cut_init: Vec<usize> = partition
                .par_iter()
                .enumerate()
                .filter(|(vertex, initial_part)| {
                    adjacency
                        .outer_view(*vertex)
                        .unwrap()
                        .iter()
                        .any(|(neighbor, _edge_weight)| partition[neighbor] != **initial_part)
                })
                .map(|(vertex, _)| vertex)
                .collect();
            // Allocate enough room so that push operations *should* never fail.
            // They could because we only make sure items are eventually unique.
            let cut = ArrayQueue::new(partition.len());
            for vertex in cut_init {
                // Sequential loop to avoid contention on ArrayQueue
                let _ = cut.push(vertex);
            }
            cut
        },
        || {
            rayon::join(
                || compute_gains(adjacency, partition),
                || {
                    partition
                        .par_iter()
                        .map(|_| AtomicBool::new(false))
                        .collect::<Vec<AtomicBool>>()
                },
            )
        },
    );

    mem::drop(enter);
    let span = tracing::info_span!("doing moves");
    let _enter = span.enter();

    let part_weights = RwLock::new(part_weights);
    let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    let move_count = AtomicUsize::new(0);
    let race_count = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);

    rayon::in_place_scope(|s| {
        for _ in 0..rayon::current_num_threads() {
            s.spawn(|_| {
                loop {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let part_weights_copy = part_weights.read().unwrap().to_vec();

                    let (moved_vertex, move_gain, initial_part, weight, _lock) = loop {
                        let vertex = match cut.pop() {
                            Some(v) => v,
                            None => {
                                stop.store(true, Ordering::Release);
                                return;
                            }
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

                        let raced = adjacency.outer_view(vertex).unwrap().iter().any(
                            |(neighbor, _edge_weight)| locks[neighbor].load(Ordering::Acquire),
                        );
                        if raced {
                            cut.push(vertex).unwrap();
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
                    cut.push(moved_vertex).unwrap();
                    for (neighbor, edge_weight) in
                        adjacency.outer_view(moved_vertex).unwrap().iter()
                    {
                        if partition[neighbor].load(Ordering::Relaxed) == initial_part {
                            cut.push(neighbor).unwrap();
                            gains[neighbor].fetch_add(2 * edge_weight, Ordering::Relaxed);
                        } else {
                            gains[neighbor].fetch_sub(2 * edge_weight, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    tracing::info!(
        ?best_edge_cut,
        ?move_count,
        ?race_count,
        "cut.len()={}",
        cut.len(),
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
        if 1 < *part_ids.iter().max().unwrap_or(&0) {
            return Err(Error::BiPartitioningOnly);
        }
        arc_swap(
            part_ids,
            weights,
            adjacency,
            self.max_moves.unwrap_or(usize::MAX),
            self.max_imbalance,
        );
        Ok(())
    }
}
