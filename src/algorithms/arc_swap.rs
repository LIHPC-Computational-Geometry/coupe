use super::Error;
use rand::Rng as _;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
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

//struct Defer<F>(Option<F>)
//where
//    F: FnOnce();
//
//fn defer<F>(f: F) -> Defer<F>
//where
//    F: FnOnce(),
//{
//    Defer(Some(f))
//}
//
//impl<F> Drop for Defer<F>
//where
//    F: FnOnce(),
//{
//    fn drop(&mut self) {
//        if let Some(f) = self.0.take() {
//            f();
//        }
//    }
//}

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
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());

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

    let part_weights = RwLock::new(part_weights);
    let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    let locks = partition
        .iter()
        .map(|_| AtomicBool::new(false))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    (0..max_moves)
        .into_par_iter()
        .try_for_each_init(rand::thread_rng, |rng, _| {
            let part_weights_copy = part_weights.read().unwrap().to_vec();
            let start = rng.gen_range(0..partition.len());
            let (moved_vertex, move_gain, initial_part) = partition[start..]
                .par_iter()
                .chain(&partition[..start])
                .zip((start..partition.len()).into_par_iter().chain(0..start))
                .find_map_any(|(initial_part, vertex)| {
                    let locked = locks[vertex]
                        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok();
                    if locked {
                        return None;
                    }

                    let initial_part = initial_part.load(Ordering::Relaxed);
                    let target_part = 1 - initial_part;

                    let gain: i64 = adjacency
                        .outer_view(vertex)
                        .unwrap()
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
                        locks[vertex].store(false, Ordering::Release);
                        return None;
                    }

                    let weight = weights[vertex];
                    let target_part_weight = part_weights_copy[target_part] + weight;
                    if max_part_weight < target_part_weight {
                        locks[vertex].store(false, Ordering::Release);
                        return None;
                    }

                    let raced = adjacency
                        .outer_view(vertex)
                        .unwrap()
                        .iter()
                        .any(|(neighbor, _)| locks[neighbor].load(Ordering::Acquire));
                    if raced {
                        locks[vertex].store(false, Ordering::Release);
                        return None;
                    }

                    Some((vertex, gain, initial_part))
                })?;

            let target_part = 1 - initial_part;
            partition[moved_vertex].store(target_part, Ordering::Relaxed);
            best_edge_cut.fetch_sub(move_gain, Ordering::Relaxed);
            locks[moved_vertex].store(false, Ordering::Release);

            Some(())
        });

    tracing::info!("final edge cut: {:?}", best_edge_cut);
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
