use super::Error;
use rand::Rng as _;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMatView;
use std::cmp;
use std::collections::HashMap;
use std::mem;
use std::ops::DerefMut as _;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::sync::MutexGuard;
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

    loop {
        let old_edge_cut = best_edge_cut.load(Ordering::Relaxed);

        let span = tracing::info_span!("compute cut");
        let enter = span.enter();

        let gains: Vec<Mutex<Option<(usize, i64)>>> = partition
            .par_iter()
            .enumerate()
            .filter_map(|(vertex, initial_part)| {
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
                if gain <= 0 {
                    None
                } else {
                    Some(Mutex::new(Some((vertex, gain))))
                }
            })
            .collect();

        if gains.is_empty() {
            break;
        }

        mem::drop(enter);
        let span = tracing::info_span!("compute vertex_to_gain");
        let enter = span.enter();

        let vertex_to_gain: Vec<AtomicUsize> = (0..partition.len())
            .into_par_iter()
            .map(|_| AtomicUsize::new(usize::MAX))
            .collect();
        gains.par_iter().enumerate().for_each(|(i, gain_entry)| {
            let (vertex, _) = gain_entry.try_lock().unwrap().unwrap();
            vertex_to_gain[vertex].store(i, Ordering::Relaxed);
        });

        mem::drop(enter);
        let span = tracing::info_span!("doing moves");
        let _enter = span.enter();

        let partition = unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

        let move_count = AtomicUsize::new(0);
        let race_count = AtomicUsize::new(0);

        (0..max_moves)
            .into_par_iter()
            .panic_fuse()
            .try_for_each_init(rand::thread_rng, |rng, _| {
                let part_weights_copy = part_weights.read().unwrap().to_vec();
                let start = rng.gen_range(0..gains.len());
                gains[start..]
                    .par_iter()
                    .chain(&gains[..start])
                    .panic_fuse()
                    .find_map_any(|gain_entry| {
                        let mut gain_entry = gain_entry.try_lock().ok()?;
                        let (vertex, gain) = (*gain_entry)?;

                        let initial_part = partition[vertex].load(Ordering::Relaxed);
                        let target_part = 1 - initial_part;

                        debug_assert!(adjacency
                            .outer_view(vertex)
                            .unwrap()
                            .iter()
                            .any(|(neighbor, _edge_weight)| partition[neighbor]
                                .load(Ordering::Relaxed)
                                != initial_part));

                        let weight = weights[vertex];
                        let target_part_weight = part_weights_copy[target_part] + weight;
                        if max_part_weight < target_part_weight {
                            return None;
                        }

                        let neighbor_gains: Option<HashMap<_, _>> = adjacency
                            .outer_view(vertex)
                            .unwrap()
                            .iter()
                            .filter_map(|(neighbor, _edge_weight)| {
                                let gain_idx = vertex_to_gain[neighbor].load(Ordering::Relaxed);
                                if gain_idx == usize::MAX {
                                    None
                                } else {
                                    let gain_entry = match gains[gain_idx].try_lock() {
                                        Ok(v) => v,
                                        Err(_) => return Some(None),
                                    };
                                    debug_assert_eq!(
                                        gain_idx,
                                        vertex_to_gain[neighbor].load(Ordering::Relaxed),
                                    );
                                    Some(Some((neighbor, gain_entry)))
                                }
                            })
                            .collect();
                        let mut neighbor_gains = match neighbor_gains {
                            Some(v) => v,
                            None => {
                                //tracing::info!("raced");
                                race_count.fetch_add(1, Ordering::Relaxed);
                                return None;
                            }
                        };

                        tracing::info!(
                            ?move_count,
                            ?race_count,
                            vertex,
                            target_part,
                            ?best_edge_cut,
                            gain,
                            "ok!",
                        );
                        move_count.fetch_add(1, Ordering::Relaxed);

                        partition[vertex].store(target_part, Ordering::Relaxed);
                        best_edge_cut.fetch_sub(gain, Ordering::Relaxed);
                        {
                            let mut part_weights = part_weights.write().unwrap();
                            part_weights[initial_part] -= weight;
                            part_weights[target_part] += weight;
                        }

                        *MutexGuard::deref_mut(&mut gain_entry) = None;
                        vertex_to_gain[vertex].store(usize::MAX, Ordering::Relaxed);

                        for (neighbor, edge_weight) in adjacency.outer_view(vertex).unwrap().iter()
                        {
                            let mut gain_entry = match neighbor_gains.remove(&neighbor) {
                                Some(v) => v,
                                None => continue,
                            };
                            let (_, neighbor_gain) =
                                MutexGuard::deref_mut(&mut gain_entry).as_mut().unwrap();
                            if partition[neighbor].load(Ordering::Relaxed) == initial_part {
                                *neighbor_gain += 2 * edge_weight;
                            } else {
                                *neighbor_gain -= 2 * edge_weight;
                            }
                            if *neighbor_gain < 0 {
                                *MutexGuard::deref_mut(&mut gain_entry) = None;
                                vertex_to_gain[neighbor].store(usize::MAX, Ordering::Relaxed);
                            }
                        }

                        Some(())
                    })?;
                Some(())
            });

        let new_edge_cut = best_edge_cut.load(Ordering::Relaxed);
        if old_edge_cut <= new_edge_cut {
            break;
        }
    }

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
