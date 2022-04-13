use super::Error;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use sprs::CsMatView;
use std::collections::HashSet;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::sync::RwLock;

fn fiduccia_mattheyses<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<i64>,
    max_passes: usize,
    max_flips_per_pass: usize,
    max_imbalance: Option<f64>,
) where
    W: std::fmt::Debug + Copy + Send + Sync + PartialOrd,
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
            .max_by(|imbalance0, imbalance1| f64::partial_cmp(imbalance0, imbalance1).unwrap())
            .unwrap()
    });
    tracing::info!(?max_imbalance);

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

    let gain_to_node = (-pmax..=pmax)
        .map(|_| Mutex::new(HashSet::new()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let node_to_gain = (0..adjacency.outer_dims())
        .map(|_| AtomicI64::new(i64::MIN))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let gain_table_idx = |gain: i64| (gain + pmax) as usize;

    let mut best_cut_size = crate::topology::cut_size(adjacency, partition);
    tracing::info!("Initial cut size: {}", best_cut_size);

    let partition = unsafe { std::mem::transmute::<&mut [usize], &[AtomicUsize]>(partition) };

    for _ in 0..max_passes {
        let span = tracing::info_span!("creating gain table");
        let enter = span.enter();

        for set in &*gain_to_node {
            set.try_lock().unwrap().clear();
        }
        partition
            .par_iter()
            .enumerate()
            .fold(
                || std::collections::BTreeMap::<usize, HashSet<usize>>::new(),
                |mut acc, (node, initial_part)| {
                    let initial_part = initial_part.load(Ordering::Relaxed);
                    let gain: i64 = adjacency
                        .outer_view(node)
                        .unwrap()
                        .iter()
                        .map(|(neighbor, edge_weight)| {
                            let neighbor_part = partition[neighbor].load(Ordering::Relaxed);
                            if neighbor_part == initial_part {
                                -*edge_weight
                            } else {
                                // neighbor_part == 1 - initial_part
                                *edge_weight
                            }
                        })
                        .sum();

                    acc.entry(gain_table_idx(gain)).or_default().insert(node);
                    acc
                },
            )
            .for_each(|map| {
                for (gain, nodes) in map {
                    gain_to_node[gain].lock().unwrap().extend(nodes);
                }
            });

        std::mem::drop(enter);

        struct Synchronized<'a, W> {
            part_weights: &'a mut [W],
            last_cut_size: i64,
        }

        let s = RwLock::new(Synchronized {
            part_weights: &mut parts_weights,
            last_cut_size: best_cut_size,
        });

        (0..max_flips_per_pass).into_par_iter().try_for_each(|_| {
            let span = tracing::info_span!("copy part_weights");
            let enter = span.enter();

            let part_weights_copy = s.read().unwrap().part_weights.to_vec();

            std::mem::drop(enter);
            let span = tracing::info_span!("find node to move");
            let enter = span.enter();

            let mut has_top_gain = true;
            let (max_pos, max_gain) = gain_to_node
                .iter()
                .rev()
                .zip((-pmax..=pmax).rev())
                .find_map(|(nodes, gain)| {
                    let nodes = match nodes.try_lock() {
                        Ok(v) => v,
                        Err(_) => {
                            has_top_gain = false;
                            return None;
                        }
                    };
                    let (best_node, _) = nodes
                        .iter()
                        .filter_map(|node| {
                            let weight = weights[*node];
                            let initial_part = partition[*node].load(Ordering::Relaxed);
                            let target_part = 1 - initial_part;
                            let initial_part_weight = part_weights_copy[initial_part] - weight;
                            let target_part_weight = part_weights_copy[target_part] + weight;
                            let imbalance = [initial_part_weight, target_part_weight]
                                .iter()
                                .map(part_imbalance)
                                .max_by(|imbalance0, imbalance1| {
                                    f64::partial_cmp(imbalance0, imbalance1).unwrap()
                                })
                                .unwrap();
                            if max_imbalance < imbalance {
                                return None;
                            }
                            Some((*node, imbalance))
                        })
                        .min_by(|(_, imbalance0), (_, imbalance1)| {
                            f64::partial_cmp(imbalance0, imbalance1).unwrap()
                        })?;
                    Some((best_node, gain))
                })?;

            tracing::info!(max_pos, max_gain, "found node");
            std::mem::drop(enter);

            if max_gain <= 0 {
                if has_top_gain {
                    tracing::info!("bad move");
                    return None;
                } else {
                    tracing::info!("could not get a good gain, retrying");
                    return Some(());
                }
            }

            let span = tracing::info_span!("lock node");
            let enter = span.enter();

            let was_present = gain_to_node[gain_table_idx(max_gain)]
                .lock()
                .unwrap()
                .remove(&max_pos);
            if !was_present {
                tracing::info!("raced, node was already locked");
                return Some(());
            }
            node_to_gain[max_pos].store(i64::MIN, Ordering::Relaxed);

            let old_part = partition[max_pos].load(Ordering::Relaxed);
            let target_part = 1 - old_part;
            partition[max_pos].store(target_part, Ordering::Relaxed);

            std::mem::drop(enter);

            {
                let span = tracing::info_span!("update part_weights and history");
                let _enter = span.enter();

                let mut s = s.write().unwrap();
                s.part_weights[old_part] -= weights[max_pos];
                s.part_weights[target_part] += weights[max_pos];

                // save cut_size
                tracing::info!(max_pos, old_part, target_part, "flip");
                for (neighbors, edge_weight) in adjacency.outer_view(max_pos).unwrap().iter() {
                    if partition[neighbors].load(Ordering::Relaxed) == old_part {
                        s.last_cut_size += edge_weight;
                    } else {
                        s.last_cut_size -= edge_weight;
                    }
                }
            }

            let span = tracing::info_span!("update gain table");
            let _enter = span.enter();

            for (neighbor, _) in adjacency.outer_view(max_pos).unwrap().iter() {
                let initial_part = partition[neighbor].load(Ordering::Relaxed);
                let outdated_gain = node_to_gain[neighbor].load(Ordering::Relaxed);
                if outdated_gain == i64::MIN {
                    continue;
                }
                let updated_gain = adjacency
                    .outer_view(neighbor)
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
                if outdated_gain != updated_gain {
                    gain_to_node[gain_table_idx(outdated_gain)]
                        .lock()
                        .unwrap()
                        .remove(&neighbor);
                    gain_to_node[gain_table_idx(updated_gain)]
                        .lock()
                        .unwrap()
                        .insert(neighbor);
                }
            }

            Some(())
        });

        let new_cut_size = s.into_inner().unwrap().last_cut_size;
        if best_cut_size <= new_cut_size {
            break;
        }
        best_cut_size = new_cut_size;
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
}

impl<'a, W> crate::Partition<(CsMatView<'a, i64>, &'a [W])> for FiducciaMattheyses
where
    W: std::fmt::Debug + Copy + Send + Sync + PartialOrd + num::Zero,
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
        );
        Ok(())
    }
}
