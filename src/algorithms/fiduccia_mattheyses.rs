use super::Error;
use itertools::Itertools;
use sprs::CsMatView;

fn fiduccia_mattheyses<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<f64>,
    max_passes: usize,
    max_flips_per_pass: usize,
    max_imbalance: Option<f64>,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) where
    W: std::fmt::Debug + Copy + PartialOrd,
    W: std::iter::Sum + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W> + num::Zero,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());

    let part_count = 1 + *partition.iter().max().unwrap();
    let mut parts_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());
    let total_weight: W = parts_weights.iter().cloned().sum();
    let ideal_part_weight = total_weight.to_f64().unwrap() / part_count as f64;
    let max_imbalance = max_imbalance.unwrap_or_else(|| {
        parts_weights
            .iter()
            .map(|part_weight| {
                (part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight
            })
            .minmax()
            .into_option()
            .unwrap()
            .1
    });
    tracing::info!(?max_imbalance);

    let mut best_cut_size = crate::topology::cut_size(adjacency, partition);
    tracing::info!("Initial cut size: {}", best_cut_size);

    for _ in 0..max_passes {
        // monitors for each pass the number of subsequent flips
        // that increase cut size. It may be beneficial in some
        // situations to allow a certain amount of them. Performing bad flips can open
        // up new sequences of good flips.
        let mut num_bad_move = 0;

        // Avoid copying partition arrays around and instead record an history
        // of flips.
        let mut flip_history = Vec::new();
        let mut cut_size_history = Vec::new();

        // create gain data structure
        // for each node, a gain is associated to each possible target part.
        // It is currently implemented wit an array of vectors:
        // [
        //  node_1: [(target_part_1, gain_1), ..., (target_part_n, gain_n)],
        //  ...,
        //  node_n: [(target_part_1, gain_1), ..., (target_part_n, gain_n)]
        // ]
        //
        // note that the current part in wich a node is is still considered as a potential target part
        // with a gain 0.
        let mut gains: Vec<Vec<(usize, f64)>> =
            vec![(0..part_count).map(|id2| (id2, 0.0)).collect(); partition.len()];

        // lock array
        // during a loop iteration, if a node is flipped during a pass,
        // it becomes locked and can't be flipped again during the following passes,
        // and is unlocked at next loop iteration.
        // locks are per node and do not depend on target partition.
        let mut locks = vec![false; partition.len()];

        // enter pass loop
        // The number of iteration of the pas loop is at most the
        // number of nodes in the mesh. However, if too many subsequent
        // bad flips are performed, the loop will break early
        for _ in 0..max_flips_per_pass {
            // construct gains
            // Right now all of the gains are recomputed at each new pass
            // a possible optimization would be to use a different gain data structure
            // to modify only some of the gains instead of recomputing everything.
            //
            // for each node (assigned to part p), and for each target part q (with p != q),
            // gain contributiuon comes from each node neighbor:
            //   - if the neighbor is in part p, then the flip will increase cut size
            //   - if the neighbor is in part q, then the flip will decrease cut size
            //   - if the neighbor is not in part p nor in q, then the flip won't affect the cut size
            for (idx, other_ids) in gains.iter_mut().enumerate() {
                for (id2, gain) in other_ids.iter_mut() {
                    *gain = 0.0;
                    if partition[idx] == *id2 {
                        // target part is current part, no gain
                        continue;
                    }
                    for (j, w) in adjacency.outer_view(idx).unwrap().iter() {
                        if partition[idx] == partition[j] {
                            *gain -= w;
                        } else if partition[j] == *id2 {
                            *gain += w;
                        }
                    }
                }
            }

            // find max gain and target part
            let (max_pos, (target_part, max_gain)) = gains
                .iter()
                .zip(locks.iter())
                .enumerate()
                .filter(|(_, (_, locked))| !*locked)
                .map(|(idx, (vec, _))| {
                    // (index of node, (target part of max gain, max gain))
                    (
                        idx,
                        *vec.iter()
                            .max_by(|(_idx1, gain1), (_id2, gain2)| {
                                gain1.partial_cmp(gain2).unwrap()
                            })
                            .unwrap(),
                    )
                })
                // get max gain of max gains computed for each node
                .max_by(|(_, (_, gain1)), (_, (_, gain2))| gain1.partial_cmp(gain2).unwrap())
                .unwrap();

            if max_gain <= 0. {
                if num_bad_move >= max_bad_move_in_a_row {
                    tracing::info!("reached max bad move in a row");
                    break;
                }
                num_bad_move += 1;
            } else {
                // a good move breaks the bad moves sequence
                num_bad_move = 0;
            }

            // lock node
            locks[max_pos] = true;

            // flip node
            let old_part = std::mem::replace(&mut partition[max_pos], target_part);
            flip_history.push((max_pos, old_part, target_part));
            parts_weights[old_part] -= weights[max_pos];
            parts_weights[target_part] += weights[max_pos];

            let imbalance = parts_weights
                .iter()
                .map(|part_weight| {
                    (part_weight.to_f64().unwrap() - ideal_part_weight) / ideal_part_weight
                })
                .minmax()
                .into_option()
                .unwrap()
                .1;
            if imbalance <= max_imbalance {
                // save cut_size
                tracing::info!(?imbalance, max_pos, old_part, target_part, "flip");
                cut_size_history.push(crate::topology::cut_size(adjacency, partition));
            } else {
                // revert flip
                tracing::info!(?imbalance, max_pos, old_part, target_part, "no flip");
                partition[max_pos] = old_part;
                parts_weights[old_part] += weights[max_pos];
                parts_weights[target_part] -= weights[max_pos];
                flip_history.pop();
            }
        }

        let old_cut_size = best_cut_size;

        // lookup for best cutsize
        let (best_pos, best_cut) = match cut_size_history
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            Some(v) => v,
            None => break,
        };

        tracing::info!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            flip_history.len()
        );
        for (idx, old_part, target_part) in flip_history.drain(best_pos + 1..) {
            partition[idx] = old_part;
            parts_weights[old_part] += weights[idx];
            parts_weights[target_part] += weights[idx];
        }

        best_cut_size = best_cut;

        if old_cut_size <= best_cut_size {
            break;
        }
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
/// Original algorithm from "A Linear-Time Heuristic for Improving Network Partitions"
/// by C.M. Fiduccia and R.M. Mattheyses.
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
/// adjacency.insert(0, 1, 1.);
/// adjacency.insert(1, 2, 1.);
/// adjacency.insert(2, 3, 1.);
/// adjacency.insert(4, 5, 1.);
/// adjacency.insert(5, 6, 1.);
/// adjacency.insert(6, 7, 1.);
/// adjacency.insert(0, 4, 1.);
/// adjacency.insert(1, 5, 1.);
/// adjacency.insert(2, 6, 1.);
/// adjacency.insert(3, 7, 1.);
///
/// // symmetry
/// adjacency.insert(1, 0, 1.);
/// adjacency.insert(2, 1, 1.);
/// adjacency.insert(3, 2, 1.);
/// adjacency.insert(5, 4, 1.);
/// adjacency.insert(6, 5, 1.);
/// adjacency.insert(7, 6, 1.);
/// adjacency.insert(4, 0, 1.);
/// adjacency.insert(5, 1, 1.);
/// adjacency.insert(6, 2, 1.);
/// adjacency.insert(7, 3, 1.);
///
/// // Set the imbalance tolerance to 25% to provide enough room for FM to do
/// // the swap.
/// coupe::FiducciaMattheyses { max_imbalance: Some(0.25), ..Default::default() }
///     .partition(&mut partition, (adjacency.view(), &weights))
///     .unwrap();
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FiducciaMattheyses {
    pub max_passes: Option<usize>,
    pub max_flips_per_pass: Option<usize>,
    pub max_imbalance: Option<f64>,
    pub max_bad_move_in_a_row: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, f64>, &'a [W])> for FiducciaMattheyses
where
    W: std::fmt::Debug + Copy + PartialOrd + num::Zero,
    W: std::iter::Sum + num::ToPrimitive,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W>,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<f64>, &'a [W]),
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
