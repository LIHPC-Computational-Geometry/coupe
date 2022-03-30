use super::Error;
use itertools::Itertools;
use sprs::CsMatView;

fn fiduccia_mattheyses<W>(
    part_ids: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<f64>,
    max_passes: impl Into<Option<usize>>,
    max_flips_per_pass: impl Into<Option<usize>>,
    max_imbalance_per_flip: impl Into<Option<W>>,
    max_bad_move_in_a_row: usize,
) where
    W: std::fmt::Debug + Copy + PartialOrd,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W> + num::Zero,
{
    let max_passes = max_passes.into();
    let max_flips_per_pass = max_flips_per_pass.into();
    let max_imbalance_per_flip = max_imbalance_per_flip.into();

    fiduccia_mattheyses_impl(
        part_ids,
        weights,
        adjacency,
        max_passes,
        max_flips_per_pass,
        max_imbalance_per_flip,
        max_bad_move_in_a_row,
    );
}

fn fiduccia_mattheyses_impl<W>(
    partition: &mut [usize],
    weights: &[W],
    adjacency: CsMatView<f64>,
    max_passes: Option<usize>,
    max_flips_per_pass: Option<usize>,
    max_imbalance_per_flip: Option<W>,
    max_bad_move_in_a_row: usize, // for each pass, the max number of subsequent moves that will decrease the gain
) where
    W: std::fmt::Debug + Copy + PartialOrd,
    W: std::ops::AddAssign + std::ops::SubAssign + std::ops::Sub<Output = W> + num::Zero,
{
    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.rows());
    debug_assert_eq!(partition.len(), adjacency.cols());

    let part_count = 1 + *partition.iter().max().unwrap();

    let mut parts_weights =
        crate::imbalance::compute_parts_load(partition, part_count, weights.iter().cloned());

    let mut cut_size = crate::topology::cut_size(adjacency, partition);
    tracing::info!("Initial cut size: {}", cut_size);
    let mut new_cut_size = cut_size;

    // Outer loop: each iteration makes a "pass" which can flip several nodes
    // at a time. Repeat passes until passes no longer decrease the cut size.
    for iter in 0.. {
        // check user defined iteration limit
        if let Some(max_passes) = max_passes {
            if iter >= max_passes {
                break;
            }
        }

        // save old cut size
        cut_size = new_cut_size;

        // monitors for each pass the number of subsequent flips
        // that increase cut size. It may be beneficial in some
        // situations to allow a certain amount of them. Performing bad flips can open
        // up new sequences of good flips.
        let mut num_bad_move = 0;

        // We save each flip data during a pass so that they can be reverted easily
        // afterwards. For instance if performing wrong flips did not open up any
        // good flip sequence, those bad flips must be reverted at the end of the pass
        // so that cut size remains optimal
        let mut saves = Vec::new(); // flip save
        let mut cut_saves = Vec::new(); // cut size save
        let mut ids_before_flip = Vec::new(); // the target id for reverting a flip

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
        let mut gains: Vec<Vec<(usize, f64)>> = (0..partition.len())
            .map(|_idx| (0..part_count).map(|id2| (id2, 0.)).collect())
            .collect();

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
        for _ in 0..partition
            .len()
            .min(max_flips_per_pass.unwrap_or(std::usize::MAX))
        {
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
                for (id2, ref mut gain) in other_ids.iter_mut() {
                    if partition[idx] == *id2 {
                        // target part is current part, no gain
                        *gain = 0.;
                    } else {
                        for (j, w) in adjacency.outer_view(idx).unwrap().iter() {
                            if partition[idx] == partition[j] {
                                *gain -= w;
                            } else if partition[j] == *id2 {
                                *gain += w;
                            }
                        }
                    }
                }
            }

            // find max gain and target part
            let (max_pos, (target_part, max_gain)) = gains
                .iter()
                .zip(locks.iter())
                .zip(weights.iter())
                .enumerate()
                .filter(|(_, ((_, locked), weight))| {
                    if let Some(max_imbalance_per_flip) = max_imbalance_per_flip {
                        !*locked && **weight <= max_imbalance_per_flip
                    } else {
                        !*locked
                    }
                })
                .map(|(idx, ((vec, _), _))| (idx, vec))
                .map(|(idx, vec)| {
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

            // save flip
            saves.push((max_pos, target_part, max_gain));

            // update imbalance
            parts_weights[partition[max_pos]] -= weights[max_pos];
            parts_weights[target_part] += weights[max_pos];

            // flip node
            ids_before_flip.push(partition[max_pos]);
            partition[max_pos] = target_part;

            // save cut_size
            cut_saves.push(crate::topology::cut_size(adjacency, partition));

            // end of pass
        }

        // lookup for best cutsize
        let (best_pos, best_cut) = cut_saves
            .iter()
            .cloned()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // rewind flips
        tracing::info!(
            "rewinding flips from pos {} to pos {}",
            best_pos + 1,
            cut_saves.len()
        );
        for i in best_pos + 1..cut_saves.len() {
            let idx = saves[i].0;
            partition[idx] = ids_before_flip[i];

            // revert weight change
            parts_weights[ids_before_flip[i]] += weights[idx];
            parts_weights[saves[i].1] += weights[idx];
        }

        new_cut_size = best_cut;

        if new_cut_size >= cut_size {
            break;
        }

        let (min_w, max_w) = parts_weights.iter().minmax().into_option().unwrap();

        // imbalance introduced by flipping nodes around parts
        let imbalance = *max_w - *min_w;
        tracing::info!(?imbalance);
    }

    tracing::info!("final cut size: {}", new_cut_size);
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
///      Point2D::new(0., 0.),
///      Point2D::new(1., 0.),
///      Point2D::new(2., 0.),
///      Point2D::new(3., 0.),
///      Point2D::new(0., 1.),
///      Point2D::new(1., 1.),
///      Point2D::new(2., 1.),
///      Point2D::new(3., 1.),
///  ];
///  let weights = [1.0; 8];
///  let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
///  let mut adjacency = CsMat::empty(sprs::CSR, 8);
///  adjacency.reserve_outer_dim(8);
///  eprintln!("shape: {:?}", adjacency.shape());
///  adjacency.insert(0, 1, 1.);
///  adjacency.insert(1, 2, 1.);
///  adjacency.insert(2, 3, 1.);
///  adjacency.insert(4, 5, 1.);
///  adjacency.insert(5, 6, 1.);
///  adjacency.insert(6, 7, 1.);
///  adjacency.insert(0, 4, 1.);
///  adjacency.insert(1, 5, 1.);
///  adjacency.insert(2, 6, 1.);
///  adjacency.insert(3, 7, 1.);
///  
///  // symmetry
///  adjacency.insert(1, 0, 1.);
///  adjacency.insert(2, 1, 1.);
///  adjacency.insert(3, 2, 1.);
///  adjacency.insert(5, 4, 1.);
///  adjacency.insert(6, 5, 1.);
///  adjacency.insert(7, 6, 1.);
///  adjacency.insert(4, 0, 1.);
///  adjacency.insert(5, 1, 1.);
///  adjacency.insert(6, 2, 1.);
///  adjacency.insert(7, 3, 1.);
///
/// coupe::FiducciaMattheyses { max_bad_move_in_a_row: 1, ..Default::default() }
///     .partition(&mut partition, (adjacency.view(), &weights))
///     .unwrap();
///
/// assert_eq!(partition[5], 0);
/// assert_eq!(partition[6], 1);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FiducciaMattheyses {
    pub max_passes: Option<usize>,
    pub max_flips_per_pass: Option<usize>,
    pub max_imbalance_per_flip: Option<f64>,
    pub max_bad_move_in_a_row: usize,
}

impl<'a, W> crate::Partition<(CsMatView<'a, f64>, &'a [W])> for FiducciaMattheyses
where
    W: std::fmt::Debug + Copy + PartialOrd + num::Zero + num::FromPrimitive,
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
            self.max_passes,
            self.max_flips_per_pass,
            self.max_imbalance_per_flip.map(|f| W::from_f64(f).unwrap()),
            self.max_bad_move_in_a_row,
        );
        Ok(())
    }
}
