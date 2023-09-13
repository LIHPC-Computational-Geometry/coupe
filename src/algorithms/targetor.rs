use crate::Error;
use itertools::Itertools;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::cmp::{self, Ordering, PartialOrd};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::IndexMut;
use std::ops::{Add, AddAssign, Sub, SubAssign};

type CWeightId = usize;
type PartId = usize;
type CriterionId = usize;
type CWeightMove = (CWeightId, PartId);
type BoxIndex = u32;

pub trait GainValue:
    Copy
    + Debug
    + FromPrimitive
    + ToPrimitive
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + PartialOrd
{
}

impl<T> GainValue for T
where
    Self: Copy + Debug,
    Self: FromPrimitive + ToPrimitive + Zero,
    Self: Add<Output = Self>,
    Self: Sub<Output = Self>,
    Self: PartialOrd,
{
}

pub trait PositiveWeight:
    Sized
    + Copy
    + Zero
    + PartialOrd
    + Clone
    + Sub<Output = Self>
    + ToPrimitive
    + Debug
    + SubAssign
    + AddAssign
{
    fn positive_or(self) -> Option<Self>;
}

// For the moment we only implement this for i32 values
// FIXME: should not be implemented for signed type!
impl PositiveWeight for i32 {
    fn positive_or(self) -> Option<Self> {
        if self >= 0 {
            Some(self)
        } else {
            None
        }
    }
}

// Structure encapsulating the indices associated with a box in the discrete solution space
#[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct BoxIndices<const NUM_CRITERIA: usize> {
    indices: [BoxIndex; NUM_CRITERIA],
}

impl<const NUM_CRITERIA: usize> BoxIndices<NUM_CRITERIA> {
    fn new(indices: &[BoxIndex]) -> Self {
        Self {
            indices: indices.try_into().unwrap(),
        }
    }
}
struct IterBoxIndices<'a, const NUM_CRITERIA: usize> {
    inner: Box<dyn Iterator<Item = BoxIndices<NUM_CRITERIA>> + 'a>,
}
impl<'a, const NUM_CRITERIA: usize> Iterator for IterBoxIndices<'a, NUM_CRITERIA> {
    type Item = BoxIndices<NUM_CRITERIA>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

trait SearchStrat<'a, const NUM_CRITERIA: usize> {
    fn gen_indices(
        &'a self,
        origin: &'a BoxIndices<NUM_CRITERIA>,
        dist: BoxIndex,
    ) -> IterBoxIndices<'a, NUM_CRITERIA>;
}

struct NeighborSearchStrat<const NUM_CRITERIA: usize> {
    nb_intervals: [BoxIndex; NUM_CRITERIA],
}

impl<'a, const NUM_CRITERIA: usize> SearchStrat<'a, NUM_CRITERIA>
    for NeighborSearchStrat<NUM_CRITERIA>
{
    fn gen_indices(
        &self,
        origin: &'a BoxIndices<NUM_CRITERIA>,
        dist: BoxIndex,
    ) -> IterBoxIndices<'a, NUM_CRITERIA> {
        let nb_criteria = NUM_CRITERIA;
        let mut left_bounds = [BoxIndex::zero(); NUM_CRITERIA];
        let mut right_bounds = [BoxIndex::zero(); NUM_CRITERIA];
        for criterion in 0..nb_criteria {
            left_bounds[criterion] = origin.indices[criterion];
            right_bounds[criterion] = self.nb_intervals[criterion] - origin.indices[criterion];
        }

        let mut rngs = Vec::new();
        for criterion in 0..nb_criteria {
            let min_rng = -cmp::min(dist, left_bounds[criterion]).to_isize().unwrap();
            let max_rng = cmp::min(dist, right_bounds[criterion]).to_isize().unwrap();
            let rng = min_rng..=max_rng;
            rngs.push(rng);
        }

        let indices_generator = rngs
            .into_iter()
            .multi_cartesian_product()
            .filter(move |indices| {
                indices.iter().map(|i| i.abs()).sum::<isize>() == dist.to_isize().unwrap()
            })
            .map(move |indices| {
                let mut box_indices = [BoxIndex::zero(); NUM_CRITERIA];
                (0..nb_criteria).for_each(|criterion| {
                    box_indices[criterion] =
                        (indices[criterion] + (origin.indices[criterion] as isize)) as BoxIndex;
                });
                BoxIndices {
                    indices: box_indices,
                }
            });

        IterBoxIndices {
            inner: Box::new(indices_generator.into_iter()),
        }
    }
}

//TODO:Update find_valid_move signature (refs and/or traits needs)
trait BoxHandler<'a, W: PositiveWeight, const NUM_CRITERIA: usize> {
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<NUM_CRITERIA>;
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<NUM_CRITERIA>,
        part_source: PartId,
        partition_imbalances: Vec<[W; NUM_CRITERIA]>,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: std::ops::Index<usize, Output = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;

    // Compute the regular delta associated with each criterion
    fn process_deltas(
        min_weights: impl IntoIterator<Item = W>,
        max_weights: impl IntoIterator<Item = W>,
        nb_intervals: impl IntoIterator<Item = BoxIndex>,
    ) -> Vec<f64> {
        min_weights
            .into_iter()
            .zip(max_weights)
            .zip(nb_intervals)
            .map(|((min_val, max_val), nb_interval)| {
                (max_val - min_val).to_f64().unwrap() / nb_interval.to_f64().unwrap()
            })
            .collect()
    }
}

struct RegularBoxHandler<W, const NUM_CRITERIA: usize>
where
    W: PositiveWeight,
{
    // Values related to the instance to be improved
    min_weights: [W; NUM_CRITERIA],
    // How many boxes for the discrete solution space
    nb_intervals: [BoxIndex; NUM_CRITERIA],
    // Discrete steps, depending on the criterion
    deltas: [f64; NUM_CRITERIA],
    // Mapping used to search moves of specific gain from a discretization of the cweight space
    pub boxes: BTreeMap<BoxIndices<NUM_CRITERIA>, Vec<CWeightId>>,
}

//FIXME:Refact this code to avoid clone calls.
impl<W, const NUM_CRITERIA: usize> RegularBoxHandler<W, NUM_CRITERIA>
where
    W: PositiveWeight,
{
    pub fn new<C, I>(cweights: C, nb_intervals: impl IntoIterator<Item = BoxIndex> + Clone) -> Self
    where
        C: IntoIterator<Item = I> + Clone,
        I: IntoIterator<Item = W>,
    {
        let boxes: BTreeMap<BoxIndices<NUM_CRITERIA>, Vec<CWeightId>> = BTreeMap::new();
        let mut cweights_iter = cweights.clone().into_iter();
        let first_cweight = cweights_iter.next().unwrap();
        let mut min_values = first_cweight.into_iter().collect::<Vec<_>>();
        let mut max_values = min_values.clone();

        for cweight in cweights_iter {
            cweight
                .into_iter()
                .zip(min_values.iter_mut())
                .zip(max_values.iter_mut())
                .for_each(|((current_val, min_val), max_val)| {
                    if current_val < *min_val {
                        *min_val = current_val.clone();
                    }
                    if current_val > *max_val {
                        *max_val = current_val.clone();
                    }
                })
        }

        let deltas =
            Self::process_deltas(min_values.clone(), max_values.clone(), nb_intervals.clone());

        let mut res = Self {
            min_weights: min_values.as_slice().try_into().unwrap(),
            nb_intervals: nb_intervals
                .into_iter()
                .collect::<Vec<_>>()
                .as_slice()
                .try_into()
                .unwrap(),
            deltas: deltas.as_slice().try_into().unwrap(),
            boxes,
        };

        cweights_iter = cweights.clone().into_iter();
        cweights_iter.enumerate().for_each(|(cweight_id, cweight)| {
            let indices = Self::box_indices(&res, cweight);
            match res.boxes.get_mut(&indices) {
                Some(vect_box_indices) => vect_box_indices.push(cweight_id),
                None => {
                    let vect_box_indices: Vec<usize> = vec![cweight_id];
                    res.boxes.insert(indices, vect_box_indices);
                }
            }
        });

        res
    }
}

impl<W, const NUM_CRITERIA: usize> Debug for RegularBoxHandler<W, NUM_CRITERIA>
where
    W: PositiveWeight,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Regular Box Handler with min_weights {:?}, nb_intervals {:?} and deltas {:?}",
            self.min_weights, self.nb_intervals, self.deltas
        )
    }
}

impl<'a, W: PositiveWeight, const NUM_CRITERIA: usize> BoxHandler<'a, W, NUM_CRITERIA>
    for RegularBoxHandler<W, NUM_CRITERIA>
where
    W: Sub<W, Output = W> + Zero + ToPrimitive,
{
    fn box_indices(&self, cweight: impl IntoIterator<Item = W>) -> BoxIndices<NUM_CRITERIA> {
        let res = BoxIndices::new(
            cweight
                .into_iter()
                .zip(self.min_weights.iter())
                .map(|(val, min)| val - *min)
                .zip(self.deltas.iter())
                .zip(self.nb_intervals.iter())
                .map(|((diff, delta), nb_interval)| match diff.positive_or() {
                    Some(_) => cmp::min(
                        (diff.to_f64().unwrap() / delta).floor() as BoxIndex,
                        *nb_interval - 1,
                    ),
                    None => 0u32,
                })
                .collect::<Vec<_>>()
                .as_slice(),
        );

        res
    }

    //FIXME:Allow partition imbalance to be composed of float values while cweights are integers
    //TODO:Create some struct encapsulating partition/solution state
    fn find_valid_move<CC, CP, CW>(
        &self,
        origin: &'a BoxIndices<NUM_CRITERIA>,
        part_source: PartId,
        partition_imbalances: Vec<[W; NUM_CRITERIA]>,
        partition: &'a CP,
        cweights: CC,
    ) -> Option<CWeightMove>
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: std::ops::Index<usize, Output = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let imbalances_iter = partition_imbalances.clone().into_iter();
        let max_imbalances = imbalances_iter
            .map(|criterion_imbalances| {
                let mut criterion_imbalances_iter = criterion_imbalances.into_iter();
                let first_part_imbalance = criterion_imbalances_iter.next().unwrap();
                match first_part_imbalance.positive_or() {
                    Some(val) => val,
                    None => criterion_imbalances_iter.next().unwrap(),
                }
            })
            .collect::<Vec<_>>();
        // let partition_imbalance: W = *max_imbalances.iter().max().unwrap();
        let partition_imbalance = max_imbalances
            .iter()
            .fold(None, |acc, &imbalance| match acc {
                None => Some(imbalance),
                Some(max) => {
                    let comparison = imbalance.partial_cmp(&max);
                    match comparison {
                        Some(Ordering::Greater) => Some(imbalance),
                        Some(Ordering::Less) | Some(Ordering::Equal) => Some(max),
                        None => acc,
                    }
                }
            })
            .unwrap();

        let strict_positive_gain = |id| {
            max_imbalances
                .iter()
                .zip(cweights[id].clone())
                .zip(partition_imbalances.clone())
                .map(|((max, cweight), criterion_imbalances)| {
                    *max == partition_imbalance
                        && cweight < (partition_imbalance + partition_imbalance)
                        || *max != partition_imbalance
                            && cweight
                                < (partition_imbalance - criterion_imbalances[1 - part_source])
                })
                .all(|valid_gain| valid_gain)
        };

        let candidates = self.boxes.get(&origin);
        let candidate_move = candidates
            .unwrap_or(&vec![])
            .iter()
            // Filter moves
            .filter(|id| partition[**id] == part_source)
            // Filter settled cweights, i.e. cweights whose move should leave
            // to a higher partition imbalance
            .find(|id| strict_positive_gain(**id))
            .map(|id| (*id, 1 - part_source))
            .clone();

        candidate_move
    }
}

trait Repartitioning<'a, W>
where
    W: PositiveWeight,
{
    // TODO: Add a builder for generic Repartitioning
    fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId>
            + std::ops::Index<usize, Output = PartId>
            + IndexMut<usize>
            + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;
}

trait PartitionImbalanceHandler<'a, W, const NUM_CRITERIA: usize>
where
    W: PositiveWeight,
{
    fn compute_imbalances<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> Vec<[W; NUM_CRITERIA]>
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone;

    // Return a couple composed of one of the most imbalanced criterion and
    // the part from which a weight should me moved.
    // FIXME:Add support to handle multiple criterion having the most imbalanced value
    fn process_imbalance<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> (CriterionId, PartId)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>;
}

#[derive(Debug)]
pub struct TargetorWIP<W, const NUM_CRITERIA: usize>
where
    W: PositiveWeight,
{
    // Instance data
    nb_intervals: [BoxIndex; NUM_CRITERIA],
    parts_target_loads: Vec<[W; NUM_CRITERIA]>,

    // // Partition state
    // partition: Vec<PartId>,
    // box_handler: Box<dyn BoxHandler<'a, T, W>>,
    box_handler: Option<RegularBoxHandler<W, NUM_CRITERIA>>,
}

impl<'a, W: PositiveWeight, const NUM_CRITERIA: usize>
    PartitionImbalanceHandler<'a, W, NUM_CRITERIA> for TargetorWIP<W, NUM_CRITERIA>
where
    W: PositiveWeight,
{
    fn compute_imbalances<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> Vec<[W; NUM_CRITERIA]>
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone,
    {
        let mut res = [[W::zero(); NUM_CRITERIA]; 2];
        for criterion in 0..NUM_CRITERIA {
            for part in 0..2 {
                res[criterion][part] -= self.parts_target_loads[criterion][part];
            }
        }

        let cweights_iter = cweights.clone().into_iter();
        partition
            .clone()
            .into_iter()
            .zip(cweights_iter)
            .for_each(|(part, cweight)| {
                cweight
                    .into_iter()
                    .enumerate()
                    .for_each(|(criterion, weight)| res[criterion][part] += weight)
            });
        res.to_vec()
    }

    /* Function returning a couple composed of the most imbalanced criterion
    and the part from which a weight should me moved.
     */
    fn process_imbalance<CC, CP, CW>(
        &self,
        partition: &'a CP,
        cweights: CC,
    ) -> (CriterionId, PartId)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CP: IntoIterator<Item = PartId> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let imbalances = self.compute_imbalances(partition, cweights);
        let mut max_imb = W::zero();
        let mut res: (CriterionId, PartId) = (CriterionId::zero(), PartId::zero());

        imbalances
            .into_iter()
            .enumerate()
            .for_each(|(criterion, loads)| {
                let max = if loads[0] >= W::zero() {
                    loads[0]
                } else {
                    loads[1]
                };
                if max > max_imb {
                    let part = 1 - (loads[0] >= loads[1]) as usize;
                    res = (criterion, part);
                    max_imb = max;
                }
            });

        res
    }
}

impl<W: PositiveWeight, const NUM_CRITERIA: usize> TargetorWIP<W, NUM_CRITERIA> {
    //FIXME:Allow partition imbalance to be composed of float values while cweights are integers
    pub fn new<CC, CT, CW>(
        // partition: Vec<PartId>,
        // cweights: CC,
        // Targetor specific parameter
        nb_intervals: CT,
        parts_target_loads: CC,
    ) -> Self
    where
        CC: IntoIterator<Item = CW> + Clone,
        CT: IntoIterator<Item = BoxIndex> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        // let res_rbh = RegularBoxHandler::new(cweights, nb_intervals.clone());
        let res_target_loads: Vec<[W; NUM_CRITERIA]> = parts_target_loads
            .into_iter()
            .map(|criterion_target_loads| {
                criterion_target_loads
                    .into_iter()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        Self {
            nb_intervals: nb_intervals
                .into_iter()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            parts_target_loads: res_target_loads,
            box_handler: None,
        }
    }

    pub fn setup_default_box_handler<CC, CW>(&mut self, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let box_handler = RegularBoxHandler::new(cweights, self.nb_intervals.clone());
        self.box_handler = Some(box_handler);
    }
}

impl<'a, W: PositiveWeight, const NUM_CRITERIA: usize> Repartitioning<'a, W>
    for TargetorWIP<W, NUM_CRITERIA>
{
    fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    where
        CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
        CP: IntoIterator<Item = PartId>
            + std::ops::Index<usize, Output = PartId>
            + IndexMut<usize>
            + Clone,
        CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        // let box_handler = match self.box_handler {
        //     Some(bh) => self.box_handler.as_ref().unwrap(),
        //     None => &RegularBoxHandler::new(cweights, self.nb_intervals.clone()),
        // };

        if self.box_handler.is_none() {
            self.setup_default_box_handler(cweights.clone());
        }
        let box_handler = self.box_handler.as_ref().unwrap();

        let mut look_for_movement = true;
        while look_for_movement {
            // Setup target part and most imbalanced criteria
            let (most_imbalanced_criterion, part_source) =
                self.process_imbalance(partition, cweights.clone());

            // Setup search strat
            let search_strat = NeighborSearchStrat {
                nb_intervals: self.nb_intervals.clone(),
            };

            // Setup target gain
            let partition_imbalances = self.compute_imbalances(partition, cweights.clone());
            let nb_criteria = partition_imbalances.len();
            let target_gain: Vec<W> = partition_imbalances
                .clone()
                .into_iter()
                .map(
                    |criterion_imbalances| match criterion_imbalances[part_source].positive_or() {
                        Some(val) => val,
                        None => W::zero(),
                    },
                )
                .collect();

            let origin = box_handler.box_indices(target_gain);
            let option_valid_move = box_handler.find_valid_move(
                &origin,
                part_source,
                partition_imbalances.clone().to_vec(),
                partition,
                cweights.clone(),
            );

            if let Some((id_cweight, target_part)) = option_valid_move {
                partition[id_cweight] = target_part;
            } else {
                let mut increase_offset = true;
                let mut offset = 1 as BoxIndex;
                while increase_offset {
                    let iter_indices = search_strat.gen_indices(&origin, offset);
                    let option_valid_move = iter_indices
                        .into_iter()
                        .map(|box_indices| {
                            box_handler.find_valid_move(
                                &box_indices,
                                part_source,
                                partition_imbalances.clone(),
                                partition,
                                cweights.clone(),
                            )
                        })
                        .find_map(|option_valid_move| option_valid_move);
                    if let Some((id_cweight, target_part)) = option_valid_move {
                        increase_offset = false;
                        partition[id_cweight] = target_part;
                    } else {
                        offset += 1;
                        let partition_imbalance =
                            partition_imbalances[most_imbalanced_criterion][part_source];
                        let bound_indices =
                            box_handler.box_indices(vec![partition_imbalance; nb_criteria]);
                        increase_offset = (0..nb_criteria).all(|criterion| {
                            (origin.indices[criterion] as isize - offset as isize).is_positive()
                                || origin.indices[criterion] + offset
                                    <= bound_indices.indices[criterion]
                        });
                    }
                }
                look_for_movement = false;
            }
        }
    }
}

impl<W: PositiveWeight, CC, CW, const NUM_CRITERIA: usize> crate::Partition<CC>
    for TargetorWIP<W, NUM_CRITERIA>
where
    CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
{
    // impl<'a, T: PositiveInteger, W: PositiveWeight, CC> crate::Partition<CC> for TargetorWIP<T, W>
    // where
    //     CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    // {
    type Metadata = usize;
    type Error = Error;

    // fn optimize<CC, CP, CW>(&mut self, partition: &'a mut CP, cweights: CC)
    // where
    //     CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
    //     CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        // part_ids: &mut CP,
        cweights: CC,
    ) -> Result<Self::Metadata, Self::Error>
// where
    //     // CC: IntoIterator<Item = CW> + Clone + std::ops::Index<usize, Output = CW>,
    //     CP: IntoIterator<Item = PartId> + Clone + std::ops::Index<usize, Output = PartId>,
    //     // CW: IntoIterator<Item = W> + Clone + std::ops::Index<usize, Output = W>,
    {
        let partition_len = part_ids.iter().count();
        let weights_len = cweights.clone().into_iter().count();
        if partition_len != weights_len {
            return Err(Error::InputLenMismatch {
                expected: partition_len,
                actual: weights_len,
            });
        }

        if 1 < part_ids.iter_mut().max().map(|&mut part| part).unwrap_or(0) {
            return Err(Error::BiPartitioningOnly);
        }

        let mut vec_partition = part_ids.to_vec();
        self.optimize(&mut vec_partition, cweights);

        Ok(0)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    struct Instance {
        pub cweights: Vec<[i32; 2]>,
        pub nb_intervals: Vec<u32>,
    }

    impl Instance {
        fn create_instance() -> Self {
            let out = Self {
                cweights: vec![[2, 2], [4, 6], [6, 4], [8, 5]],
                nb_intervals: vec![3u32, 2u32],
            };

            out
        }
    }

    #[test]
    fn check_regular_box_handler() {
        let instance = Instance::create_instance();

        // Split with steps 1.0 on the first criterion and 0.5 on the second one.
        let rbh = RegularBoxHandler::new(instance.cweights.clone(), instance.nb_intervals);
        let mut expected_box_indices: Vec<Vec<u32>> = Vec::with_capacity(3);
        expected_box_indices.extend([vec![0, 0], vec![1, 1], vec![2, 1], vec![2, 1]]);

        expected_box_indices
            .iter()
            .zip(instance.cweights.clone())
            .for_each(|(box_indices, cweight)| {
                let values = rbh.box_indices(cweight);
                assert!(
                    box_indices
                        .iter()
                        .zip(values.indices.iter())
                        .all(|(expected_val, computed_val)| expected_val == computed_val),
                    "Indices are not equal {:?}, {:?} ",
                    box_indices,
                    values
                );
            });

        let part_source = 0;
        let partition = vec![part_source; instance.cweights.len()];
        let partition_imbalances = vec![[10, -10], [9, -8]];
        let space_box_indices = vec![
            BoxIndices::new(&[0, 0]),
            BoxIndices::new(&[1, 0]),
            BoxIndices::new(&[2, 0]),
            BoxIndices::new(&[0, 1]),
            BoxIndices::new(&[1, 1]),
            BoxIndices::new(&[2, 1]),
        ];
        let iter_box_indices = space_box_indices.iter();

        let expected_moves = vec![Some((0, 1)), None, None, None, Some((1, 1)), Some((2, 1))];
        expected_moves
            .iter()
            .zip(iter_box_indices)
            .for_each(|(expected_move, box_indices)| {
                let candidate_move = rbh.find_valid_move(
                    &box_indices,
                    part_source,
                    partition_imbalances.clone(),
                    &partition.clone(),
                    instance.cweights.clone(),
                );
                assert_eq!(
                    *expected_move, candidate_move,
                    "Moves are not equal. Expected {:?} but returned {:?} ",
                    *expected_move, candidate_move
                );
            });
    }

    #[test]
    fn check_targetor_without_search() {
        let instance = Instance::create_instance();
        let mut partition = vec![0; instance.cweights.len()];
        let rbh: RegularBoxHandler<i32, 2> =
            RegularBoxHandler::new(instance.cweights.clone(), instance.nb_intervals);
        let partition_target_loads = vec![vec![10, 10], vec![9, 8]];

        let mut targetor: TargetorWIP<i32, 2> = TargetorWIP::new(
            // partition.clone(),
            rbh.nb_intervals.clone(),
            // instance.cweights.clone(),
            partition_target_loads,
        );

        targetor.optimize(&mut partition, instance.cweights.clone());
        let expected_partition_res = vec![0, 1, 1, 0];
        // assert!(
        //     targetor.partition == expected_partition_res,
        //     "Partition are not equal. Expected {:?}, returned {:?}",
        //     expected_partition_res,
        //     targetor.partition
        // );
        assert_eq!(
            partition, expected_partition_res,
            "Partition are not equal. Expected {:?}, returned {:?}",
            expected_partition_res, partition
        );
    }

    #[test]
    fn check_targetor_with_search() {
        let instance = Instance::create_instance();
        let mut partition = vec![0; instance.cweights.len()];

        // Custom made data triggering exploration on the bottom right of the
        // weight space.
        let nb_intervals = [6, 2];
        let partition_target_loads = vec![[12, 8], [15, 2]];

        let rbh: RegularBoxHandler<i32, 2> =
            RegularBoxHandler::new(instance.cweights.clone(), nb_intervals);

        let mut targetor: TargetorWIP<i32, 2> = TargetorWIP::new(
            // partition.clone(),
            rbh.nb_intervals.clone(),
            // instance.cweights.clone(),
            partition_target_loads,
        );

        targetor.optimize(&mut partition, instance.cweights.clone());
        let expected_partition_res = vec![0, 0, 0, 1];
        assert_eq!(
            partition, expected_partition_res,
            "Partition are not equal. Expected {:?}, returned {:?}",
            expected_partition_res, partition
        );
    }
}
