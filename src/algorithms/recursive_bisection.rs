use super::Error;
use crate::geometry::OrientedBoundingBox;
use crate::geometry::PointND;
use crate::BoundingBox;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::ToTypenum;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::cmp;
use std::iter::Sum;
use std::mem::MaybeUninit;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// Taken from the stdlib
///
/// TODO remove this once stabilized.
/// Tracking issue: <https://github.com/rust-lang/rust/issues/96097>
///
/// # Safety
///
/// Do not call with unhabited types.
unsafe fn array_assume_init<T, const N: usize>(array: [MaybeUninit<T>; N]) -> [T; N] {
    // SAFETY:
    // * The caller guarantees that all elements of the array are initialized
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout
    // * `MaybeUninit` does not drop, so there are no double-frees
    // And thus the conversion is safe
    //std::intrinsics::assert_inhabited::<[T; N]>();
    (&array as *const _ as *const [T; N]).read()
}

fn array_init<F, T, const N: usize>(mut f: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    unsafe {
        let mut output: [MaybeUninit<T>; N] = MaybeUninit::uninit().assume_init();
        for (i, u) in output.iter_mut().enumerate() {
            u.write(f(i));
        }
        array_assume_init(output)
    }
}

fn array_map_mut<'a, F, T, U, const N: usize>(input: &'a mut [T; N], mut map: F) -> [U; N]
where
    F: FnMut(&'a mut T) -> U,
    U: 'a,
{
    unsafe {
        let mut output: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        for (u, t) in output.iter_mut().zip(input) {
            u.write(map(t));
        }
        array_assume_init(output)
    }
}

fn array_unzip<F, T, U, V, const N: usize>(input: [T; N], mut map: F) -> ([U; N], [V; N])
where
    F: FnMut(T) -> (U, V),
{
    unsafe {
        let mut output1: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        let mut output2: [MaybeUninit<V>; N] = MaybeUninit::uninit().assume_init();
        for ((u, v), t) in output1.iter_mut().zip(&mut output2).zip(input) {
            let (new_u, new_v) = map(t);
            u.write(new_u);
            v.write(new_v);
        }
        (array_assume_init(output1), array_assume_init(output2))
    }
}

fn array_unzip_mut<'a, F, T, U, V, const N: usize>(
    input: &'a mut [T; N],
    mut map: F,
) -> ([U; N], [V; N])
where
    F: FnMut(&'a mut T) -> (U, V),
    U: 'a,
    V: 'a,
{
    unsafe {
        let mut output1: [MaybeUninit<U>; N] = MaybeUninit::uninit().assume_init();
        let mut output2: [MaybeUninit<V>; N] = MaybeUninit::uninit().assume_init();
        for ((u, v), t) in output1.iter_mut().zip(&mut output2).zip(input) {
            let (new_u, new_v) = map(t);
            u.write(new_u);
            v.write(new_v);
        }
        (array_assume_init(output1), array_assume_init(output2))
    }
}

struct Items<'a, const D: usize, W> {
    points: [&'a mut [f32]; D],
    weights: &'a mut [W],
    parts: &'a mut [&'a AtomicUsize],
}

// TODO replace with derive once [T; D] implements Default for all D.
impl<'a, const D: usize, W> Default for Items<'a, D, W> {
    fn default() -> Self {
        Self {
            points: [(); D].map(|_| {
                let a: &mut [f32] = &mut [];
                a
            }),
            weights: &mut [],
            parts: &mut [],
        }
    }
}

/// Split the arrays in `items` into the ones that have `points[i][coord]`
/// strictly lower than `items.point[pivot][coord]` and the others.
fn reorder_split<const D: usize, W>(
    mut items: Items<'_, D, W>,
    pivot: usize,
    coord: usize,
) -> (Items<'_, D, W>, Items<'_, D, W>) {
    for ps in &mut items.points {
        ps.swap(0, pivot);
    }
    items.weights.swap(0, pivot);
    items.parts.swap(0, pivot);

    let (pivot, mut points) = array_unzip_mut(&mut items.points, |p| p.split_at_mut(1));
    let pivot = pivot[coord][0];
    let (_, weights) = items.weights.split_at_mut(1);
    let (_, parts) = items.parts.split_at_mut(1);

    let mut l = 0;
    let mut r = points[0].len();
    loop {
        unsafe {
            let coords = points.get_unchecked(coord);
            while l < r && *coords.get_unchecked(l) < pivot {
                l += 1;
            }
            while l < r && pivot <= *coords.get_unchecked(r - 1) {
                r -= 1;
            }
            if r <= l {
                break;
            }
            r -= 1;
            for p in &mut points {
                p.swap(l, r);
            }
            weights.swap(l, r);
            parts.swap(l, r);
            l += 1;
        }
    }

    for p in &mut items.points {
        p.swap(0, l);
    }
    items.weights.swap(0, l);
    items.parts.swap(0, l);
    let (points_left, points_right) = array_unzip(items.points, |p| p.split_at_mut(l));
    let (weights_left, weights_right) = items.weights.split_at_mut(l);
    let (parts_left, parts_right) = items.parts.split_at_mut(l);
    let left = Items {
        points: points_left,
        weights: weights_left,
        parts: parts_left,
    };
    let right = Items {
        points: points_right,
        weights: weights_right,
        parts: parts_right,
    };
    (left, right)
}

/// Return value of [par_rcb_split].
struct SplitResult<'a, const D: usize, W> {
    left: Items<'a, D, W>,
    right: Items<'a, D, W>,
    /// Weight of the left part, used to compute the sum for the next iteration.
    weight_left: W,
    /// Coordinate values of the split, used to compute the [BoundingBox]es.
    left_max_bound: f32,
    right_min_bound: f32,
}

impl<'a, const D: usize, W> Default for SplitResult<'a, D, W>
where
    W: Default,
{
    fn default() -> Self {
        Self {
            left: Items::default(),
            right: Items::default(),
            weight_left: W::default(),
            left_max_bound: f32::NEG_INFINITY,
            right_min_bound: f32::INFINITY,
        }
    }
}

/// Accumulation result of [par_rcb_split].
struct Accumulation<W> {
    /// Number of points on the left of split_target.
    count_left: usize,
    /// Weight of the points on the left of split_target.
    weight_left: W,
    /// Weight of the points on the right of split_target.
    weight_right: W,
    /// Closest point to split_target while on the left of split_target.
    closest_left_idx: Option<usize>,
    /// Left point's distance to split target.
    closest_left_dist: f32,
    /// Closest point to split_target while on the right of split_target.
    closest_right_idx: Option<usize>,
    /// Right point's distance to split target.
    closest_right_dist: f32,
}

impl<W> Default for Accumulation<W>
where
    W: Default,
{
    fn default() -> Self {
        Self {
            count_left: 0,
            weight_left: W::default(),
            weight_right: W::default(),
            closest_left_idx: None,
            closest_left_dist: f32::INFINITY,
            closest_right_idx: None,
            closest_right_dist: f32::INFINITY,
        }
    }
}

impl<W> Accumulation<W>
where
    W: RcbWeight,
{
    fn add_point(mut self, i: usize, diff_with_target: f32, weight: W) -> Self {
        if diff_with_target < 0.0 {
            // Point is on the left.
            let distance = -diff_with_target;
            self.count_left += 1;
            self.weight_left += weight;
            if distance < self.closest_left_dist {
                self.closest_left_idx = Some(i);
                self.closest_left_dist = distance;
            }
        } else {
            // Point is on the right.
            let distance = diff_with_target;
            self.weight_right += weight;
            if distance < self.closest_right_dist {
                self.closest_right_idx = Some(i);
                self.closest_right_dist = distance;
            }
        }
        self
    }

    fn merge(a: Self, b: Self) -> Self {
        let closest_left_idx;
        let closest_left_dist;
        if a.closest_left_dist < b.closest_left_dist {
            closest_left_idx = a.closest_left_idx;
            closest_left_dist = a.closest_left_dist;
        } else {
            closest_left_idx = b.closest_left_idx;
            closest_left_dist = b.closest_left_dist;
        }
        let closest_right_idx;
        let closest_right_dist;
        if a.closest_right_dist < b.closest_right_dist {
            closest_right_idx = a.closest_right_idx;
            closest_right_dist = a.closest_right_dist;
        } else {
            closest_right_idx = b.closest_right_idx;
            closest_right_dist = b.closest_right_dist;
        }
        Self {
            count_left: a.count_left + b.count_left,
            weight_left: a.weight_left + b.weight_left,
            weight_right: a.weight_right + b.weight_right,
            closest_left_idx,
            closest_left_dist,
            closest_right_idx,
            closest_right_dist,
        }
    }
}

/// Splits the given items into two sets of similar weights (parallel version).
fn par_rcb_split<const D: usize, W>(
    items: Items<'_, D, W>,
    coord: usize,
    tolerance: f64,
    mut min: f32,
    mut max: f32,
    sum: W,
) -> SplitResult<'_, D, W>
where
    W: RcbWeight,
{
    let span = tracing::info_span!("par_rcb_split");
    let _enter = span.enter();

    let ideal_part_weight = sum.to_f64().unwrap() / 2.0;
    let max_part_weight = W::from_f64(ideal_part_weight * (1.0 + tolerance)).unwrap();
    let mut allow_interrupts = true;

    loop {
        let split_target = (min + max) / 2.0;

        let res = items.points[coord]
            .par_iter()
            .with_min_len(4096)
            .zip(&*items.weights)
            .enumerate()
            .try_fold(Accumulation::default, |acc, (idx, (point, weight))| {
                Ok(acc.add_point(idx, point - split_target, *weight))
            })
            .try_reduce(Accumulation::default, |acc0, acc1| {
                let acc = Accumulation::merge(acc0, acc1);
                if allow_interrupts {
                    if max_part_weight < acc.weight_left {
                        return Err(acc);
                    }
                    if max_part_weight < acc.weight_right {
                        return Err(acc);
                    }
                }
                Ok(acc)
            });

        let partial_res = match res {
            Ok(res) => {
                // Both parts are below max_part_weight.
                let closest_left_idx = match res.closest_left_idx {
                    Some(v) => v,
                    None => {
                        // If there is no point below split_target, then items
                        // should be empty?
                        return SplitResult {
                            left: items,
                            weight_left: sum,
                            left_max_bound: split_target,
                            right_min_bound: split_target,
                            ..SplitResult::default()
                        };
                    }
                };
                let closest_left = split_target - res.closest_left_dist;
                let closest_right = split_target + res.closest_right_dist;
                let (left, right) = reorder_split(items, closest_left_idx, coord);
                return SplitResult {
                    left,
                    right,
                    weight_left: res.weight_left,
                    left_max_bound: closest_left,
                    right_min_bound: closest_right,
                };
            }
            // Otherwise, some part is too big, and partial_res has the results
            // of the run which was interrupted early.
            Err(partial_res) => partial_res,
        };

        if partial_res.weight_left < partial_res.weight_right {
            min = split_target;
        } else {
            max = split_target;
        }

        if approx::abs_diff_eq!(min, max) {
            // Special case for when a lot of weights share the same coordinate.
            // We might not be able to respect the imbalance constraints.
            allow_interrupts = false;
        }
    }
}

fn rcb_recurse<const D: usize, W>(
    items: Items<'_, D, W>,
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
    sum: W,
    bb: BoundingBox<D>,
) where
    W: RcbWeight,
{
    if items.parts.is_empty() {
        return;
    }
    if iter_count == 0 {
        let span = tracing::info_span!(
            "rcb_recurse: apply_part_id",
            item_count = items.parts.len(),
            iter_id,
        );
        let _enter = span.enter();

        items
            .parts
            .into_par_iter()
            .for_each(|part| part.store(iter_id, Ordering::Relaxed));
        return;
    }

    let min = bb.p_min[coord] as f32;
    let max = bb.p_max[coord] as f32;
    let SplitResult {
        left,
        right,
        weight_left,
        left_max_bound,
        right_min_bound,
    } = par_rcb_split(items, coord, tolerance, min, max, sum);

    let mut bb_left = bb.clone();
    bb_left.p_max[coord] = left_max_bound as f64;
    let mut bb_right = bb;
    bb_right.p_min[coord] = right_min_bound as f64;

    rayon::join(
        || {
            rcb_recurse(
                left,
                iter_count - 1,
                2 * iter_id + 1,
                (coord + 1) % D,
                tolerance,
                weight_left,
                bb_left,
            )
        },
        || {
            rcb_recurse(
                right,
                iter_count - 1,
                2 * iter_id + 2,
                (coord + 1) % D,
                tolerance,
                sum - weight_left,
                bb_right,
            )
        },
    );
}

fn rcb<const D: usize, P, W>(
    partition: &mut [usize],
    points: P,
    weights: W,
    iter_count: usize,
    tolerance: f64,
) -> Result<(), Error>
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator + Clone,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let points = points.into_par_iter();
    let weights = weights.into_par_iter();

    if weights.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: weights.len(),
        });
    }
    if points.len() != partition.len() {
        return Err(Error::InputLenMismatch {
            expected: partition.len(),
            actual: points.len(),
        });
    }

    let mut coords = array_init(|coord| {
        points
            .clone()
            .map(|point| point[coord] as f32)
            .collect::<Vec<f32>>()
    });
    let mut weights: Vec<_> = weights.collect();

    let atomic_partition = unsafe {
        // Rust does not seem to have a strict aliasing rule like C does, so the
        // transmute here looks safe, but we still need to ensure partition is
        // properly aligned for atomic types. While this should always be the
        // case, better safe than sorry.
        let (before, partition, after) = partition.align_to_mut::<AtomicUsize>();
        assert!(before.is_empty() && after.is_empty());
        &*partition
    };
    let mut atomic_partition: Vec<&AtomicUsize> = atomic_partition.par_iter().collect();
    let sum = weights.par_iter().cloned().sum();
    let bb = match BoundingBox::from_points(points) {
        Some(v) => v,
        None => return Ok(()), // `items` is empty.
    };

    let points = array_map_mut(&mut coords, |coord| &mut coord[..]);
    let items = Items {
        points,
        weights: &mut weights,
        parts: &mut atomic_partition,
    };
    rcb_recurse(items, iter_count, 0, 0, tolerance, sum, bb);

    // Part IDs must start from zero.
    let part_id_offset = *partition.par_iter().min().unwrap();
    partition
        .par_iter_mut()
        .for_each(|part_id| *part_id -= part_id_offset);

    Ok(())
}

/// Trait alias for values accepted as weights by [Rcb] and [Rib].
pub trait RcbWeight
where
    Self: Copy + std::fmt::Debug + Default + Send + Sync,
    Self: Sum + PartialOrd + ToPrimitive + FromPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self> + AddAssign,
{
}

impl<T> RcbWeight for T
where
    Self: Copy + std::fmt::Debug + Default + Send + Sync,
    Self: Sum + PartialOrd + ToPrimitive + FromPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self> + AddAssign,
{
}

/// # Recursive Coordinate Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
///
/// This is the most simple and straightforward geometric algorithm. It operates
/// as follows for a N-dimensional set of points:
///
/// At each iteration, select a vector `n` of the canonical basis
/// `(e_0, ..., e_{n-1})`. Then, split the set of points with an hyperplane
/// orthogonal to `n`, such that the two parts of the splits are evenly
/// weighted. Finally, recurse by reapplying the algorithm to the two parts with
/// an other normal vector selection.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points = [
///     Point2D::new(1., 1.),
///     Point2D::new(-1., 1.),
///     Point2D::new(1., -1.),
///     Point2D::new(-1., -1.),
/// ];
/// let weights = [1; 4];
/// let mut partition = [0; 4];
///
/// // Generate a partition of 4 parts (2 splits).
/// coupe::Rcb { iter_count: 2, ..Default::default() }
///     .partition(&mut partition, (points, weights))?;
///
/// // All points are in different parts.
/// for i in 0..4 {
///     for j in 0..4 {
///         if j == i {
///             continue
///         }
///         assert_ne!(partition[i], partition[j])
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Berger, M. J. and Bokhari, S. H., 1987. A partitioning strategy for
/// nonuniform problems on multiprocessors. *IEEE Transactions on Computers*,
/// C-36(5):570–580. <doi:10.1109/TC.1987.1676942>.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rcb {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    ///
    /// If this equals zero, RCB will not do anything.
    pub iter_count: usize,

    /// Tolerance on the normalized imbalance, for each split.  Please note that
    /// the overall imbalance might end up above this threshold.
    ///
    /// Negative values are interpreted as zeroes.
    pub tolerance: f64,
}

impl<const D: usize, P, W> crate::Partition<(P, W)> for Rcb
where
    P: rayon::iter::IntoParallelIterator<Item = PointND<D>>,
    P::Iter: rayon::iter::IndexedParallelIterator + Clone,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (P, W),
    ) -> Result<Self::Metadata, Self::Error> {
        rcb(part_ids, points, weights, self.iter_count, self.tolerance)
    }
}

// pub because it is also useful for multijagged and required for benchmarks
pub fn axis_sort<const D: usize>(
    points: &[PointND<D>],
    permutation: &mut [usize],
    current_coord: usize,
) {
    permutation.par_sort_unstable_by(|i1, i2| {
        if points[*i1][current_coord] < points[*i2][current_coord] {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    })
}

fn rib<const D: usize, W>(
    partition: &mut [usize],
    points: &[PointND<D>],
    weights: W,
    n_iter: usize,
    tolerance: f64,
) -> Result<(), Error>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    let obb = match OrientedBoundingBox::from_points(points) {
        Some(v) => v,
        None => return Ok(()),
    };
    let points = points.par_iter().map(|p| obb.obb_to_aabb(p));
    // When the rotation is done, we just apply RCB
    rcb(partition, points, weights, n_iter, tolerance)
}

/// # Recursive Inertial Bisection algorithm
///
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
///
/// A variant of the [Recursive Coordinate Bisection algorithm][crate::Rcb]
/// where a basis change is performed beforehand so that the first coordinate of
/// the new basis is colinear to the inertia axis of the set of points. This has
/// the goal of producing better shaped partition than [RCB][crate::Rcb].
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// // Here, the inertia axis is the y axis.
/// // We thus expect Rib to split horizontally first.
/// let points = [
///     Point2D::new(1., 10.),
///     Point2D::new(-1., 10.),
///     Point2D::new(1., -10.),
///     Point2D::new(-1., -10.),
/// ];
/// let weights = [1; 4];
/// let mut partition = [0; 4];
///
/// // Generate a partition of 2 parts (1 split).
/// coupe::Rib { iter_count: 1, ..Default::default() }
///     .partition(&mut partition, (&points, weights))?;
///
/// // The two points at the top are in the same part.
/// assert_eq!(partition[0], partition[1]);
///
/// // The two points at the bottom are in the same part.
/// assert_eq!(partition[2], partition[3]);
///
/// // There are two different parts.
/// assert_ne!(partition[1], partition[2]);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Williams, Roy D., 1991. Performance of dynamic load balancing algorithms for
/// unstructured mesh calculations. *Concurrency: Practice and Experience*,
/// 3(5):457–481. <doi:10.1002/cpe.4330030502>.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rib {
    /// The number of iterations of the algorithm. This will yield a partition
    /// of at most `2^num_iter` parts.
    pub iter_count: usize,

    /// Same meaning as [`Rcb::tolerance`].
    pub tolerance: f64,
}

impl<'a, const D: usize, W> crate::Partition<(&'a [PointND<D>], W)> for Rib
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    W: rayon::iter::IntoParallelIterator,
    W::Item: RcbWeight,
    W::Iter: rayon::iter::IndexedParallelIterator,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&'a [PointND<D>], W),
    ) -> Result<Self::Metadata, Self::Error> {
        rib(part_ids, points, weights, self.iter_count, self.tolerance)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools as _;
    use proptest::prelude::*;

    use super::*;
    use crate::geometry::Point2D;

    fn gen_point_sample() -> Vec<Point2D> {
        vec![
            Point2D::from([4., 6.]),
            Point2D::from([9., 5.]),
            Point2D::from([-1.2, 7.]),
            Point2D::from([0., 0.]),
            Point2D::from([3., 9.]),
            Point2D::from([-4., 3.]),
            Point2D::from([1., 2.]),
        ]
    }

    proptest!(
        #[test]
        fn test_reorder_split(
            points in (1..24_usize).prop_flat_map(|point_count| {
                prop::collection::vec(prop::num::f32::POSITIVE, point_count)
            }),
        ) {
            for index in 0..points.len() {
                let mut p0 = points.clone();
                let mut weights = vec![1_i32; points.len()];
                let part = AtomicUsize::new(0);
                let mut parts = vec![&part; points.len()];
                let items = Items {
                    points: [&mut p0],
                    weights: &mut weights,
                    parts: &mut parts,
                };
                let pivot = points[index];
                let (left, right) = reorder_split(items, index, 0);
                prop_assert!(left.points[0].iter().all(|l| *l < pivot));
                prop_assert!(right.points[0].iter().all(|r| pivot <= *r));
                for e in &points {
                    prop_assert_ne!(
                        left.points[0].contains(e),
                        right.points[0].contains(e),
                        "{} is either missing or in both sides",
                        e,
                    );
                }
            }
        }
    );

    #[test]
    fn test_axis_sort_x() {
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort(&points, &mut permutation, 0);

        assert_eq!(permutation, vec![5, 2, 3, 6, 4, 0, 1]);
    }

    #[test]
    fn test_axis_sort_y() {
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort(&points, &mut permutation, 1);

        assert_eq!(permutation, vec![3, 6, 5, 1, 0, 2, 4]);
    }

    proptest!(
        #[test]
        fn test_par_rcb_split(
            (mut points, mut weights) in (1..200_usize)
                .prop_flat_map(|point_count| {
                    (
                        prop::collection::vec(0.0..1.0_f32, point_count),
                        prop::collection::vec(1..1000_u32, point_count),
                    )
                })
        ) {
            let (&min, &max) = points.iter().minmax().into_option().unwrap();
            let sum: u32 = weights.iter().sum();

            let part = AtomicUsize::new(0);
            let mut parts = vec![&part; points.len()];
            let items = Items {
                points: [&mut points],
                weights: &mut weights,
                parts: &mut parts,
            };

            let SplitResult {
                left,
                right,
                weight_left,
                left_max_bound: split_pos,
                ..
            } = par_rcb_split(items, 0, 0.05, min, max, sum);

            prop_assert!(left.points[0].iter().all(|l| *l < split_pos));
            prop_assert!(right.points[0].iter().all(|r| split_pos <= *r));
            prop_assert_eq!(weight_left, left.weights.iter().sum());
            prop_assert_eq!(sum - weight_left, right.weights.iter().sum());
        }
    );

    #[test]
    fn test_rcb_basic() {
        let weights = [1.; 8];
        let points = [
            Point2D::from([-1.3, 6.]),
            Point2D::from([2., -4.]),
            Point2D::from([1., 1.]),
            Point2D::from([-3., -2.5]),
            Point2D::from([-1.3, -0.3]),
            Point2D::from([2., 1.]),
            Point2D::from([-3., 1.]),
            Point2D::from([1.3, -2.]),
        ];

        let mut partition = [0; 8];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1) // make the test deterministic
            .build()
            .unwrap()
            .install(|| rcb(&mut partition, points, weights, 2, 0.05))
            .unwrap();

        assert_eq!(partition[0], partition[6]);
        assert_eq!(partition[1], partition[7]);
        assert_eq!(partition[2], partition[5]);
        assert_eq!(partition[3], partition[4]);

        let (p_id1, p_id2, p_id3, p_id4) = (partition[0], partition[1], partition[2], partition[3]);

        let p1 = partition.iter().filter(|p_id| **p_id == p_id1);
        let p2 = partition.iter().filter(|p_id| **p_id == p_id2);
        let p3 = partition.iter().filter(|p_id| **p_id == p_id3);
        let p4 = partition.iter().filter(|p_id| **p_id == p_id4);

        assert_eq!(p1.count(), 2);
        assert_eq!(p2.count(), 2);
        assert_eq!(p3.count(), 2);
        assert_eq!(p4.count(), 2);
    }

    //#[test] // Disabled by default because of its need for a random source.
    fn _test_rcb_rand() {
        use std::collections::HashMap;

        let points: Vec<Point2D> = (0..40000)
            .map(|_| Point2D::from([rand::random(), rand::random()]))
            .collect();
        let weights: Vec<f64> = (0..points.len()).map(|_| rand::random()).collect();

        let mut partition = vec![0; points.len()];
        rcb(&mut partition, points, weights.par_iter().cloned(), 3, 0.05).unwrap();

        let mut loads: HashMap<usize, f64> = HashMap::new();
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for (weight_id, part) in partition.iter().enumerate() {
            let weight = weights[weight_id];
            *loads.entry(*part).or_default() += weight;
            *sizes.entry(*part).or_default() += 1;
        }
        for ((part, load), size) in loads.iter().zip(sizes.values()) {
            println!("{part:?} -> {size}:{load:.1}");
        }
    }
}
