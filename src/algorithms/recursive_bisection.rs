use super::Error;
use crate::geometry::OrientedBoundingBox;
use crate::geometry::PointND;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::ToTypenum;
use rayon::prelude::*;
use std::cmp;
use std::iter::Sum;
use std::mem;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

struct Item<'p, const D: usize, W> {
    point: PointND<D>,
    weight: W,
    part: &'p AtomicUsize,
}

fn rcb_split<const D: usize, W>(
    items: &mut [Item<D, W>],
    coord: usize,
    tolerance: f64,
    mut min: f64,
    mut max: f64,
    sum: W,
) -> usize
where
    W: RcbWeight,
{
    let mut prev_count_left = usize::MAX;
    loop {
        let split_target = (min + max) / 2.0;
        let (count_left, weight_left) = items
            .par_iter()
            .filter(|item| item.point[coord] < split_target)
            .fold(
                || (0, W::default()),
                |(count, weight), item| (count + 1, weight + item.weight),
            )
            .reduce(
                || (0, W::default()),
                |(count0, weight0), (count1, weight1)| (count0 + count1, weight0 + weight1),
            );

        let imbalance = {
            let ideal_weight_left = sum.to_f64().unwrap() / 2.0;
            let weight_left = weight_left.to_f64().unwrap();
            f64::abs((weight_left - ideal_weight_left) / ideal_weight_left)
        };
        if count_left == prev_count_left || imbalance < tolerance {
            return count_left;
        }
        prev_count_left = count_left;

        let weight_right = sum - weight_left;
        if weight_left < weight_right {
            min = split_target;
        } else {
            max = split_target;
        }
    }
}

fn rcb_recurse<const D: usize, W>(
    items: &mut [Item<D, W>],
    iter_count: usize,
    iter_id: usize,
    coord: usize,
    tolerance: f64,
) where
    W: RcbWeight,
{
    if items.is_empty() {
        // Would make min/max computation panic.
        return;
    }
    if iter_count == 0 {
        items
            .into_par_iter()
            .for_each(|item| item.part.store(iter_id, Ordering::Relaxed));
        return;
    }

    let sum: W = items.par_iter().map(|item| item.weight).sum();
    let (min, max) = items
        .par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), item| {
                (
                    f64::min(min, item.point[coord]),
                    f64::max(max, item.point[coord]),
                )
            },
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min0, max0), (min1, max1)| (f64::min(min0, min1), f64::max(max0, max1)),
        );

    let split_idx = rcb_split(items, coord, tolerance, min, max, sum);
    let (left, right) = if split_idx == items.len() {
        items.split_at_mut(items.len())
    } else {
        let (left, _, _right_minus_one) = items
            .select_nth_unstable_by(split_idx, |item1, item2| {
                f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
            });
        let left_len = left.len();
        items.split_at_mut(left_len)
    };

    rayon::join(
        || {
            rcb_recurse(
                left,
                iter_count - 1,
                2 * iter_id + 1,
                (coord + 1) % D,
                tolerance,
            )
        },
        || {
            rcb_recurse(
                right,
                iter_count - 1,
                2 * iter_id + 2,
                (coord + 1) % D,
                tolerance,
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
    P::Iter: rayon::iter::IndexedParallelIterator,
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
    if partition.is_empty() {
        // Would make the partition.min() at the end panic.
        return Ok(());
    }

    let mut items: Vec<_> = points
        .zip(weights)
        .zip(unsafe { mem::transmute::<&mut [usize], &[AtomicUsize]>(&mut *partition) })
        .map(|((point, weight), part)| Item {
            point,
            weight,
            part,
        })
        .collect();

    rcb_recurse(&mut items, iter_count, 0, 0, tolerance);

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
    Self: Sum + PartialOrd + num::ToPrimitive,
    Self: Add<Output = Self> + Sub<Output = Self> + AddAssign,
{
}

impl<T> RcbWeight for T
where
    Self: Copy + std::fmt::Debug + Default + Send + Sync,
    Self: Sum + PartialOrd + num::ToPrimitive,
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
///     .partition(&mut partition, (points, weights))
///     .unwrap();
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
/// ```
///
/// # Reference
///
/// Berger, M. J. and Bokhari, S. H., 1987. A partitioning strategy for
/// nonuniform problems on multiprocessors. *IEEE Transactions on Computers*,
/// C-36(5):570–580. <doi:10.1109/TC.1987.1676942>.
#[derive(Default)]
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
    P::Iter: rayon::iter::IndexedParallelIterator,
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
///     .partition(&mut partition, (&points, weights))
///     .unwrap();
///
/// // The two points at the top are in the same part.
/// assert_eq!(partition[0], partition[1]);
///
/// // The two points at the bottom are in the same part.
/// assert_eq!(partition[2], partition[3]);
///
/// // There are two different parts.
/// assert_ne!(partition[1], partition[2]);
/// ```
///
/// # Reference
///
/// Williams, Roy D., 1991. Performance of dynamic load balancing algorithms for
/// unstructured mesh calculations. *Concurrency: Practice and Experience*,
/// 3(5):457–481. <doi:10.1002/cpe.4330030502>.
#[derive(Default)]
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
