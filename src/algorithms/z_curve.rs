//! An implementation of the Z-order space filling curve.
//! It aims to reorder a set of points with spacial hashing.
//!
//! First, a Minimal Bounding rectangle is constructed from a given set of points.
//! The Mbr is then recursively split following a quadtree refinement scheme until each cell
//! contains at most one point. Then, a spatial binary hash is determined for each point as follows.
//!
//! The hash of a point is initially 0 and for each new split required to isolate it,
//! the new hash is computed by `new_hash = 4 * previous_hash + b` where `0 <= b <= 3`.
//! `b` is chosen by looking up in which quadrant of the current Mbr the point is. The mapping is defined as follows:
//!
//!   - `BottomLeft => 0`
//!   - `BottomRight => 1`
//!   - `TopLeft => 2`
//!   - `TopRight => 3`
//!
//! Finally, the points are reordered according to the order of their hash.

use super::multi_jagged::split_at_mut_many;
use crate::geometry::OrientedBoundingBox;
use crate::PointND;

use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::ToTypenum;
use rayon::prelude::*;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

// Z-curve hash can get quite large. For instance,
// in 2D, an order greater than 64 will overflow u128.
// maybe it would be more appropriate to use a BigInt
type HashType = u128;
const HASH_TYPE_MAX: HashType = u128::MAX;

fn z_curve_partition<const D: usize>(
    partition: &mut [usize],
    points: &[PointND<D>],
    part_count: usize,
    order: u32,
) where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    debug_assert_eq!(partition.len(), points.len());

    let max_order = (HASH_TYPE_MAX as f64).log(f64::from(1 << D)) as u32;
    assert!(
        order <= max_order,
        "Cannot use the z-curve partition algorithm with an order > {} because it would currently overflow hashes capacity",
        max_order,
    );

    // Bounding box used to construct Point hashes
    let obb = match OrientedBoundingBox::from_points(points) {
        Some(v) => v,
        None => return,
    };

    let mut permutation: Vec<_> = (0..points.len()).into_par_iter().collect();

    // reorder points
    z_curve_partition_recurse(points, order, &obb, &mut permutation);

    let points_per_partition = points.len() / part_count;
    let remainder = points.len() % part_count;

    let atomic_handle = AtomicPtr::from(partition.as_mut_ptr());

    // give an id to each partition
    //
    // instead of calling par_chunks with points_per_partition, which could yield an extra
    // undesired partition, or a quite high partition imbalance, we compute an index threshold
    // that splits chunks of len points_per_partition and point_per_partition + 1.
    // Doing so makes sure there is always the correct amount of chunks, and that they are not too imbalanced.
    let threshold_idx = (points_per_partition + 1) * remainder;
    permutation[..threshold_idx]
        .par_chunks(points_per_partition + 1)
        .chain(permutation[threshold_idx..].par_chunks(points_per_partition))
        .enumerate()
        .for_each(|(id, chunk)| {
            let ptr = atomic_handle.load(atomic::Ordering::Relaxed);
            for idx in chunk {
                unsafe { std::ptr::write(ptr.add(*idx), id) }
            }
        });
}

// reorders `permu` to sort points by increasing z-curve hash
fn z_curve_partition_recurse<const D: usize>(
    points: &[PointND<D>],
    order: u32,
    mbr: &OrientedBoundingBox<D>,
    permu: &mut [usize],
) {
    // we stop recursion if there is only 1 point left to avoid useless calls
    if order == 0 || permu.len() <= 1 {
        return;
    }

    // compute the quadrant in which each point is.
    // default to dummy value for points outside of the current mbr
    let regions = points
        .par_iter()
        .map(|p| mbr.region(p).unwrap_or(0))
        .collect::<Vec<_>>();

    // use pdqsort to break equal elements pattern
    permu.par_sort_unstable_by_key(|idx| regions[*idx] as u8);

    // Now we need to split the permutation array in 2^dim
    // such that each subslice contains only points from the same quadrant
    // instead of traversing the whole array, we can just perform a few binary searches
    // to find the split positions since the array is already sorted

    let mut split_positions = (1..2usize.pow(D as u32)).collect::<Vec<_>>();
    for n in split_positions.iter_mut() {
        *n = permu
            .binary_search_by(|idx| {
                if (regions[*idx] as u8) < *n as u8 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err();
    }

    let slices = split_at_mut_many(permu, &split_positions);
    slices.into_par_iter().enumerate().for_each(|(i, slice)| {
        z_curve_partition_recurse(points, order - 1, &mbr.sub_mbr(i as u32), slice);
    })
}

/// # Z space-filling curve algorithm
///
/// The Z-curve uses space hashing to partition points. The points in the same part of a partition
/// have the same Z-hash. This hash is computed by recursively constructing a N-dimensional region tree.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), std::convert::Infallible> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points = [
///     Point2D::new(0., 0.),
///     Point2D::new(1., 1.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 9.),
///     Point2D::new(9., 1.),
///     Point2D::new(10., 0.),
///     Point2D::new(10., 10.),
///     Point2D::new(9., 9.),
/// ];
/// let mut partition = [0; 8];
///
/// // generate a partition of 4 parts
/// coupe::ZCurve { part_count: 4, order: 5 }
///     .partition(&mut partition, &points)?;
///
/// assert_eq!(partition[0], partition[1]);
/// assert_eq!(partition[2], partition[3]);
/// assert_eq!(partition[4], partition[5]);
/// assert_eq!(partition[6], partition[7]);
/// # Ok(())
/// # }
/// ```  
#[derive(Clone, Copy, Debug)]
pub struct ZCurve {
    pub part_count: usize,
    pub order: u32,
}

impl<'a, const D: usize> crate::Partition<&'a [PointND<D>]> for ZCurve
where
    Const<D>: DimSub<Const<1>> + ToTypenum,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    type Metadata = ();
    type Error = std::convert::Infallible;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        points: &'a [PointND<D>],
    ) -> Result<Self::Metadata, Self::Error> {
        z_curve_partition(part_ids, points, self.part_count, self.order);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Point2D;

    #[test]
    fn test_partition() {
        let points = [
            Point2D::from([0., 0.]),
            Point2D::from([20., 10.]),
            Point2D::from([0., 10.]),
            Point2D::from([20., 0.]),
            Point2D::from([14., 7.]),
            Point2D::from([4., 7.]),
            Point2D::from([14., 2.]),
            Point2D::from([4., 2.]),
        ];

        let mut ids = [0; 8];
        z_curve_partition(&mut ids, &points, 4, 1);
        for id in ids {
            println!("{}", id);
        }
        assert_eq!(ids[0], ids[7]);
        assert_eq!(ids[1], ids[4]);
        assert_eq!(ids[2], ids[5]);
        assert_eq!(ids[3], ids[6]);
    }
}
