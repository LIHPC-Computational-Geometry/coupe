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
use geometry::{Mbr, PointND};

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::base::U1;
use nalgebra::DefaultAllocator;
use nalgebra::DimName;

use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

pub fn z_curve_partition<D>(
    points: &[PointND<D>],
    num_partitions: usize,
    order: u32,
) -> Vec<ProcessUniqueId>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    let max_order = (f64::from(std::u32::MAX)).log(f64::from(2u32.pow(D::dim() as u32))) as u32;
    assert!(
        order <= max_order,
        format!("Cannot use the z-curve partition algorithm with an order > {} because it would currently overflow hashes capacity", max_order)
    );

    // Mbr used to construct Point hashes
    let mbr = Mbr::from_points(&points);

    let mut permutation: Vec<_> = (0..points.len()).into_par_iter().collect();

    let initial_id = ProcessUniqueId::new();
    let mut ininial_partition: Vec<_> = points.par_iter().map(|_| initial_id).collect();

    // reorder points
    z_curve_partition_recurse(points, order, &mbr, &mut permutation);

    let points_per_partition = points.len() / num_partitions;

    let atomic_handle = AtomicPtr::from(ininial_partition.as_mut_ptr());
    // give an id to each partition
    permutation
        .par_chunks(points_per_partition)
        .for_each(|chunk| {
            let id = ProcessUniqueId::new();
            let ptr = atomic_handle.load(atomic::Ordering::Relaxed);
            for idx in chunk {
                unsafe { std::ptr::write(ptr.add(*idx), id) }
            }
        });

    ininial_partition
}

// reorders `permu` to sort points by increasing z-curve hash
fn z_curve_partition_recurse<D>(
    points: &[PointND<D>],
    order: u32,
    mbr: &Mbr<D>,
    permu: &mut [usize],
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
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

    let mut split_positions = (1..2usize.pow(D::dim() as u32)).collect::<Vec<_>>();
    for n in split_positions.iter_mut() {
        *n = permu
            .binary_search_by(|idx| {
                if (regions[*idx] as u8) < *n as u8 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }).unwrap_err();
    }

    let slices = split_at_mut_many(permu, &split_positions);
    slices.into_par_iter().enumerate().for_each(|(i, slice)| {
        z_curve_partition_recurse(points, order - 1, &mbr.sub_mbr(i as u32), slice);
    })
}

// reorders a slice of Point3D in increasing z-curve order
#[allow(unused)]
pub(crate) fn z_curve_reorder<D>(points: &[PointND<D>], order: u32) -> Vec<usize>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    let max_order = (f64::from(std::u32::MAX)).log(f64::from(2u32.pow(D::dim() as u32))) as u32;
    assert!(
        order <= max_order,
        format!("Cannot use the z-curve partition algorithm with an order > {} because it would currently overflow hashes capacity", max_order)
    );

    let mut permu: Vec<_> = (0..points.len()).into_par_iter().collect();
    z_curve_reorder_permu(points, permu.as_mut_slice(), order);
    permu
}

// reorders a slice of indices such that the associated array of Point3D is sorted
// by increasing z-order
#[allow(unused)]
pub(crate) fn z_curve_reorder_permu<D>(points: &[PointND<D>], permu: &mut [usize], order: u32)
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    let mbr = Mbr::from_points(&points);
    let hashes = permu
        .par_iter()
        .map(|idx| compute_hash(&points[*idx], order, &mbr))
        .collect::<Vec<_>>();

    permu.par_sort_by_key(|idx| hashes[*idx]);
}

fn compute_hash<D>(point: &PointND<D>, order: u32, mbr: &Mbr<D>) -> usize
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    let current_hash = mbr
        .region(point)
        .expect("Cannot compute the z-hash of a point outside of the current Mbr.");

    if order == 0 {
        current_hash as usize
    } else {
        // TODO: this can overflow if (2^dim)^order > u32::MAX
        // maybe we should use a BigInt
        (2usize.pow(D::dim() as u32)).pow(order) * current_hash as usize
            + compute_hash(point, order - 1, &mbr.sub_mbr(current_hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::Point2D;

    #[test]
    fn test_partition() {
        let points = vec![
            Point2D::new(0., 0.),
            Point2D::new(20., 10.),
            Point2D::new(0., 10.),
            Point2D::new(20., 0.),
            Point2D::new(14., 7.),
            Point2D::new(4., 7.),
            Point2D::new(14., 2.),
            Point2D::new(4., 2.),
        ];

        let ids = z_curve_partition(&points, 4, 1);
        for id in ids.iter() {
            println!("{}", id);
        }
        assert_eq!(ids[0], ids[7]);
        assert_eq!(ids[1], ids[4]);
        assert_eq!(ids[2], ids[5]);
        assert_eq!(ids[3], ids[6]);
    }
}
