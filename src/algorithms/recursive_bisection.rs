use crate::geometry::*;

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::{DimDiff, DimSub};
use nalgebra::DefaultAllocator;
use nalgebra::DimName;
use nalgebra::U1;

use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

/// # Recursive Coordinate Bisection algorithm
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
/// ## Inputs
/// - `ids`: global identifiers of the objects to partition
/// - `weights`: weights corsponding to a cost relative to the objects
/// - `coordinates`: the N-D coordinates of the objects to partition
///
/// ## Output
/// A Vec of couples `(usize, ProcessUniqueId)`
///
/// the first component of each couple is the id of an object and
/// the second component is the id of the partition to which that object was assigned
pub fn rcb<D>(points: &[PointND<D>], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    let len = weights.len();
    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(len)
        .collect::<Vec<_>>();

    rcb_recurse(
        &points,
        &weights,
        &mut permutation,
        &AtomicPtr::new(initial_partition.as_mut_ptr()),
        n_iter,
        0,
    );
    initial_partition
}

pub fn rcb_recurse<D>(
    points: &[PointND<D>],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
    n_iter: usize,
    current_coord: usize,
) where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    if n_iter == 0 {
        // No iteration left. The current
        // ids become a part of the final partition.
        // Generate a partition id and return.
        let part_id = ProcessUniqueId::new();
        permutation.par_iter().for_each(|idx| {
            let ptr = partition.load(atomic::Ordering::Relaxed);

            // Unsafe usage explanation:
            //
            // In this implementation, the partition is represented as
            // a contiguous array of ids. It is allocated once, and modified in place.
            // Neither the array nor a slice of it is ever copied. When recursing, a pointer to
            // the array is passed to children functions, which both have mutable access to the
            // partition array from different threads (That's why the pointer is wrapped in a
            // Arc<AtomicPtr<T>>). It is not possible to have shared mutable access to memory
            // across (one/several) threads in safe code. However, the raw pointer is indexed
            // through the permutation array which contains valid indices and is not shared across
            // children calls/threads. This ensures that ptr.add(*idx) is valid memory and every
            // element of the partition array will be written on exactly once.
            unsafe { std::ptr::write(ptr.add(*idx), part_id) }
        });
    } else {
        // We split the objects in two parts of equal weights
        // The split is perfomed alongside the x or y axis,
        // alternating at each iteration.

        // We first need to sort the objects w.r.t. x or y position
        axis_sort(points, permutation, current_coord);

        // We then seek the split position
        let split_pos = half_weight_pos_permu(weights, permutation);
        let (left_permu, right_permu) = permutation.split_at_mut(split_pos);

        // Once the split is performed
        // we recursively iterate by calling
        // the algorithm on the two generated parts.
        // In the next iteration, the split aixs will
        // be orthogonal to the current one

        let dim = D::dim();
        rayon::join(
            || {
                rcb_recurse(
                    points,
                    weights,
                    left_permu,
                    partition,
                    n_iter - 1,
                    (current_coord + 1) % dim,
                )
            },
            || {
                rcb_recurse(
                    points,
                    weights,
                    right_permu,
                    partition,
                    n_iter - 1,
                    (current_coord + 1) % dim,
                )
            },
        );
    }
}

// pub because it is also useful for multijagged and required for benchmarks
pub fn axis_sort<D>(points: &[PointND<D>], permutation: &mut [usize], current_coord: usize)
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
{
    permutation.par_sort_by(|i1, i2| {
        if points[*i1][current_coord] < points[*i2][current_coord] {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
}

// Computes a slice index which splits
// the slice in two parts of equal weights
// i.e. sorted_weights[..idx].sum() == sorted_weights[idx..].sum
fn half_weight_pos_permu(weights: &[f64], permutation: &[usize]) -> usize {
    let half_weight = permutation.par_iter().map(|idx| weights[*idx]).sum::<f64>() / 2.;

    let mut current_weight_idx;
    let mut current_weight_sum = 0.;

    let mut scan = permutation
        .par_iter()
        .enumerate()
        .fold_with((std::usize::MAX, 0.), |(low, acc), (idx, val)| {
            if idx < low {
                (idx, acc + weights[*val])
            } else {
                (low, acc + weights[*val])
            }
        })
        .collect::<Vec<_>>()
        .into_iter();

    // above this, the code was parallel
    // what follows is sequential

    loop {
        let current = scan.next().unwrap();
        if current_weight_sum + current.1 > half_weight {
            current_weight_idx = current.0;
            break;
        }

        current_weight_sum += current.1;
    }

    // seek from current_weight_idx
    while current_weight_sum < half_weight {
        current_weight_idx += 1;
        current_weight_sum += weights[permutation[current_weight_idx]];
    }

    current_weight_idx
}

/// # Recursive Inertia Bisection algorithm
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
/// ## Inputs
/// - `ids`: global identifiers of the objects to partition
/// - `weights`: weights corsponding to a cost relative to the objects
/// - `coordinates`: the 2D coordinates of the objects to partition
///
/// ## Output
/// A Vec of couples `(usize, ProcessUniqueId)`
///
/// the first component of each couple is the id of an object and
/// the second component is the id of the partition to which that object was assigned
///
/// The main difference with the RCB algorithm is that, in RCB, points are split
/// with a separator which is parallel to either the x axis or the y axis. With RIB,
/// The global shape of the data is first considered and the separator is computed to
/// be parallel to the inertia axis of the global shape, which aims to lead to better shaped
/// partitions.
pub fn rib<D>(points: &[PointND<D>], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId>
where
    D: DimName + DimSub<U1>,
    DefaultAllocator: Allocator<f64, D>
        + Allocator<f64, D, D>
        + Allocator<f64, U1, D>
        + Allocator<f64, DimDiff<D, U1>>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, D, D>>::Buffer: Send + Sync,
{
    let mbr = Mbr::from_points(&points);

    let points = points
        .par_iter()
        .map(|p| mbr.mbr_to_aabb(p))
        .collect::<Vec<_>>();

    // When the rotation is done, we just apply RCB
    rcb(&points, weights, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_point_sample() -> Vec<Point2D> {
        vec![
            Point2D::new(4., 6.),
            Point2D::new(9., 5.),
            Point2D::new(-1.2, 7.),
            Point2D::new(0., 0.),
            Point2D::new(3., 9.),
            Point2D::new(-4., 3.),
            Point2D::new(1., 2.),
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
}
