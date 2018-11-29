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
use geometry::{Mbr2D, Point2D, Quadrant};

use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

pub fn z_curve_partition(
    points: &[Point2D],
    num_partitions: usize,
    order: u32,
) -> Vec<ProcessUniqueId> {
    let mbr = Mbr2D::from_points(points.iter());
    let initial_id = ProcessUniqueId::new();
    let mut ininial_partition: Vec<_> = points.par_iter().map(|_| initial_id).collect();
    let mut permutation: Vec<_> = (0..points.len()).into_par_iter().collect();

    z_curve_partition_recurse(
        points,
        order,
        &mbr,
        &mut permutation,
        &AtomicPtr::from(ininial_partition.as_mut_ptr()),
    );

    let points_per_partition = points.len() / num_partitions;

    let atomic_handle = AtomicPtr::from(ininial_partition.as_mut_ptr());
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

fn z_curve_partition_recurse(
    points: &[Point2D],
    order: u32,
    mbr: &Mbr2D,
    permu: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
) {
    if order == 0 || permu.len() <= 1 {
        return;
    }

    let quadrants = points
        .par_iter()
        .map(|p| mbr.quadrant(p).unwrap_or(Quadrant::BottomLeft))
        .collect::<Vec<_>>();

    // only 4 different elements
    // use pdqsort to break equal elements pattern (O(N)-ish??)
    permu.par_sort_unstable_by_key(|idx| quadrants[*idx] as u8);

    // find split positions in ordered array
    let p1 = permu
        .binary_search_by(|idx| {
            if (quadrants[*idx] as u8) < 1 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }).unwrap_err();

    let p2 = permu
        .binary_search_by(|idx| {
            if (quadrants[*idx] as u8) < 2 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }).unwrap_err();

    let p3 = permu
        .binary_search_by(|idx| {
            if (quadrants[*idx] as u8) < 3 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }).unwrap_err();

    let slices = split_at_mut_many(permu, &[p1, p2, p3]);
    slices.into_par_iter().enumerate().for_each(|(i, slice)| {
        let quadrant = match i {
            0 => Quadrant::BottomLeft,
            1 => Quadrant::BottomRight,
            2 => Quadrant::TopLeft,
            3 => Quadrant::TopRight,
            _ => unreachable!(),
        };

        z_curve_partition_recurse(points, order - 1, &mbr.sub_mbr(quadrant), slice, partition);
    })
}

pub(crate) fn z_curve_reorder(points: &[Point2D], order: u32) -> Vec<usize> {
    let mut permu: Vec<_> = (0..points.len()).into_par_iter().collect();
    z_curve_reorder_permu(points, permu.as_mut_slice(), order);
    permu
}

pub(crate) fn z_curve_reorder_permu(points: &[Point2D], permu: &mut [usize], order: u32) {
    let mbr = Mbr2D::from_points(points.iter());
    let hashes = permu
        .par_iter()
        .map(|idx| compute_hash(&points[*idx], order, &mbr))
        .collect::<Vec<_>>();

    permu.par_sort_by_key(|idx| hashes[*idx]);
}

fn compute_hash(point: &Point2D, order: u32, mbr: &Mbr2D) -> usize {
    let current_hash = mbr
        .quadrant(point)
        .expect("Cannot compute the z-hash of a point outside of the current Mbr.");

    if order == 0 {
        current_hash as usize
    } else {
        4u32.pow(order) as usize * current_hash as usize
            + compute_hash(point, order - 1, &mbr.sub_mbr(current_hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reorder() {
        let mut points = vec![
            Point2D::new(0., 0.),
            Point2D::new(20., 10.),
            Point2D::new(0., 10.),
            Point2D::new(20., 0.),
            Point2D::new(14., 7.),
            Point2D::new(4., 7.),
            Point2D::new(14., 2.),
            Point2D::new(4., 2.),
        ];

        let indices = z_curve_reorder(points.as_mut_slice(), 2);
        println!("{:?}", indices);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 7);
        assert_eq!(indices[2], 6);
        assert_eq!(indices[3], 3);
        assert_eq!(indices[4], 5);
        assert_eq!(indices[5], 2);
        assert_eq!(indices[6], 4);
        assert_eq!(indices[7], 1);
    }

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
