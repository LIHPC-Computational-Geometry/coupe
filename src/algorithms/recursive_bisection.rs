use crate::geometry::Mbr;
use crate::geometry::PointND;

use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;

fn rcb_recurse<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
    partition: &mut Vec<ProcessUniqueId>,
    current_part: ProcessUniqueId,
    current_coord: usize,
    tolerance: f64,
) {
    if n_iter == 0 {
        return;
    }

    let sum: f64 = weights
        .par_iter()
        .zip(partition.as_slice())
        .filter(|(_weight, weight_part)| **weight_part == current_part)
        .map(|(weight, _weight_part)| *weight)
        .sum();

    let max_imbalance = tolerance * sum;

    let (min, max) = points
        .par_iter()
        .zip(partition.as_slice())
        .filter(|(_point, point_part)| **point_part == current_part)
        .map(|(point, _point_part)| point[current_coord])
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), projection| (f64::min(min, projection), f64::max(max, projection)),
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(min0, max0), (min1, max1)| (f64::min(min0, min1), f64::max(max0, max1)),
        );

    let mut split_min = min;
    let mut split_max = max;
    let mut split_target = (split_max + split_min) / 2.0;

    let mut prev_weight_left = 0.0;

    loop {
        let (weight_left, weight_right) = weights
            .par_iter()
            .zip(points)
            .enumerate()
            // TODO zip
            .filter(|(weight_id, (_, _))| partition[*weight_id] == current_part)
            .map(|(_id, (weight, point))| {
                let point = point[current_coord];
                if point < split_target {
                    (*weight, 0.0)
                } else {
                    (0.0, *weight)
                }
            })
            .reduce(
                || (0.0, 0.0),
                |(weight_left0, weight_right0), (weight_left1, weight_right1)| {
                    (weight_left0 + weight_left1, weight_right0 + weight_right1)
                },
            );

        let imbalance = f64::abs(weight_left - weight_right);
        if imbalance < max_imbalance || prev_weight_left == weight_left {
            break;
        }

        if weight_left < weight_right {
            split_min = split_target;
        } else {
            split_max = split_target;
        }
        split_target = (split_max + split_min) / 2.0;

        prev_weight_left = weight_left;
    }

    let new_part = ProcessUniqueId::new();
    partition
        .par_iter_mut()
        .zip(points)
        .filter(|(point_part, point)| {
            **point_part == current_part && point[current_coord] < split_target
        })
        .for_each(|(point_part, _point)| *point_part = new_part);

    let next_coord = (current_coord + 1) % D;
    rcb_recurse(
        points,
        weights,
        n_iter - 1,
        partition,
        current_part,
        next_coord,
        tolerance,
    );
    rcb_recurse(
        points,
        weights,
        n_iter - 1,
        partition,
        new_part,
        next_coord,
        tolerance,
    );
}

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    let len = weights.len();
    let initial_id = ProcessUniqueId::new();
    let mut partition = vec![initial_id; len];

    rcb_recurse(points, weights, n_iter, &mut partition, initial_id, 0, 0.05);

    partition
}

// pub because it is also useful for multijagged and required for benchmarks
pub fn axis_sort<const D: usize>(
    points: &[PointND<D>],
    permutation: &mut [usize],
    current_coord: usize,
) {
    permutation.par_sort_by(|i1, i2| {
        if points[*i1][current_coord] < points[*i2][current_coord] {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
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
pub fn rib<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    let mbr = Mbr::from_points(points);

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
        let weights = vec![1.; 8];
        let points = vec![
            Point2D::from([-1.3, 6.]),
            Point2D::from([2., -4.]),
            Point2D::from([1., 1.]),
            Point2D::from([-3., -2.5]),
            Point2D::from([-1.3, -0.3]),
            Point2D::from([2., 1.]),
            Point2D::from([-3., 1.]),
            Point2D::from([1.3, -2.]),
        ];

        let partition = rcb(&points, &weights, 2);

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

    #[test]
    fn test_rcb() {
        use std::collections::HashMap;

        let points: Vec<Point2D> = (0..40)
            .map(|_| Point2D::new(rand::random(), rand::random()))
            .collect();
        let weights: Vec<f64> = (0..points.len()).map(|_| rand::random()).collect();
        let partition = rcb(&points, &weights, 2);
        let mut loads: HashMap<ProcessUniqueId, f64> = HashMap::new();
        let mut sizes: HashMap<ProcessUniqueId, usize> = HashMap::new();
        for (weight_id, part) in partition.iter().enumerate() {
            let weight = weights[weight_id];
            *loads.entry(*part).or_default() += weight;
            *sizes.entry(*part).or_default() += 1;
        }
        for ((part, load), size) in loads.iter().zip(sizes.values()) {
            println!("{:?} -> {}:{}", part, size, load);
        }
    }
}
