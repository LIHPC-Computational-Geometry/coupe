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

#[derive(Clone)]
struct Item<const D: usize> {
    initial_idx: usize,
    point: PointND<D>,
    weight: f64,
    part_id: ProcessUniqueId,
}

// items musn't be empty.
fn weighted_median_aux<const D: usize>(
    mut items: &mut [Item<D>],
    coord: usize,
    tolerance: f64,
) -> usize {
    let mut offset = 0;
    let sum: f64 = items.par_iter().map(|item| item.weight).sum();

    let mut sup_weight_lower = 0.0;
    loop {
        const MAX_SORT: usize = 64;
        let items_len = items.len();
        if items_len < MAX_SORT {
            items.sort_unstable_by(|item1, item2| {
                f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
            });
            let sum: f64 = items.par_iter().map(|item| item.weight).sum();
            let mut half = 0.0;
            let mut half_idx = 0;
            for item in items {
                if sum < half * 2.0 {
                    break;
                }
                half_idx += 1;
                half += item.weight;
            }
            if half_idx == items_len {
                half_idx -= 1;
            }
            return offset + half_idx;
        }

        let median_idx = items.len() / 2;
        let (lower, median, upper) = items
            .select_nth_unstable_by(items.len() / 2, |item1, item2| {
                f64::partial_cmp(&item1.point[coord], &item2.point[coord]).unwrap()
            });
        let lower_sum = sup_weight_lower + lower.par_iter().map(|item| item.weight).sum::<f64>();
        let upper_sum = sum - lower_sum - median.weight;
        let lower_sum_norm = lower_sum / sum;
        let upper_sum_norm = upper_sum / sum;

        if f64::abs(upper_sum_norm - lower_sum_norm) < tolerance {
            break offset + median_idx;
        } else if 0.5 < lower_sum_norm {
            items = lower;
        } else {
            items = upper;
            sup_weight_lower += lower_sum;
            offset += median_idx;
        }
    }
}

fn weighted_median<const D: usize>(
    mut items: &mut [Item<D>],
    coord: usize,
    tolerance: f64,
) -> (&mut [Item<D>], &mut [Item<D>]) {
    if items.is_empty() {
        return (&mut [], &mut []);
    }
    let median = weighted_median_aux(&mut items, coord, tolerance);
    items.split_at_mut(median)
}

fn rcb_recurse<const D: usize>(items: &mut [Item<D>], n_iter: usize, coord: usize, tolerance: f64) {
    if n_iter == 0 {
        let part_id = ProcessUniqueId::new();
        for item in items {
            item.part_id = part_id;
        }
        return;
    }

    let (left, right) = weighted_median(items, coord, tolerance);
    let coord = (coord + 1) % D;

    rayon::join(
        || rcb_recurse(left, n_iter - 1, coord, tolerance),
        || rcb_recurse(right, n_iter - 1, coord, tolerance),
    );
}

pub fn rcb<const D: usize>(
    points: &[PointND<D>],
    weights: &[f64],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    let initial_id = ProcessUniqueId::new();
    let mut items: Vec<_> = points
        .iter()
        .zip(weights)
        .enumerate()
        .map(|(initial_idx, (&point, &weight))| Item {
            initial_idx,
            point,
            weight,
            part_id: initial_id,
        })
        .collect();

    rcb_recurse(&mut items, n_iter, 0, 0.05);

    let mut partition = vec![initial_id; items.len()];
    for item in items {
        partition[item.initial_idx] = item.part_id;
    }

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
    fn test_weighted_median() {
        let mut items: Vec<_> = gen_point_sample()
            .into_iter()
            .enumerate()
            .map(|(i, point)| Item {
                point,
                weight: rand::random(),
                initial_idx: i,
                part_id: ProcessUniqueId::new(),
            })
            .collect();
        let (left, right) = weighted_median(&mut items, 0, 0.01);
        let left: Vec<_> = left
            .to_vec()
            .into_iter()
            .map(|item| (item.point, item.weight))
            .collect();
        let right: Vec<_> = right
            .to_vec()
            .into_iter()
            .map(|item| (item.point, item.weight))
            .collect();
        let weight_left: f64 = left.iter().map(|(_point, weight)| weight).sum();
        let weight_right: f64 = right.iter().map(|(_point, weight)| weight).sum();
        let items: Vec<_> = items
            .into_iter()
            .map(|item| (item.point, item.weight))
            .collect();
        println!("total: {:?}", items);
        println!("left {} {:?}", weight_left, left);
        println!("right {} {:?}", weight_right, right);
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
