mod bounding_box;
mod grid;
mod point_nd;

use crate::bounding_box::BBox2D;
use crate::point_nd::Point2D;

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;

fn find_weighted_median(weights: &[i32]) -> (usize, i32) {
    let ideal_partition_weight: i32 = weights.iter().sum::<i32>() / 2;

    let mut idx: usize = 0;
    let weighted_median = weights
        .iter()
        .fold_while(0, |acc, x| {
            if acc >= ideal_partition_weight {
                Done(acc)
            } else {
                idx += 1;
                Continue(acc + x)
            }
        })
        .into_inner();

    (idx, weighted_median)
}

fn rcb(
    level: usize,
    points: &[Point2D],
    weights: &[i32],
    part_ids: &mut [usize],
    axis: usize,
    _bbox: Option<BBox2D>,
) {
    if level == 0 {
        return;
    }

    let _points_1d: Vec<i32> = points.iter().map(|point| point[axis]).collect();

    let (pivot, _weighted_median) = find_weighted_median(weights);
    // dbg!(level, pivot, weighted_median);

    // Compute lower
    let bbox_lower = BBox2D::from_points(&points[..pivot]);
    rcb(
        level - 1,
        &points[..pivot],
        &weights[..pivot],
        &mut part_ids[..pivot],
        (axis + 1) % 2,
        bbox_lower,
    );

    // Compute upper
    let bbox_upper = BBox2D::from_points(&points[pivot..]);
    let new_id = part_ids[0] + 2_usize.pow((level - 1) as u32);
    for id in part_ids[pivot..].iter_mut() {
        *id = new_id;
    }
    rcb(
        level - 1,
        &points[pivot..],
        &weights[pivot..],
        &mut part_ids[pivot..],
        (axis + 1) % 2,
        bbox_upper,
    );
}

fn main() {
    let points = vec![
        Point2D::new(-2, -2),
        Point2D::new(-1, -2),
        Point2D::new(1, -2),
        Point2D::new(2, -2),
        Point2D::new(-2, -1),
        Point2D::new(-1, -1),
        Point2D::new(1, -1),
        Point2D::new(2, -1),
        Point2D::new(-2, 1),
        Point2D::new(-1, 1),
        Point2D::new(1, 1),
        Point2D::new(2, 1),
        Point2D::new(-2, 2),
        Point2D::new(-1, 2),
        Point2D::new(1, 2),
        Point2D::new(2, 2),
    ];
    // let weights = vec![1; points.len()];
    let weights = [1, 1, 1, 1, 2, 2, 2, 1, 4, 4, 2, 1, 8, 4, 2, 1];
    assert_eq!(weights.len(), points.len());
    let mut part_ids = vec![0_usize; points.len()];
    let bbox = BBox2D::from_points(&points);

    rcb(2, &points, &weights, &mut part_ids, 0, bbox);
    dbg!(weights);
    dbg!(part_ids);
}
