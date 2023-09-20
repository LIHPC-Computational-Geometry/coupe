mod bounding_box;
mod grid;
mod point_nd;

use crate::bounding_box::BBox2D;
use crate::grid::Grid;
use crate::point_nd::Point2D;

fn main() {
    let points = vec![
        Point2D::new(-8, -2),
        Point2D::new(-9, 1),
        Point2D::new(0, -8),
        Point2D::new(-9, 5),
        Point2D::new(-2, -6),
        Point2D::new(3, -2),
        Point2D::new(6, 10),
        Point2D::new(0, 1),
        Point2D::new(-8, 3),
        Point2D::new(0, 7),
        Point2D::new(0, 9),
        Point2D::new(1, 5),
        Point2D::new(10, 0),
        Point2D::new(-8, 7),
        Point2D::new(0, -7),
        Point2D::new(5, -4),
    ];
    let weights = [1, 1, 1, 1, 2, 2, 2, 1, 4, 4, 2, 1, 8, 4, 2, 1];
    assert_eq!(weights.len(), points.len());

    let bbox = BBox2D::from_points(&points);
    let grid = Grid::<2, i32>::from_points(bbox.unwrap(), 3, 2, (&points, &weights));
    let (new_points, _new_weights) = grid.reordered_points_and_weights((&points, &weights));
    let mut part_ids = vec![0_usize; points.len()];

    grid.rcb(2, &new_points, &mut part_ids, 0, bbox.unwrap());
    println!("Point(x, y):\tWeight\tPartition");
    for (p, w, i) in itertools::multizip((points, weights, part_ids)) {
        println!("{:+?}:\t{w}\t{i}", (p.x(), p.y()));
    }
}
