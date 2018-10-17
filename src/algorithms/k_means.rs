//! An implementation of the balanced k-means algorithm inspired from
//! "Balanced k-means for Parallel Geometric Partitioning" by Moritz von Looz,
//! Charilaos Tzovas and Henning Meyerhenke (2018, University of Cologne)

use geometry::{Mbr2D, Point2D};
use itertools::Itertools;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;

pub fn balanced_k_means(
    points: Vec<Point2D>,
    num_partitions: usize,
    epsilon: f64,
    maximum_imbalance: f64,
    deltha_threshold: f64,
) -> Vec<(Point2D, ProcessUniqueId)> {
    unimplemented!()
}

pub fn assign_and_balance(
    centers: Vec<Point2D>,
    local_points: Vec<Point2D>,
    weights: Vec<f64>,
    influences: Vec<f64>,
    previous_assignments: Vec<(Point2D, ProcessUniqueId)>,
    ubs: Vec<f64>,
    lbs: Vec<f64>,
    epsilon: f64,
    max_iter: usize,
) -> (
    Vec<(Point2D, ProcessUniqueId)>, // assignments
    Vec<f64>,                        // influences
    f64,                             // ub
    f64,                             // lb
) {
    let mbr = Mbr2D::from_points(local_points.iter());
    let distances_to_mbr = centers
        .iter()
        .zip(influences.iter())
        .map(|(center, influence)| mbr.distance_to_point(center) / influence)
        .collect::<Vec<_>>();

    let local_block_sizes: Vec<usize> = centers.iter().map(|_| 0).collect();

    let centers = centers
        .iter()
        .zip(distances_to_mbr)
        .sorted_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(Ordering::Equal));

    let mut zipped = izip!(
        local_points,
        weights,
        previous_assignments,
        influences,
        ubs,
        lbs
    );

    for _ in 0..max_iter {
        izip!(
            local_points,
            weights,
            previous_assignments,
            influences,
            ubs,
            lbs
        ).map(|(local_point, weight, previous_assignment, influence, ub, lb)| {});
    }

    unimplemented!();
}
