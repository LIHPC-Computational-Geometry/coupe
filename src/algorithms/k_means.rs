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
    mut centers: Vec<Point2D>,
    center_ids: Vec<ProcessUniqueId>,
    mut local_points: Vec<Point2D>,
    mut weights: Vec<f64>,
    mut influences: Vec<f64>,
    mut assignments: Vec<ProcessUniqueId>,
    mut ubs: Vec<f64>,
    mut lbs: Vec<f64>,
    mut epsilon: f64,
    mut max_iter: usize,
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

    let (centers, distances_to_mbr): (Vec<_>, Vec<_>) = centers
        .into_iter()
        .zip(distances_to_mbr)
        .sorted_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(Ordering::Equal))
        .into_iter()
        .unzip();

    for _ in 0..max_iter {
        local_points
            .iter_mut()
            .zip(assignments.iter_mut())
            .zip(lbs.iter_mut())
            .zip(ubs.iter_mut())
            .for_each(|(((p, id), lb), ub)| {
                if lb < ub {
                    let (new_lb, new_ub, new_assignment) =
                        best_values(*p, &centers, &center_ids, &distances_to_mbr, &influences);

                    *lb = new_lb;
                    *ub = new_ub;
                    if let Some(new_assignment) = new_assignment {
                        *id = new_assignment;
                    }
                }
            });
    }

    // TODO: check imbalance, adapt influence, update lb & ub
    unimplemented!();
}

/// Most inner loop of the algorithm that aims to optimize
/// clusters assignments
pub fn best_values(
    point: Point2D,
    centers: &[Point2D],
    center_ids: &[ProcessUniqueId],
    distances_to_mbr: &[f64],
    influences: &[f64],
) -> (
    f64,                     // new lb
    f64,                     // new ub
    Option<ProcessUniqueId>, // new assignment for the current point (None if the same assignment is kept)
) {
    use itertools::FoldWhile::{Continue, Done};

    let (lb, ub, a) = centers
        .iter()
        .zip(center_ids)
        .zip(distances_to_mbr)
        .zip(influences)
        // compute for each cluster, the effective distance
        // between the current point and the cluster, defined by
        // effective_distance = distance(cluster, point) / influence(cluster)
        .map(|(((center, id), distance_to_mbr), influence)| {
            (
                (center, id),
                distance_to_mbr,
                (center - point).norm() / influence,
            )
        })
        // lookup through every cluster to find new best bounds for the current point
        // and keep track of a new assignment that is better than the current one
        .fold_while(
            // lower and upper bounds are initially None to
            // represent that they are uninitialized
            (None, None, None),
            |(lb, ub, a), ((_center, id), distance_to_mbr, effective_distance)| match (lb, ub) {
                (Some(lb), _) if *distance_to_mbr > lb => Done((Some(lb), ub, a)),
                (Some(lb), ub) if effective_distance < lb => {
                    Continue((Some(effective_distance), ub, a))
                }
                (None, ub) => Continue((ub, Some(effective_distance), Some(*id))),
                (_, Some(ub)) if effective_distance < ub => {
                    Continue((Some(ub), Some(effective_distance), Some(*id)))
                }
                _ => Continue((lb, ub, a)),
            },
        ).into_inner();

    (lb.unwrap(), ub.unwrap(), a)
}
