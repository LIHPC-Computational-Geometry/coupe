//! An implementation of the balanced k-means algorithm inspired from
//! "Balanced k-means for Parallel Geometric Partitioning" by Moritz von Looz,
//! Charilaos Tzovas and Henning Meyerhenke (2018, University of Cologne)

use geometry::{self, Mbr2D, Point2D};
use itertools::Itertools;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;

use super::z_curve;

/// A wrapper type for ProcessUniqueId
/// to enforce that it represents temporary ids
/// for the k-means algorithm and not a partition id
type ClusterId = ProcessUniqueId;

const MAX_ITER: usize = 100;

pub fn balanced_k_means(
    points: Vec<Point2D>,
    num_partitions: usize,
    epsilon: f64,
    delta_threshold: f64,
) -> Vec<(Point2D, ProcessUniqueId)> {
    // custom weights are not yet supported
    let weights: Vec<_> = points.iter().map(|_| 1.).collect();

    // sort points with Z-curve
    let qt = z_curve::ZCurveQuadtree::from_points(points);
    let points = qt.reorder();

    let points_per_center = points.len() / num_partitions;

    // select num_partitions initial centers from the ordered points
    let centers: Vec<_> = points
        .iter()
        .cloned()
        .skip(points_per_center / 2)
        .step_by(points_per_center)
        .collect();

    let center_ids: Vec<_> = centers.iter().map(|_| ClusterId::new()).collect();
    let assignments: Vec<_> = center_ids
        .iter()
        .cloned()
        .flat_map(|id| ::std::iter::repeat(id).take(points_per_center))
        .take(points.len())
        .collect();

    let influences: Vec<_> = centers.iter().map(|_| 1.).collect();
    let lbs: Vec<_> = points.iter().map(|_| 0.).collect();
    let ubs: Vec<_> = points.iter().map(|_| 1.).collect();

    balanced_k_means_iter(
        points,
        weights,
        centers,
        &center_ids,
        assignments,
        influences,
        lbs,
        ubs,
        epsilon,
        MAX_ITER,
        delta_threshold,
    )
}

fn balanced_k_means_iter(
    points: Vec<Point2D>,
    weights: Vec<f64>,
    centers: Vec<Point2D>,
    center_ids: &[ClusterId],
    assignments: Vec<ClusterId>,
    influences: Vec<f64>,
    lbs: Vec<f64>,
    ubs: Vec<f64>,
    epsilon: f64,
    current_iter: usize,
    delta_threshold: f64,
) -> Vec<(Point2D, ClusterId)> {
    let (assignments, influences, mut ubs, mut lbs) = assign_and_balance(
        assignments,
        influences,
        lbs,
        ubs,
        &centers,
        &center_ids,
        &points,
        &weights,
        epsilon,
        MAX_ITER,
    );

    let new_centers: Vec<_> = assignments
        .iter()
        .zip(points.iter().cloned())
        .into_group_map()
        .into_iter()
        .map(|(_, points)| geometry::center(&points))
        .collect();

    let distances_moved: Vec<_> = centers
        .into_iter()
        .zip(new_centers.clone())
        .map(|(c1, c2)| (c1 - c2).norm())
        .collect();

    let delta_max = distances_moved
        .iter()
        .max_by(|d1, d2| d1.partial_cmp(d2).unwrap_or(Ordering::Equal))
        .unwrap();

    if *delta_max < delta_threshold || current_iter == 0 {
        points.into_iter().zip(assignments).collect()
    } else {
        relax_bounds(&mut lbs, &mut ubs, &distances_moved, &influences);
        balanced_k_means_iter(
            points,
            weights,
            new_centers,
            center_ids,
            assignments,
            influences,
            lbs,
            ubs,
            epsilon,
            current_iter - 1,
            delta_threshold,
        )
    }
}

fn assign_and_balance(
    mut assignments: Vec<ClusterId>,
    mut influences: Vec<f64>,
    mut lbs: Vec<f64>,
    mut ubs: Vec<f64>,
    centers: &[Point2D],
    center_ids: &[ClusterId],
    points: &[Point2D],
    weights: &[f64],
    epsilon: f64,
    max_iter: usize,
) -> (
    Vec<ClusterId>, // assignments
    Vec<f64>,       // influences
    Vec<f64>,       // ubs
    Vec<f64>,       // lbs
) {
    let mbr = Mbr2D::from_points(points.iter());
    let distances_to_mbr = centers
        .iter()
        .zip(influences.iter())
        .map(|(center, influence)| mbr.distance_to_point(center) / influence)
        .collect::<Vec<_>>();

    let (centers, distances_to_mbr): (Vec<_>, Vec<_>) = centers
        .into_iter()
        .zip(distances_to_mbr)
        .sorted_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(Ordering::Equal))
        .into_iter()
        .unzip();

    let target_weight = weights.iter().sum::<f64>() / weights.iter().count() as f64;

    for _ in 0..max_iter {
        points
            .iter()
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

        // TODO: check imbalance, adapt influence, update lb & ub
        // Compute total weight for each cluster
        let weights_map = assignments
            .iter()
            .cloned()
            .zip(weights.iter())
            .into_group_map();

        let new_weights: Vec<_> = weights_map
            .into_iter()
            .map(|(_, weights)| weights.into_iter().sum::<f64>())
            .collect();

        if imbalance(&new_weights) < epsilon {
            return (assignments, influences, lbs, ubs);
        }

        // If this point is reached, the current assignments
        // are too imbalanced.
        // The influences are then adapted to produce better
        // assignments during next iteration.
        influences
            .iter_mut()
            .zip(new_weights)
            .for_each(|(influence, weight)| {
                let ratio = target_weight / weight;
                let max_diff = 0.05 * *influence;
                let new_influence = *influence / ratio.sqrt();
                if (*influence - new_influence).abs() < max_diff {
                    *influence = new_influence;
                } else if new_influence > *influence {
                    *influence += max_diff;
                } else {
                    *influence -= max_diff;
                }
            });

        // Compute new centers
        let new_centers: Vec<_> = assignments
            .iter()
            .cloned()
            .zip(points.iter().cloned())
            .into_group_map()
            .into_iter()
            .map(|(id, points)| (id, geometry::center(&points)))
            .collect();

        let distances_to_old_centers: Vec<_> = centers
            .iter()
            .zip(new_centers.iter())
            .map(|(center, (_, new_center))| (*center - new_center).norm())
            .collect();

        relax_bounds(&mut lbs, &mut ubs, &distances_to_old_centers, &influences);
    }

    (assignments, influences, lbs, ubs)
}

// relax lower and upper bounds according to influence
// modification.
fn relax_bounds(lbs: &mut [f64], ubs: &mut [f64], distances_moved: &[f64], influences: &[f64]) {
    let max_distance_influence_ratio = distances_moved
        .iter()
        .zip(influences.iter())
        .map(|(distance, influence)| distance / influence)
        .max_by(|r1, r2| r1.partial_cmp(r2).unwrap_or(Ordering::Equal))
        .unwrap_or(0.);

    ubs.iter_mut()
        .zip(distances_moved)
        .zip(influences.iter())
        .for_each(|((ub, distance), influence)| {
            *ub -= distance / influence;
        });

    lbs.iter_mut().for_each(|lb| {
        *lb += max_distance_influence_ratio;
    });
}

fn imbalance(weights: &[f64]) -> f64 {
    use itertools::MinMaxResult::*;
    match weights.iter().minmax() {
        MinMax(min, max) => max - min,
        _ => 0.,
    }
}

/// Most inner loop of the algorithm that aims to optimize
/// clusters assignments
fn best_values(
    point: Point2D,
    centers: &[Point2D],
    center_ids: &[ClusterId],
    distances_to_mbr: &[f64],
    influences: &[f64],
) -> (
    f64,               // new lb
    f64,               // new ub
    Option<ClusterId>, // new cluster assignment for the current point (None if the same assignment is kept)
) {
    let mut best_value = ::std::f64::MAX;
    let mut snd_best_value = ::std::f64::MAX;
    let mut assignment = None;

    for (((center, id), distance_to_mbr), influence) in centers
        .iter()
        .zip(center_ids)
        .zip(distances_to_mbr)
        .zip(influences)
    {
        if *distance_to_mbr > snd_best_value {
            break;
        }
        let effective_distance = (center - point).norm() / influence;
        if effective_distance < best_value {
            assignment = Some(*id);
            snd_best_value = best_value;
            best_value = effective_distance;
        } else if effective_distance < snd_best_value {
            snd_best_value = effective_distance;
        }
    }

    (snd_best_value, best_value, assignment)
}
