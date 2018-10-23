//! An implementation of the balanced k-means algorithm inspired from
//! "Balanced k-means for Parallel Geometric Partitioning" by Moritz von Looz,
//! Charilaos Tzovas and Henning Meyerhenke (2018, University of Cologne)

use geometry::{self, Mbr2D, Point2D};
use itertools::Itertools;
use rayon;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;

use super::z_curve;

/// A wrapper type for ProcessUniqueId
/// to enforce that it represents temporary ids
/// for the k-means algorithm and not a partition id
type ClusterId = ProcessUniqueId;

pub fn simplified_k_means(
    points: Vec<Point2D>,
    weights: Vec<f64>,
    num_partitions: usize,
    imbalance_tol: f64,
    mut n_iter: isize,
) -> (Vec<(Point2D, ProcessUniqueId)>, Vec<f64>) {
    let qt = z_curve::ZCurveQuadtree::new(points, weights);
    let (points, weights) = qt.reorder();

    let points_per_center = points.len() / num_partitions;

    let mut centers: Vec<_> = points
        .iter()
        .cloned()
        .skip(points_per_center / 2)
        .step_by(points_per_center)
        .collect();

    let center_ids: Vec<_> = centers.par_iter().map(|_| ClusterId::new()).collect();

    let mut influences = centers.par_iter().map(|_| 1.).collect::<Vec<_>>();

    let mut assignments: Vec<_> = center_ids
        .iter()
        .cloned()
        .flat_map(|id| ::std::iter::repeat(id).take(points_per_center))
        .take(points.len())
        .collect();

    let mut imbalance = ::std::f64::MAX;

    let target_weight = weights.par_iter().sum::<f64>() / num_partitions as f64;

    while imbalance > imbalance_tol && n_iter > 0 {
        n_iter -= 1;

        // find new assignments
        points
            .par_iter()
            .zip(assignments.par_iter_mut())
            .for_each(|(point, assignment)| {
                // find closest center
                let mut distances = ::std::iter::repeat(*point)
                    .zip(centers.iter())
                    .zip(center_ids.iter())
                    .zip(influences.iter())
                    .map(|(((p, center), id), influence)| (*id, (p - center).norm() * influence))
                    .collect::<Vec<_>>();

                distances
                    .as_mut_slice()
                    .sort_unstable_by(|(_, d1), (_, d2)| {
                        d1.partial_cmp(d2).unwrap_or(Ordering::Equal)
                    });

                // update assignment with closest center found
                *assignment = distances.into_iter().next().unwrap().0;
            });

        // update centers position from new assignments
        centers
            .par_iter_mut()
            .zip(center_ids.par_iter())
            .for_each(|(center, id)| {
                let new_center = geometry::center(
                    &points
                        .iter()
                        .zip(assignments.iter())
                        .filter(|(_, point_id)| *id == **point_id)
                        .map(|(p, _)| *p)
                        .collect::<Vec<_>>(),
                );

                *center = new_center;
            });

        // compute cluster weights
        let cluster_weights = center_ids
            .par_iter()
            .map(|id| {
                assignments
                    .iter()
                    .zip(weights.iter())
                    .filter(|(point_id, _)| *id == **point_id)
                    .fold(0., |acc, (_, weight)| acc + weight)
            }).collect::<Vec<_>>();

        // update influence
        cluster_weights
            .par_iter()
            .zip(influences.par_iter_mut())
            .for_each(|(cluster_weight, influence)| {
                let ratio = target_weight / *cluster_weight as f64;

                let new_influence = *influence / ratio.sqrt();
                let max_diff = 0.05 * *influence;
                if (*influence - new_influence).abs() < max_diff {
                    *influence = new_influence;
                } else if new_influence > *influence {
                    *influence += max_diff;
                } else {
                    *influence -= max_diff;
                }
            });

        // update imbalance
        imbalance = self::imbalance(&cluster_weights);
    }

    (points.into_iter().zip(assignments).collect(), weights)
}

fn imbalance(weights: &[f64]) -> f64 {
    match (
        weights
            .par_iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)),
        weights
            .par_iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)),
    ) {
        (Some(min), Some(max)) => max - min,
        _ => 0.,
    }
}
