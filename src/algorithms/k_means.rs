//! An implementation of the balanced k-means algorithm inspired from
//! "Balanced k-means for Parallel Geometric Partitioning" by Moritz von Looz,
//! Charilaos Tzovas and Henning Meyerhenke (2018, University of Cologne)

use geometry::{self, Point2D};
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;

use super::z_curve;

use geometry::Mbr2D;
use itertools::iproduct;
use itertools::Itertools;

/// A wrapper type for ProcessUniqueId
/// to enforce that it represents temporary ids
/// for the k-means algorithm and not a partition id
type ClusterId = ProcessUniqueId;

/// A simplified implementation of the algorithm described in the paper
/// by Moritz von Looz et al. that follows the same idea but without the small
/// optimizations that would improve the efficiency of the algorithm. In particular,
/// this version shows some noticeable oscillations when imposing a restrictive balance constraint.
/// It also skips the bounding boxes optimization which would slightly reduce the complexity of the
/// algorithm.
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

/// Settings to tune the balanced k-means algorithm
///
/// ## Attributes
///   - `num_partitions`: the exact number of partitions the algorithm is guarenteed to yield.
///   - `imbalance_tol`: the relative imbalance tolerance of the generated partitions, in `%` of the target weight of each partition.
///   - `delta_threshold`: the distance threshold for the cluster movements under which the algorithm stops.
///   - `max_iter`: the maximum number of times each cluster will move before stopping the algorithm
///   - `max_balance_iter`: the maximum number of iterations of the load balancing loop. It will limit how much each cluster
///      influence can grow between each cluster movement.
///   - `erode`: sets whether or not cluster influence is modified according to errosion's rules between each cluster movement
///   - `mbr_early_break`: sets whether or not bounding box optimization is enabled.
#[derive(Debug, Clone, Copy)]
pub struct BalancedKmeansSettings {
    pub num_partitions: usize,
    pub imbalance_tol: f64,
    pub delta_threshold: f64,
    pub max_iter: usize,
    pub max_balance_iter: usize,
    pub erode: bool,
    pub mbr_early_break: bool,
}

impl Default for BalancedKmeansSettings {
    fn default() -> Self {
        Self {
            num_partitions: 7,
            imbalance_tol: 5.,
            delta_threshold: 0.01,
            max_iter: 50,
            max_balance_iter: 1, // for now, `max_balance_iter > 1` yields poor convergence time
            erode: false,        // for now, `erode` yields` enabled yields wrong results
            mbr_early_break: false, // for now, `mbr_early_break` enabled yields wrong results
        }
    }
}

pub fn balanced_k_means(
    points: Vec<Point2D>,
    settings: impl Into<Option<BalancedKmeansSettings>>,
) -> Vec<(Point2D, ProcessUniqueId)> {
    let settings = settings.into().unwrap_or_default();

    // custom weights are not yet supported
    let weights: Vec<_> = points.iter().map(|_| 1.).collect();

    // sort points with Z-curve
    let qt = z_curve::ZCurveQuadtree::new(points, weights);
    let (points, weights) = qt.reorder();

    // Compute how many points will be initially assigned to each cluster
    let points_per_center = points.len() / settings.num_partitions;

    // select num_partitions initial centers from the ordered points
    let centers: Vec<_> = points
        .iter()
        .cloned()
        // for each partition yielded by the Z-curve reordering
        // we select the median point to be the initial cluster center
        // because it is in most cases in the middle of the partition
        .skip(points_per_center / 2)
        .step_by(points_per_center)
        .collect();

    // generate unique ids for each initial partition that will live throughout
    // the algorithm (no new id is generated afterwards)
    let center_ids: Vec<_> = centers.iter().map(|_| ClusterId::new()).collect();

    // generate initial assignments, i.e.
    // map [id0, id1, ..., idn]
    //    to [[id0, ..., id0], ..., [idn, ..., idn]]
    //         partition_1             partition_n
    let assignments: Vec<_> = center_ids
        .iter()
        .cloned()
        .flat_map(|id| std::iter::repeat(id).take(points_per_center))
        .take(points.len())
        .collect();

    // Generate initial influences (to 1)
    let influences: Vec<_> = centers.iter().map(|_| 1.).collect();

    // Generate initial lower and upper bounds. These two variables represent bounds on
    // the effective distance between an point and the cluster it is assigned to.
    let lbs: Vec<_> = points.iter().map(|_| 0.).collect();
    let ubs: Vec<_> = points.iter().map(|_| std::f64::MAX).collect(); // we use f64::MAX to represent infinity

    balanced_k_means_iter(
        Inputs { points, weights },
        Clusters {
            centers,
            center_ids: &center_ids,
        },
        AlgorithmState {
            assignments,
            influences,
            lbs,
            ubs,
        },
        &settings,
        settings.max_iter,
    )
}

struct Inputs {
    points: Vec<Point2D>,
    weights: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Clusters<T, U> {
    centers: T,
    center_ids: U,
}

struct AlgorithmState {
    assignments: Vec<ClusterId>,
    influences: Vec<f64>,
    lbs: Vec<f64>,
    ubs: Vec<f64>,
}

// This is the main loop of the algorithm. It handles:
//  - calling the load balance routine
//  - moving each cluster after load balance
//  - checking delta threshold
//  - relaxing lower and upper bounds
fn balanced_k_means_iter(
    inputs: Inputs,
    clusters: Clusters<Vec<Point2D>, &[ClusterId]>,
    state: AlgorithmState,
    settings: &BalancedKmeansSettings,
    current_iter: usize,
) -> Vec<(Point2D, ClusterId)> {
    let Inputs { points, weights } = inputs;
    let Clusters {
        centers,
        center_ids,
    } = clusters;
    let AlgorithmState {
        assignments,
        influences,
        lbs,
        ubs,
    } = state;

    let (assignments, influences, mut ubs, mut lbs) = assign_and_balance(
        AlgorithmState {
            assignments,
            influences,
            lbs,
            ubs,
        },
        Clusters {
            centers: &centers,
            center_ids: &center_ids,
        },
        &points,
        &weights,
        settings,
    );

    // Compute new centers from the load balance routine assignments output
    let new_centers = center_ids
        .iter()
        // map each center id to the new center point
        // we cannot just compute the centers fron the assignments
        // because the new centers have to be in the same order as the old ones
        .map(|center_id| {
            let points = assignments
                .iter()
                .cloned()
                .zip(points.iter().cloned())
                .filter(|(assignment, _)| *assignment == *center_id)
                .map(|(_, point)| point)
                .collect::<Vec<_>>();
            geometry::center(&points)
        }).collect::<Vec<_>>();

    // Compute the distances moved by each center from their previous location
    let distances_moved: Vec<_> = centers
        .into_iter()
        .zip(new_centers.clone())
        .map(|(c1, c2)| (c1 - c2).norm())
        .collect();

    let delta_max = distances_moved
        .iter()
        .max_by(|d1, d2| d1.partial_cmp(d2).unwrap_or(Ordering::Equal))
        .unwrap();

    // if delta_max is below a given threshold, it means that the clusters no longer move a lot at each iteration
    // and the algorithm has become somewhat stable.
    if *delta_max < settings.delta_threshold || current_iter == 0 {
        points.into_iter().zip(assignments).collect()
    } else {
        relax_bounds(&mut lbs, &mut ubs, &distances_moved, &influences);
        balanced_k_means_iter(
            Inputs { points, weights },
            Clusters {
                centers: new_centers,
                center_ids,
            },
            AlgorithmState {
                assignments,
                influences,
                lbs,
                ubs,
            },
            settings,
            current_iter - 1,
        )
    }
}

// This is the main load balance routine. It handles:
//   - reordering the clusters according to their distance to a bounding box of all the points
//   - assigning each point to the closest cluster according to the effective distance
//   - checking partitions imbalance
//   - increasing of diminishing clusters influence based on their imbalance
//   - relaxing upper and lower bounds
fn assign_and_balance(
    state: AlgorithmState,
    clusters: Clusters<&[Point2D], &[ClusterId]>,
    points: &[Point2D],
    weights: &[f64],
    settings: &BalancedKmeansSettings,
) -> (
    Vec<ClusterId>, // assignments
    Vec<f64>,       // influences
    Vec<f64>,       // ubs
    Vec<f64>,       // lbs
) {
    let AlgorithmState {
        mut assignments,
        mut influences,
        mut lbs,
        mut ubs,
    } = state;
    let Clusters {
        centers,
        center_ids,
    } = clusters;
    // compute the distances from each cluster center to the minimal
    // bounding rectangle of the set of points
    let mbr = Mbr2D::from_points(points.iter());
    let distances_to_mbr = centers
        .iter()
        .zip(influences.iter())
        .map(|(center, influence)| mbr.distance_to_point(center) * influence)
        .collect::<Vec<_>>();

    let (zipped, distances_to_mbr): (Vec<_>, Vec<_>) = centers
        .into_iter()
        .zip(center_ids)
        .zip(distances_to_mbr)
        .sorted_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(Ordering::Equal))
        .into_iter()
        .unzip();

    let (centers, center_ids): (Vec<_>, Vec<_>) = zipped.into_iter().unzip();

    // Compute the weight that each cluster should be after the end of the algorithm
    let target_weight = weights.iter().sum::<f64>() / (centers.len() as f64);

    for _ in 0..settings.max_balance_iter {
        // Compute new assignments point to cluster assignments
        // based on the current clusters and influences state
        points
            .iter()
            .zip(assignments.iter_mut())
            .zip(lbs.iter_mut())
            .zip(ubs.iter_mut())
            .for_each(|(((p, id), lb), ub)| {
                if lb < ub {
                    let (new_lb, new_ub, new_assignment) = best_values(
                        *p,
                        &centers,
                        &center_ids,
                        &distances_to_mbr,
                        &influences,
                        settings,
                    );

                    *lb = new_lb;
                    *ub = new_ub;
                    if let Some(new_assignment) = new_assignment {
                        *id = new_assignment;
                    }
                }
            });

        // Compute total weight for each cluster
        let new_weights = center_ids
            .iter()
            .map(|center_id| {
                assignments
                    .iter()
                    .cloned()
                    .zip(weights.iter())
                    .filter(|(assignment, _)| *assignment == *center_id)
                    .map(|(_, weight)| *weight)
                    .sum::<f64>()
            }).collect::<Vec<_>>();

        // return if maximum imbalance is small enough
        if imbalance(&new_weights) < settings.imbalance_tol {
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
                // We limit the influence variation to 5% each time
                // to preven the algorithm from becoming unstable
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

        // Compute new centers from new assigments
        let new_centers = center_ids
            .iter()
            .map(|center_id| {
                let points = assignments
                    .iter()
                    .cloned()
                    .zip(points.iter().cloned())
                    .filter(|(assignment, _)| *assignment == *center_id)
                    .map(|(_, point)| point)
                    .collect::<Vec<_>>();
                geometry::center(&points)
            }).collect::<Vec<_>>();

        let distances_to_old_centers: Vec<_> = centers
            .iter()
            .zip(new_centers.iter())
            .map(|(center, new_center)| (*center - new_center).norm())
            .collect();

        relax_bounds(&mut lbs, &mut ubs, &distances_to_old_centers, &influences);

        if settings.erode {
            let average_diameters = assignments
                .iter()
                .zip(points.iter().cloned())
                .into_group_map()
                .into_iter()
                .map(|(_assignment, points)| max_distance(&points))
                .sum::<f64>()
                / centers.len() as f64;

            // erode influence
            influences
                .iter_mut()
                .zip(distances_to_old_centers.iter())
                .for_each(|(influence, distance)| {
                    *influence =
                        influence.log(10.) * (1. - erosion(*distance, average_diameters)).exp()
                });
        }
    }

    (assignments, influences, lbs, ubs)
}

// relax lower and upper bounds according to influence
// modification.
//
// new_lb(p) = lb(p) - max_{c'} delta(c') / influence(c')
// new_ub(p) = ub(p) + delta(c) / influence(c)
fn relax_bounds(lbs: &mut [f64], ubs: &mut [f64], distances_moved: &[f64], influences: &[f64]) {
    let max_distance_influence_ratio = distances_moved
        .iter()
        .zip(influences.iter())
        .map(|(distance, influence)| distance * influence)
        .max_by(|r1, r2| r1.partial_cmp(r2).unwrap_or(Ordering::Equal))
        .unwrap_or(0.);

    ubs.iter_mut()
        .zip(distances_moved)
        .zip(influences.iter())
        .for_each(|((ub, distance), influence)| {
            *ub += distance * influence;
        });

    lbs.iter_mut().for_each(|lb| {
        *lb -= max_distance_influence_ratio;
    });
}

/// Most inner loop of the algorithm that aims to optimize
/// clusters assignments
fn best_values(
    point: Point2D,
    centers: &[Point2D],
    center_ids: &[ClusterId],
    distances_to_mbr: &[f64],
    influences: &[f64],
    settings: &BalancedKmeansSettings,
) -> (
    f64,               // new lb
    f64,               // new ub
    Option<ClusterId>, // new cluster assignment for the current point (None if the same assignment is kept)
) {
    let mut best_value = std::f64::MAX;
    let mut snd_best_value = std::f64::MAX;
    let mut assignment = None;

    for (((center, id), distance_to_mbr), influence) in centers
        .iter()
        .zip(center_ids)
        .zip(distances_to_mbr)
        .zip(influences)
    {
        if *distance_to_mbr > snd_best_value && settings.mbr_early_break {
            break;
        }

        let effective_distance = (center - point).norm() * influence;
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

// erosion(c) = 2 / (1 + exp(min(-delta(c)/beta(C), 0))) - 1
// where beta(C) is the average cluster diameter
fn erosion(distance_moved: f64, average_cluster_diameter: f64) -> f64 {
    2. / (1. + (-distance_moved / average_cluster_diameter).min(0.).exp()) - 1.
}

// computes the maximum distance between two points in the array
fn max_distance(points: &[Point2D]) -> f64 {
    iproduct!(points, points)
        .map(|(p1, p2)| (p1 - p2).norm())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
}
