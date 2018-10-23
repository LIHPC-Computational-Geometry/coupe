use geometry::*;
use rayon;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;
use std::cmp::Ordering;

use nalgebra::Vector2;

/// # Recursive Coordinate Bisection algorithm
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
pub fn rcb(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point2D>) {
    rcb_impl(ids, weights, coordinates, n_iter, true)
}

fn rcb_impl(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
    x_axis: bool, // set if bissection is performed w.r.t. x or y axis
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point2D>) {
    if n_iter == 0 {
        // No iteration left. The current
        // ids become a part of the final partition.
        // Generate a partition id and return.
        let part_id = ProcessUniqueId::new();
        (
            ids.into_par_iter().map(|id| (id, part_id)).collect(),
            weights,
            coordinates,
        )
    } else {
        // We split the objects in two parts of equal weights
        // The split is perfomed alongside the x or y axis,
        // alternating at each iteration.

        // We first need to sort the objects w.r.t. x or y position
        let (mut ids, mut weights, mut coordinates) = axis_sort(ids, weights, coordinates, x_axis);

        // We then seek the split position
        let split_pos = half_weight_pos(weights.as_slice());

        let left_ids = ids.drain(0..split_pos).collect();
        let left_weights = weights.drain(0..split_pos).collect();
        let left_coordinates = coordinates.drain(0..split_pos).collect();

        // Once the split is performed
        // we recursively iterate by calling
        // the algorithm on the two generated parts.
        // In the next iteration, the split aixs will
        // be orthogonal to the current one
        let (left_partition, right_partition) = rayon::join(
            || {
                rcb_impl(
                    left_ids,
                    left_weights,
                    left_coordinates,
                    n_iter - 1,
                    !x_axis,
                )
            },
            || rcb_impl(ids, weights, coordinates, n_iter - 1, !x_axis),
        );

        // We stick the partitions back together
        // to return a single collection of objects
        (
            left_partition
                .0
                .into_par_iter()
                .chain(right_partition.0)
                .collect(),
            left_partition
                .1
                .into_par_iter()
                .chain(right_partition.1)
                .collect(),
            left_partition
                .2
                .into_par_iter()
                .chain(right_partition.2)
                .collect(),
        )
    }
}

// sort input vectors w.r.t. coordinates
// i.e. by increasing x or increasing y
pub fn axis_sort(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    x_axis: bool,
) -> (Vec<usize>, Vec<f64>, Vec<Point2D>) {
    let mut zipped = ids
        .into_par_iter()
        .zip(weights)
        .zip(coordinates)
        .collect::<Vec<_>>();

    zipped
        .as_mut_slice()
        .par_sort_unstable_by(|(_, p1), (_, p2)| {
            if x_axis {
                p1.x.partial_cmp(&p2.x).unwrap_or(Ordering::Equal)
            } else {
                p1.y.partial_cmp(&p2.y).unwrap_or(Ordering::Equal)
            }
        });

    let (still_zipped, coordinates): (Vec<_>, Vec<_>) = zipped.into_par_iter().unzip();
    let (ids, weights): (Vec<_>, Vec<_>) = still_zipped.into_par_iter().unzip();
    (ids, weights, coordinates)
}

// Computes a slice index which splits
// the slice in two parts of equal weights
// i.e. sorted_weights[..idx].sum() == sorted_weights[idx..].sum
fn half_weight_pos(sorted_weights: &[f64]) -> usize {
    let mut half_weight = sorted_weights.par_iter().sum::<f64>() / 2.;
    let mut pos = 0;
    for w in sorted_weights {
        if half_weight > 0. {
            pos += 1;
        } else {
            break;
        }
        half_weight -= w;
    }

    pos
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
pub fn rib(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point2D>) {
    // Compute the inertia vector of the set of points
    let j = inertia_matrix(&weights, &coordinates);
    let inertia = intertia_vector(j);

    // In this implementation, the separator is not actually
    // parallel to the inertia vector. Instead, a global rotation
    // is performed on each point such that the inertia vector is
    // mapped to be parallel to the x axis.

    // such a rotation is given by the following angle
    // alpha = arccos(dot(inertia, x_axis) / (norm(inertia) * norm(x_axis)))
    // In our case, both the inertia and x_axis vector are unit vector.
    let x_unit_vector = Vector2::new(1., 0.);
    let angle = inertia.dot(&x_unit_vector).acos();

    let coordinates = rotate(coordinates, angle);

    // When the rotation is done, we just apply RCB
    rcb(ids, weights, coordinates, n_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_point_sample() -> Vec<Point2D> {
        vec![
            Point2D::new(4., 6.),
            Point2D::new(9., 5.),
            Point2D::new(-1.2, 7.),
            Point2D::new(0., 0.),
            Point2D::new(3., 9.),
            Point2D::new(-4., 3.),
            Point2D::new(1., 2.),
        ]
    }

    #[test]
    fn test_axis_sort_x() {
        let ids: Vec<usize> = (0..7).collect();
        let weights: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
        let points = gen_point_sample();

        let (ids, _, _) = axis_sort(ids, weights, points, true);

        assert_eq!(ids, vec![5, 2, 3, 6, 4, 0, 1]);
    }

    #[test]
    fn test_axis_sort_y() {
        let ids: Vec<usize> = (0..7).collect();
        let weights: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
        let points = gen_point_sample();

        let (ids, _, _) = axis_sort(ids, weights, points, false);

        assert_eq!(ids, vec![3, 6, 5, 1, 0, 2, 4]);
    }
}
