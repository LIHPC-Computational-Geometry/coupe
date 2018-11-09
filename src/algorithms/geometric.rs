use geometry::*;
use nalgebra::{DVector, Vector2};
use rayon;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};
use std::sync::Arc;

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
    ids: &[usize],
    weights: &[f64],
    coordinates: &[Point2D],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    let mut permutation = (0..ids.len()).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(ids.len())
        .collect::<Vec<_>>();

    rcb_recurse(
        &ids,
        &weights,
        &coordinates,
        &mut permutation,
        Arc::new(AtomicPtr::new(initial_partition.as_mut_ptr())),
        n_iter,
        true,
    );
    initial_partition
}

fn rcb_recurse(
    ids: &[usize],
    weights: &[f64],
    points: &[Point2D],
    permutation: &mut [usize],
    partition: Arc<AtomicPtr<ProcessUniqueId>>,
    n_iter: usize,
    x_axis: bool, // set if bissection is performed w.r.t. x or y axis
) {
    if n_iter == 0 {
        // No iteration left. The current
        // ids become a part of the final partition.
        // Generate a partition id and return.
        let part_id = ProcessUniqueId::new();
        permutation.par_iter().for_each(|idx| {
            let ptr = partition.load(atomic::Ordering::Relaxed);

            // Unsafe usage explanation:
            //
            // In this implementation, the partition is represented as
            // a contiguous array of ids. It is allocated once, and modified in place.
            // Neither the array nor a slice of it is ever copied. When recursing, a pointer to
            // the array is passed to children functions, which both have mutable access to the
            // partition array from different threads (That's why the pointer is wrapped in a
            // Arc<AtomicPtr<T>>). It is not possible to have shared mutable access to memory
            // across (one/several) threads in safe code. However, the raw pointer is indexed
            // through the permutation array which contains valid indices and is not shared across
            // children calls/threads. This ensures that ptr.add(*idx) is valid memory and every
            // element of the partition array will be written on exactly once.
            unsafe { std::ptr::write(ptr.add(*idx), part_id) }
        });
    } else {
        // We split the objects in two parts of equal weights
        // The split is perfomed alongside the x or y axis,
        // alternating at each iteration.

        // We first need to sort the objects w.r.t. x or y position
        axis_sort_permu(points, permutation, x_axis);

        // We then seek the split position
        let split_pos = half_weight_pos_permu(weights, permutation);
        let (left_permu, right_permu) = permutation.split_at_mut(split_pos);

        // Once the split is performed
        // we recursively iterate by calling
        // the algorithm on the two generated parts.
        // In the next iteration, the split aixs will
        // be orthogonal to the current one
        let left_partition = partition.clone();
        rayon::join(
            || {
                rcb_recurse(
                    ids,
                    weights,
                    points,
                    left_permu,
                    left_partition,
                    n_iter - 1,
                    !x_axis,
                )
            },
            || {
                rcb_recurse(
                    ids,
                    weights,
                    points,
                    right_permu,
                    partition,
                    n_iter - 1,
                    !x_axis,
                )
            },
        );
    }
}

pub fn axis_sort_permu(points: &[Point2D], permutation: &mut [usize], x_axis: bool) {
    if x_axis {
        permutation.par_sort_by(|i1, i2| points[*i1].x.partial_cmp(&points[*i2].x).unwrap());
    } else {
        permutation.par_sort_by(|i1, i2| points[*i1].y.partial_cmp(&points[*i2].y).unwrap());
    }
}

// sort input vectors w.r.t. coordinates
// i.e. by increasing x or increasing y
pub fn axis_sort(
    ids: &[usize],
    weights: &[f64],
    coordinates: &[Point2D],
    x_axis: bool,
) -> (Vec<usize>, Vec<f64>, Vec<Point2D>) {
    let mut zipped = ids
        .par_iter()
        .cloned()
        .zip(weights.par_iter().cloned())
        .zip(coordinates.par_iter().cloned())
        .collect::<Vec<_>>();

    if x_axis {
        zipped
            .as_mut_slice()
            .par_sort_by(|(_, p1), (_, p2)| p1.x.partial_cmp(&p2.x).unwrap_or(Ordering::Equal));
    } else {
        zipped
            .as_mut_slice()
            .par_sort_by(|(_, p1), (_, p2)| p1.y.partial_cmp(&p2.y).unwrap_or(Ordering::Equal));
    }

    let (still_zipped, coordinates): (Vec<_>, Vec<_>) = zipped.into_par_iter().unzip();
    let (ids, weights): (Vec<_>, Vec<_>) = still_zipped.into_par_iter().unzip();
    (ids, weights, coordinates)
}

/// Sort `ids`, `weights`, `points` simultaneously
/// by increasing x' where x' is the first coordinate
/// of an orthonormal basis (direction, e2, ..., ek) of R^k
pub fn axis_sort_nd(
    ids: Vec<usize>,
    weights: Vec<f64>,
    points: Vec<Point>,
    direction: &DVector<f64>,
) -> (Vec<usize>, Vec<f64>, Vec<Point>) {
    let base_shift = householder_reflection(&direction);
    let new_points = points.iter().map(|p| &base_shift * p).collect::<Vec<_>>();

    let mut zipped = ids
        .into_par_iter()
        .zip(weights)
        .zip(points)
        .enumerate()
        .collect::<Vec<_>>();

    zipped
        .as_mut_slice()
        .par_sort_unstable_by(|(i, _), (j, _)| {
            new_points[*i][0]
                .partial_cmp(&new_points[*j][0])
                .unwrap_or(Ordering::Equal)
        });

    let (_is, still_zipped): (Vec<_>, Vec<_>) = zipped.into_par_iter().unzip();
    let (still_zipped, points): (Vec<_>, Vec<_>) = still_zipped.into_par_iter().unzip();
    let (ids, weights): (Vec<_>, Vec<_>) = still_zipped.into_par_iter().unzip();
    (ids, weights, points)
}

/// # N-Dimensional Recursive Coordinate Bisection algorithm
/// Partitions a mesh based on the nodes coordinates and coresponding weights.
/// ## Inputs
/// - `ids`: global identifiers of the objects to partition
/// - `weights`: weights corsponding to a cost relative to the objects
/// - `coordinates`: the ND coordinates of the objects to partition
///
/// ## Output
/// A Vec of couples `(usize, ProcessUniqueId)`
///
/// the first component of each couple is the id of an object and
/// the second component is the id of the partition to which that object was assigned
pub fn rcb_nd(
    ids: Vec<usize>,
    weights: Vec<f64>,
    points: Vec<Point>,
    n_iter: usize,
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point>) {
    let dim = points[0].len();
    rcb_nd_recurse(ids, weights, points, n_iter, dim, 0)
}

fn rcb_nd_recurse(
    ids: Vec<usize>,
    weights: Vec<f64>,
    points: Vec<Point>,
    n_iter: usize,
    dim: usize,
    current_coord: u32,
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point>) {
    if n_iter == 0 {
        let part_id = ProcessUniqueId::new();
        (
            ids.into_par_iter().map(|id| (id, part_id)).collect(),
            weights,
            points,
        )
    } else {
        let normal = canonical_vector(dim, current_coord as usize);
        let (mut ids, mut weights, mut points) = axis_sort_nd(ids, weights, points, &normal);
        let split_pos = half_weight_pos(&weights);

        let left_ids = ids.drain(0..split_pos).collect();
        let left_weights = weights.drain(0..split_pos).collect();
        let left_points = points.drain(0..split_pos).collect();

        let (left_partition, right_partition) = rayon::join(
            || {
                rcb_nd_recurse(
                    left_ids,
                    left_weights,
                    left_points,
                    n_iter - 1,
                    dim,
                    (current_coord + 1) % dim as u32,
                )
            },
            || {
                rcb_nd_recurse(
                    ids,
                    weights,
                    points,
                    n_iter - 1,
                    dim,
                    (current_coord + 1) % dim as u32,
                )
            },
        );

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

// Computes a slice index which splits
// the slice in two parts of equal weights
// i.e. sorted_weights[..idx].sum() == sorted_weights[idx..].sum
fn half_weight_pos_permu(weights: &[f64], permutation: &[usize]) -> usize {
    let half_weight = permutation.par_iter().map(|idx| weights[*idx]).sum::<f64>() / 2.;

    let mut current_weight_idx;
    let mut current_weight_sum = 0.;

    let mut scan = permutation
        .par_iter()
        .enumerate()
        .fold_with((std::usize::MAX, 0.), |(low, acc), (idx, val)| {
            if idx < low {
                (idx, acc + weights[*val])
            } else {
                (low, acc + weights[*val])
            }
        }).collect::<Vec<_>>()
        .into_iter();

    // above this, the code was parallel
    // what follows is sequential

    loop {
        let current = scan.next().unwrap();
        if current_weight_sum + current.1 > half_weight {
            current_weight_idx = current.0;
            break;
        }

        current_weight_sum += current.1;
    }

    // seek from current_weight_idx
    while current_weight_sum < half_weight {
        current_weight_idx += 1;
        current_weight_sum += weights[permutation[current_weight_idx]];
    }

    current_weight_idx
}

fn half_weight_pos(sorted_weights: &[f64]) -> usize {
    let half_weight = sorted_weights.par_iter().sum::<f64>() / 2.;

    let mut current_weight_idx;
    let mut current_weight_sum = 0.;

    let mut scan = sorted_weights
        .par_iter()
        .enumerate()
        .fold_with((std::usize::MAX, 0.), |(low, acc), (idx, val)| {
            if idx < low {
                (idx, acc + val)
            } else {
                (low, acc + val)
            }
        }).collect::<Vec<_>>()
        .into_iter();

    // above this, the code was parallel
    // what follows is sequential

    loop {
        let current = scan.next().unwrap();
        if current_weight_sum + current.1 > half_weight {
            current_weight_idx = current.0;
            break;
        }

        current_weight_sum += current.1;
    }

    // seek from current_weight_idx
    while current_weight_sum < half_weight {
        current_weight_idx += 1;
        current_weight_sum += sorted_weights[current_weight_idx];
    }

    current_weight_idx
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
    ids: &[usize],
    weights: &[f64],
    coordinates: &[Point2D],
    n_iter: usize,
) -> Vec<ProcessUniqueId> {
    // Compute the inertia vector of the set of points
    let j = inertia_matrix(weights, coordinates);
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

    let coordinates = rotate(coordinates.to_vec(), angle);

    // When the rotation is done, we just apply RCB
    rcb(ids, weights, &coordinates, n_iter)
}

pub fn rib_nd(
    ids: Vec<usize>,
    weights: Vec<f64>,
    points: Vec<Point>,
    n_iter: usize,
) -> (Vec<(usize, ProcessUniqueId)>, Vec<f64>, Vec<Point>) {
    let j = inertia_matrix_nd(&weights, &points);
    let inertia = intertia_vector_nd(j);

    let base_shift = householder_reflection(&inertia);
    let points = points
        .into_par_iter()
        .map(|p| &base_shift * p)
        .collect::<Vec<_>>();

    rcb_nd(ids, weights, points, n_iter)
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

        let (ids, _, _) = axis_sort(&ids, &weights, &points, true);

        assert_eq!(ids, vec![5, 2, 3, 6, 4, 0, 1]);
    }

    #[test]
    fn test_axis_sort_y() {
        let ids: Vec<usize> = (0..7).collect();
        let weights: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
        let points = gen_point_sample();

        let (ids, _, _) = axis_sort(&ids, &weights, &points, false);

        assert_eq!(ids, vec![3, 6, 5, 1, 0, 2, 4]);
    }

    #[test]
    fn test_hyperplane_sort() {
        let ids: Vec<usize> = (0..3).collect();
        let weights: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
        let points = vec![
            Point::from_row_slice(3, &[0., 0., 0.]),
            Point::from_row_slice(3, &[-1., -1., 1.]),
            Point::from_row_slice(3, &[-2., -2., 2.]),
        ];

        let (ids, _, _points) = axis_sort_nd(
            ids,
            weights,
            points,
            &Point::from_row_slice(3, &[1., 1., -1.]),
        );

        assert_eq!(ids, vec![2, 1, 0]);
    }
}
