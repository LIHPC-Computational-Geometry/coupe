use geometry::*;
use nalgebra::{DVector, Vector2};
use rayon;
use rayon::prelude::*;
use snowflake::ProcessUniqueId;

use std::cmp::Ordering;
use std::sync::atomic::{self, AtomicPtr};

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
pub fn rcb_2d(points: &[Point2D], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    let len = weights.len();
    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(len)
        .collect::<Vec<_>>();

    rcb_2d_recurse(
        &points,
        &weights,
        &mut permutation,
        &AtomicPtr::new(initial_partition.as_mut_ptr()),
        n_iter,
        true,
    );
    initial_partition
}

fn rcb_2d_recurse(
    points: &[Point2D],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
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
        axis_sort_2d(points, permutation, x_axis);

        // We then seek the split position
        let split_pos = half_weight_pos_permu(weights, permutation);
        let (left_permu, right_permu) = permutation.split_at_mut(split_pos);

        // Once the split is performed
        // we recursively iterate by calling
        // the algorithm on the two generated parts.
        // In the next iteration, the split aixs will
        // be orthogonal to the current one

        rayon::join(
            || rcb_2d_recurse(points, weights, left_permu, partition, n_iter - 1, !x_axis),
            || rcb_2d_recurse(points, weights, right_permu, partition, n_iter - 1, !x_axis),
        );
    }
}

pub fn axis_sort_2d(points: &[Point2D], permutation: &mut [usize], x_axis: bool) {
    if x_axis {
        permutation.par_sort_by(|i1, i2| fast_f64_cmp(points[*i1].x, points[*i2].x));
    } else {
        permutation.par_sort_by(|i1, i2| fast_f64_cmp(points[*i1].y, points[*i2].y));
    }
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn next(self) -> Self {
        use self::Axis::*;
        match self {
            X => Y,
            Y => Z,
            Z => X,
        }
    }
}

// Implementation is a copy/paste from rcb_2d
pub fn rcb_3d(points: &[Point3D], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    let len = weights.len();
    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(len)
        .collect::<Vec<_>>();

    rcb_3d_recurse(
        points,
        weights,
        &mut permutation,
        &AtomicPtr::new(initial_partition.as_mut_ptr()),
        n_iter,
        Axis::X,
    );
    initial_partition
}

fn rcb_3d_recurse(
    points: &[Point3D],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
    n_iter: usize,
    axis: Axis,
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
        axis_sort_3d(points, permutation, axis);

        // We then seek the split position
        let split_pos = half_weight_pos_permu(weights, permutation);
        let (left_permu, right_permu) = permutation.split_at_mut(split_pos);

        // Once the split is performed
        // we recursively iterate by calling
        // the algorithm on the two generated parts.
        // In the next iteration, the split aixs will
        // be orthogonal to the current one
        rayon::join(
            || {
                rcb_3d_recurse(
                    points,
                    weights,
                    left_permu,
                    partition,
                    n_iter - 1,
                    axis.next(),
                )
            },
            || {
                rcb_3d_recurse(
                    points,
                    weights,
                    right_permu,
                    partition,
                    n_iter - 1,
                    axis.next(),
                )
            },
        );
    }
}

fn axis_sort_3d(points: &[Point3D], permutation: &mut [usize], axis: Axis) {
    match axis {
        Axis::X => permutation.par_sort_by(|i1, i2| fast_f64_cmp(points[*i1].x, points[*i2].x)),
        Axis::Y => permutation.par_sort_by(|i1, i2| fast_f64_cmp(points[*i1].y, points[*i2].y)),
        Axis::Z => permutation.par_sort_by(|i1, i2| fast_f64_cmp(points[*i1].z, points[*i2].z)),
    }
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
pub fn rcb_nd(points: &[Point], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    let dim = points[0].len();
    let len = points.len();

    let mut permutation = (0..len).into_par_iter().collect::<Vec<_>>();
    let initial_id = ProcessUniqueId::new();
    let mut initial_partition = rayon::iter::repeat(initial_id)
        .take(len)
        .collect::<Vec<_>>();

    rcb_nd_recurse(
        points,
        weights,
        &mut permutation,
        &AtomicPtr::new(initial_partition.as_mut_ptr()),
        n_iter,
        dim,
        0,
    );
    initial_partition
}

fn rcb_nd_recurse(
    points: &[Point],
    weights: &[f64],
    permutation: &mut [usize],
    partition: &AtomicPtr<ProcessUniqueId>,
    n_iter: usize,
    dim: usize,
    current_coord: u32,
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
        let normal = canonical_vector(dim, current_coord as usize);
        axis_sort_nd(points, permutation, &normal);
        let split_pos = half_weight_pos_permu(weights, permutation);

        let (left_permu, right_permu) = permutation.split_at_mut(split_pos);

        rayon::join(
            || {
                rcb_nd_recurse(
                    points,
                    weights,
                    left_permu,
                    partition,
                    n_iter - 1,
                    dim,
                    (current_coord + 1) % dim as u32,
                )
            },
            || {
                rcb_nd_recurse(
                    points,
                    weights,
                    right_permu,
                    partition,
                    n_iter - 1,
                    dim,
                    (current_coord + 1) % dim as u32,
                )
            },
        );
    }
}

fn axis_sort_nd(points: &[Point], permutation: &mut [usize], direction: &DVector<f64>) {
    let base_shift = householder_reflection(&direction);
    let new_points = points.iter().map(|p| &base_shift * p).collect::<Vec<_>>();
    permutation.par_sort_by(|p1, p2| fast_f64_cmp(new_points[*p1][0], new_points[*p2][0]));
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
pub fn rib_2d(points: &[Point2D], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    // Compute the inertia vector of the set of points
    let j = inertia_matrix_2d(weights, points);
    let inertia = intertia_vector_2d(j);

    // In this implementation, the separator is not actually
    // parallel to the inertia vector. Instead, a global rotation
    // is performed on each point such that the inertia vector is
    // mapped to be parallel to the x axis.

    // such a rotation is given by the following angle
    // alpha = arccos(dot(inertia, x_axis) / (norm(inertia) * norm(x_axis)))
    // In our case, both the inertia and x_axis vector are unit vector.
    let x_unit_vector = Vector2::new(1., 0.);
    let angle = inertia.dot(&x_unit_vector).acos();

    let points = rotate_vec(points.to_vec(), angle);

    // When the rotation is done, we just apply RCB
    rcb_2d(&points, weights, n_iter)
}

pub fn rib_3d(points: &[Point3D], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    let mbr = Mbr3D::from_points(points.iter());

    let points = points
        .par_iter()
        .map(|p| mbr.mbr_to_aabb(p))
        .collect::<Vec<_>>();

    // When the rotation is done, we just apply RCB
    rcb_3d(&points, weights, n_iter)
}

pub fn rib_nd(points: &[Point], weights: &[f64], n_iter: usize) -> Vec<ProcessUniqueId> {
    let j = inertia_matrix_nd(&weights, &points);
    let inertia = intertia_vector_nd(j);

    let base_shift = householder_reflection(&inertia);
    let points = points
        .into_par_iter()
        .map(|p| &base_shift * p)
        .collect::<Vec<_>>();

    rcb_nd(&points, weights, n_iter)
}

// a faster cmp than a.partial_cmp(&b) but does not handle NaNs
fn fast_f64_cmp(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
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
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort_2d(&points, &mut permutation, true);

        assert_eq!(permutation, vec![5, 2, 3, 6, 4, 0, 1]);
    }

    #[test]
    fn test_axis_sort_y() {
        let points = gen_point_sample();
        let mut permutation = (0..points.len()).collect::<Vec<usize>>();

        axis_sort_2d(&points, &mut permutation, false);

        assert_eq!(permutation, vec![3, 6, 5, 1, 0, 2, 4]);
    }

    #[test]
    fn test_hyperplane_sort() {
        let points = vec![
            Point::from_row_slice(3, &[0., 0., 0.]),
            Point::from_row_slice(3, &[-1., -1., 1.]),
            Point::from_row_slice(3, &[-2., -2., 2.]),
        ];
        let mut permutation: Vec<usize> = (0..3).collect();

        axis_sort_nd(
            &points,
            &mut permutation,
            &Point::from_row_slice(3, &[1., 1., -1.]),
        );

        assert_eq!(permutation, vec![2, 1, 0]);
    }
}
