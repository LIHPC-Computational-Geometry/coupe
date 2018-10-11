use geometry::Point2D;
use itertools::{self, Itertools};
use snowflake::ProcessUniqueId;
use std::cmp::Ordering;

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{Matrix2, Vector2};

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
) -> Vec<(usize, ProcessUniqueId)> {
    rcb_impl(ids, weights, coordinates, n_iter, true)
}

fn rcb_impl(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    n_iter: usize,
    x_axis: bool, // set if bissection is performed w.r.t. x or y axis
) -> Vec<(usize, ProcessUniqueId)> {
    if n_iter == 0 {
        // No iteration left. The current
        // ids become a part of the final partition.
        // Generate a partition id and return.
        let part_id = ProcessUniqueId::new();
        ids.into_iter().map(|id| (id, part_id)).collect()
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
        let left_partition = rcb_impl(
            left_ids,
            left_weights,
            left_coordinates,
            n_iter - 1,
            !x_axis,
        );

        let right_partition = rcb_impl(ids, weights, coordinates, n_iter - 1, !x_axis);

        // We stick the partitions back together
        // to return a single collection of objects
        left_partition.into_iter().chain(right_partition).collect()
    }
}

// sort input vectors w.r.t. coordinates
// i.e. by increasing x or increasing y
fn axis_sort(
    ids: Vec<usize>,
    weights: Vec<f64>,
    coordinates: Vec<Point2D>,
    x_axis: bool,
) -> (Vec<usize>, Vec<f64>, Vec<Point2D>) {
    // We use multizip here
    // to sort the three collections
    // at once with a single critertion
    // on the coordinates
    let sorted =
        itertools::multizip((ids, weights, coordinates)).sorted_by(|(_, _, p1), (_, _, p2)| {
            if x_axis {
                p1.x.partial_cmp(&p2.x).unwrap_or(Ordering::Equal)
            } else {
                p1.y.partial_cmp(&p2.y).unwrap_or(Ordering::Equal)
            }
        });

    let (ids, weights, coordinates) = unzip(sorted.into_iter());
    (ids.collect(), weights.collect(), coordinates.collect())
}

// Computes a slice index which splits
// the slice in two parts of equal weights
// i.e. sorted_weights[..idx].sum() == sorted_weights[idx..].sum
fn half_weight_pos(sorted_weights: &[f64]) -> usize {
    let mut half_weight = sorted_weights.iter().sum::<f64>() / 2.;
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

// quick and dirty equivalent of std::iter::unzip for itertools::multizip
fn unzip<A, B, C>(
    zipped: impl Iterator<Item = (A, B, C)> + Clone,
) -> (
    impl Iterator<Item = A>,
    impl Iterator<Item = B>,
    impl Iterator<Item = C>,
)
where
    A: Clone,
    B: Clone,
    C: Clone,
{
    (
        zipped.clone().map(|(a, _, _)| a.clone()),
        zipped.clone().map(|(_, b, _)| b),
        zipped.map(|(_, _, c)| c),
    )
}

// Computes the inertia matrix of a
// collection of weighted points.
pub fn inertia_matrix(weights: &[f64], coordinates: &[Point2D]) -> Matrix2<f64> {
    // We compute the centroid of the collection
    // of points which is required to construct
    // the inertia matrix.
    // centroid = (1 / sum(weights)) \sum_i weight_i * point_i
    let total_weight = weights.iter().sum::<f64>();
    let centroid = weights
        .iter()
        .zip(coordinates)
        .fold(Vector2::new(0., 0.), |acc, (w, p)| acc + p.map(|e| e * w))
        / total_weight;

    // The inertia matrix is a 2x2 matrix (or a dxd matrix in dimension d)
    // it is defined as follows:
    // J = \sum_i (1/weight_i)*(point_i - centroid) * transpose(point_i - centroid)
    // It is by construction a symmetric matrix
    weights
        .iter()
        .zip(coordinates)
        .fold(Matrix2::zeros(), |acc, (w, p)| {
            acc + ((p - centroid) * (p - centroid).transpose()).map(|e| e * w)
        })
}

// Computes an inertia vector of
// an inertia matrix. Is is defined to be
// an eigenvector of the smallest of the
// matrix eigenvalues
pub fn intertia_vector(mat: Matrix2<f64>) -> Vector2<f64> {
    // by construction the inertia matrix is symmetric
    let sym = SymmetricEigen::new(mat);
    if sym.eigenvalues[0] <= sym.eigenvalues[1] {
        sym.eigenvectors.column(0).clone_owned()
    } else {
        sym.eigenvectors.column(0).clone_owned()
    }
}

// Rotates each point of an angle (in radians) counter clockwise
fn rotate(coordinates: Vec<Point2D>, angle: f64) -> Vec<Point2D> {
    // A rotation of angle theta is defined in 2D by a 2x2 matrix
    // |  cos(theta) sin(theta) |
    // | -sin(theta) cos(theta) |
    let rot_matrix = Matrix2::new(angle.cos(), angle.sin(), -angle.sin(), angle.cos());
    coordinates.into_iter().map(|c| rot_matrix * c).collect()
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
) -> Vec<(usize, ProcessUniqueId)> {
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
    use nalgebra::Vector3;

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

    #[test]
    fn test_inertia_matrix() {
        let points = vec![
            Point2D::new(3., 0.),
            Point2D::new(0., 3.),
            Point2D::new(6., -3.),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix(&weights, &points);
        let expected = Matrix2::new(18., -18., -18., 18.);

        assert_ulps_eq!(mat, expected);
    }

    #[test]
    fn test_inertia_vector() {
        let points = vec![
            Point2D::new(3., 0.),
            Point2D::new(0., 3.),
            Point2D::new(6., -3.),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix(&weights, &points);
        let vec = intertia_vector(mat);
        let vec = Vector3::new(vec.x, vec.y, 0.);
        let expected = Vector3::<f64>::new(1., -1., 0.);

        eprintln!("{}", vec);

        assert_ulps_eq!(expected.cross(&vec).norm(), 0.);
    }
}
