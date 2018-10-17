//! A few useful geometric types

use itertools::Itertools;
use nalgebra::{Matrix2, SymmetricEigen, Vector2, Vector3};
use rayon::prelude::*;

pub type Point2D = Vector2<f64>;
pub type Point3D = Vector3<f64>;

/// An Axis Aligned Bounding Box
#[derive(Debug, Clone, Copy)]
pub struct Aabb2D {
    p_min: Point2D,
    p_max: Point2D,
}

impl Aabb2D {
    pub fn new(p_min: Point2D, p_max: Point2D) -> Self {
        Self { p_min, p_max }
    }

    /// Constructs a new `Aabb2D` from an iterator of `Point2D`.
    ///
    /// The resulting `Aabb` is the smallest Aabb that contains every points of the iterator.
    pub fn from_points<'a, I>(points: I) -> Self
    where
        I: Iterator<Item = &'a Point2D> + Clone,
    {
        use itertools::MinMaxResult::*;

        // FIX: what happens if the points are aligned and parallel to x or y axis?
        let (x_min, x_max) = match points.clone().map(|p| p.x).minmax() {
            MinMax(x_min, x_max) => (x_min, x_max),
            _ => panic!("Cannot construct a bounding box from less than two points"),
        };

        let (y_min, y_max) = match points.map(|p| p.y).minmax() {
            MinMax(y_min, y_max) => (y_min, y_max),
            _ => unreachable!(),
        };

        Self::new(Point2D::new(x_min, y_min), Point2D::new(x_max, y_max))
    }

    /// Computes the aspect ratio of the Aabb.
    ///
    /// It is defined as follows:
    /// `ratio = max(width/height, height/width)`
    ///
    /// If `width` or `height` is equal to `0` then `ratio = 0`.
    /// Otherwise, `ratio >= 1`
    pub fn aspect_ratio(&self) -> f64 {
        let width = self.p_max.x - self.p_min.x;
        let height = self.p_max.y - self.p_min.y;

        let ratio = width / height;
        if ratio < 1. && ratio.abs() > ::std::f64::EPSILON {
            1. / ratio
        } else {
            ratio
        }
    }

    /// Computes the distance between a point and the current Aabb.
    pub fn distance_to_point(&self, point: &Point2D) -> f64 {
        let clamped_x = match point.x {
            x if x > self.p_max.x => self.p_max.x,
            x if x < self.p_min.x => self.p_min.x,
            x => x,
        };
        let clamped_y = match point.y {
            y if y > self.p_max.y => self.p_max.y,
            y if y < self.p_min.y => self.p_min.y,
            y => y,
        };
        Vector2::new(clamped_x - point.x, clamped_y - point.y).norm()
    }
}

/// A 2D Minimal bounding rectangle.
///
/// A MBR of a set of points is the smallest rectangle that contains every element of the set.
///
/// As opposed to an Aabb, a Mbr is not always axis aligned.
#[derive(Debug, Copy, Clone)]
pub struct Mbr2D {
    aabb: Aabb2D,
    rotation: f64,
}

impl Mbr2D {
    pub fn new(aabb: Aabb2D, rotation: f64) -> Self {
        Self { aabb, rotation }
    }

    /// Constructs a new `Aabb2D` from an iterator of `Point2D`.
    ///
    /// The resulting `Aabb` is the smallest Aabb that contains every points of the iterator.
    pub fn from_points<I, T>(points: I) -> Self
    where
        I: Iterator<Item = T> + Clone,
        T: ::std::ops::Deref<Target = Point2D>,
    {
        let weights = points.clone().map(|_| 1.);
        let mat = inertia_matrix(
            &weights.collect::<Vec<_>>(),
            &points.clone().map(|v| *v).collect::<Vec<_>>(),
        );
        let vec = intertia_vector(mat);
        let angle = vec.dot(&Vector2::new(1., 0.)).acos();

        Self {
            aabb: Aabb2D::from_points(
                rotate(points.map(|v| *v).clone().collect::<Vec<Point2D>>(), angle).iter(),
            ),
            rotation: angle,
        }
    }

    /// Computes the aspect ratio of the Mbr.
    ///
    /// It is defined as follows:
    /// `ratio = max(width/height, height/width)`
    ///
    /// If `width` or `height` is equal to `0` then `ratio = 0`.
    pub fn aspect_ratio(&self) -> f64 {
        self.aabb.aspect_ratio()
    }

    /// Computes the distance between a point and the current mbr.
    pub fn distance_to_point(&self, point: &Point2D) -> f64 {
        self.aabb
            .distance_to_point(&rotate(vec![*point], self.rotation)[0])
    }
}

// Computes the inertia matrix of a
// collection of weighted points.
pub fn inertia_matrix(weights: &[f64], coordinates: &[Point2D]) -> Matrix2<f64> {
    // We compute the centroid of the collection
    // of points which is required to construct
    // the inertia matrix.
    // centroid = (1 / sum(weights)) \sum_i weight_i * point_i
    let total_weight = weights.par_iter().sum::<f64>();
    let centroid = weights
        .par_iter()
        .zip(coordinates)
        .fold(
            || Vector2::new(0., 0.),
            |acc, (w, p)| acc + p.map(|e| e * w),
        ).sum::<Vector2<_>>()
        / total_weight;

    // The inertia matrix is a 2x2 matrix (or a dxd matrix in dimension d)
    // it is defined as follows:
    // J = \sum_i (1/weight_i)*(point_i - centroid) * transpose(point_i - centroid)
    // It is by construction a symmetric matrix
    weights
        .par_iter()
        .zip(coordinates)
        .fold(Matrix2::zeros, |acc, (w, p)| {
            acc + ((p - centroid) * (p - centroid).transpose()).map(|e| e * w)
        }).sum()
}

// Computes an inertia vector of
// an inertia matrix. Is is defined to be
// an eigenvector of the smallest of the
// matrix eigenvalues
pub fn intertia_vector(mat: Matrix2<f64>) -> Vector2<f64> {
    // by construction the inertia matrix is symmetric
    SymmetricEigen::new(mat)
        .eigenvectors
        .column(0)
        .clone_owned()
}

// Rotates each point of an angle (in radians) counter clockwise
pub(crate) fn rotate(coordinates: Vec<Point2D>, angle: f64) -> Vec<Point2D> {
    // A rotation of angle theta is defined in 2D by a 2x2 matrix
    // |  cos(theta) sin(theta) |
    // | -sin(theta) cos(theta) |
    let rot_matrix = Matrix2::new(angle.cos(), angle.sin(), -angle.sin(), angle.cos());
    coordinates
        .into_par_iter()
        .map(|c| rot_matrix * c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb2d() {
        let points = vec![
            Point2D::new(1., 2.),
            Point2D::new(0., 0.),
            Point2D::new(3., 1.),
            Point2D::new(5., 4.),
            Point2D::new(4., 5.),
        ];

        let aabb = Aabb2D::from_points(points.iter());
        let aspect_ratio = aabb.aspect_ratio();

        assert_ulps_eq!(aabb.p_min, Point2D::new(0., 0.));
        assert_ulps_eq!(aabb.p_max, Point2D::new(5., 5.));
        assert_ulps_eq!(aspect_ratio, 1.);
    }

    #[test]
    #[should_panic]
    fn test_aabb2d_invalid_input_1() {
        let points = vec![];
        let _aabb = Aabb2D::from_points(points.iter());
    }

    #[test]
    #[should_panic]
    fn test_aabb2d_invalid_input_2() {
        let points = vec![Point2D::new(5., -9.2)];
        let _aabb = Aabb2D::from_points(points.iter());
    }

    #[test]
    fn test_mbr2d() {
        let points = vec![
            Point2D::new(5., 3.),
            Point2D::new(0., 0.),
            Point2D::new(1., -1.),
            Point2D::new(4., 4.),
        ];

        let mbr = Mbr2D::from_points(points.iter());
        let aspect_ratio = mbr.aspect_ratio();

        assert_ulps_eq!(aspect_ratio, 4.);
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

    #[test]
    fn test_mbr_distance_to_point() {
        let points = vec![
            Point2D::new(0., 1.),
            Point2D::new(1., 0.),
            Point2D::new(5., 6.),
            Point2D::new(6., 5.),
        ];

        let mbr = Mbr2D::from_points(points.iter());

        let test_points = vec![
            Point2D::new(2., 2.),
            Point2D::new(0., 0.),
            Point2D::new(5., 7.),
        ];

        let distances: Vec<_> = test_points
            .iter()
            .map(|p| mbr.distance_to_point(p))
            .collect();

        relative_eq!(distances[0], 0.);
        relative_eq!(distances[1], 2_f64.sqrt() / 2.);
        relative_eq!(distances[2], 1.);
    }
}
