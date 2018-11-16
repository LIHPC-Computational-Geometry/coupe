//! A few useful geometric types

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, SymmetricEigen, Vector2, Vector3};
use rayon::prelude::*;

pub type Point2D = Vector2<f64>;
pub type Point3D = Vector3<f64>;
pub type Point = DVector<f64>;

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

    pub fn p_min(&self) -> Point2D {
        self.p_min
    }

    pub fn p_max(&self) -> Point2D {
        self.p_max
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

    /// Constructs a new Aabb that is a sub-Aabb of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_aabb(&self, quadrant: Quadrant) -> Self {
        use self::Quadrant::*;
        let center = self.center();
        match quadrant {
            BottomLeft => Aabb2D::new(self.p_min, center),
            BottomRight => Aabb2D::new(
                Point2D::new(center.x, self.p_min.y),
                Point2D::new(self.p_max.x, center.y),
            ),
            TopLeft => Aabb2D::new(
                Point2D::new(self.p_min.x, center.y),
                Point2D::new(center.x, self.p_max.y),
            ),
            TopRight => Aabb2D::new(center, self.p_max),
        }
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
        if !self.contains(point) {
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
        } else {
            let center = self.center();
            let x_dist = if point.x > center.x {
                (self.p_max.x - point.x).abs()
            } else {
                (self.p_min.x - point.x).abs()
            };
            let y_dist = if point.y > center.y {
                (self.p_max.y - point.y).abs()
            } else {
                (self.p_min.y - point.y).abs()
            };

            x_dist.max(y_dist)
        }
    }

    /// Computes the center of the Aabb
    pub fn center(&self) -> Point2D {
        Point2D::new(
            0.5 * (self.p_min.x + self.p_max.x),
            0.5 * (self.p_min.y + self.p_max.y),
        )
    }

    /// Returns wheter or not the specified point is contained in the Aabb
    pub fn contains(&self, point: &Point2D) -> bool {
        let eps = 10. * ::std::f64::EPSILON;
        point.x < self.p_max.x + eps
            && point.x > self.p_min.x - eps
            && point.y < self.p_max.y + eps
            && point.y > self.p_min.y - eps
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// Returns `None` if the specified point is not contained in the Aabb
    pub fn quadrant(&self, point: &Point2D) -> Option<Quadrant> {
        use self::Quadrant::*;
        if !self.contains(point) {
            return None;
        }

        let center = self.center();

        match (point.x, point.y) {
            (x, y) if x < center.x && y < center.y => Some(BottomLeft),
            (x, y) if x > center.x && y < center.y => Some(BottomRight),
            (x, y) if x < center.x && y > center.y => Some(TopLeft),
            (x, y) if x > center.x && y > center.y => Some(TopRight),
            _ => unreachable!(),
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quadrant {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
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

    pub fn rotation(&self) -> f64 {
        self.rotation
    }

    pub fn aabb(&self) -> &Aabb2D {
        &self.aabb
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
        let mat = inertia_matrix_2d(
            &weights.collect::<Vec<_>>(),
            &points.clone().map(|v| *v).collect::<Vec<_>>(),
        );
        let vec = intertia_vector_2d(mat);
        let angle = vec.dot(&Vector2::new(1., 0.)).acos();

        Self {
            aabb: Aabb2D::from_points(
                rotate_vec(points.map(|v| *v).clone().collect::<Vec<Point2D>>(), angle).iter(),
            ),
            rotation: angle,
        }
    }

    /// Constructs a new Mbr that is a sub-Mbr of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_mbr(&self, quadrant: Quadrant) -> Self {
        Self {
            aabb: self.aabb.sub_aabb(quadrant),
            rotation: self.rotation,
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
            .distance_to_point(&rotate_vec(vec![*point], self.rotation)[0])
    }

    /// Computes the center of the Mbr
    pub fn center(&self) -> Point2D {
        rotate_vec(vec![self.aabb.center()], -self.rotation)[0]
    }

    /// Returns wheter or not the specified point is contained in the Mbr
    pub fn contains(&self, point: &Point2D) -> bool {
        self.aabb
            .contains(&rotate_vec(vec![*point], self.rotation)[0])
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// A Mbr quadrant is defined as a quadrant of the associated Aabb.
    /// Returns `None` if the specified point is not contained in the Aabb.
    pub fn quadrant(&self, point: &Point2D) -> Option<Quadrant> {
        self.aabb
            .quadrant(&rotate_vec(vec![*point], self.rotation)[0])
    }

    /// Returns the rotated min and max points of the Aabb.
    pub fn minmax(&self) -> (Point2D, Point2D) {
        let minmax = rotate_vec(vec![self.aabb.p_min, self.aabb.p_max], -self.rotation);
        (minmax[0], minmax[1])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb3D {
    p_min: Point3D,
    p_max: Point3D,
}

impl Aabb3D {
    pub fn new(p_min: Point3D, p_max: Point3D) -> Self {
        Self { p_min, p_max }
    }

    pub fn p_min(&self) -> Point3D {
        self.p_min
    }

    pub fn p_max(&self) -> Point3D {
        self.p_max
    }

    /// Constructs a new `Aabb2D` from an iterator of `Point2D`.
    ///
    /// The resulting `Aabb` is the smallest Aabb that contains every points of the iterator.
    pub fn from_points<'a, I>(points: I) -> Self
    where
        I: Iterator<Item = &'a Point3D> + Clone,
    {
        use itertools::MinMaxResult::*;

        // FIX: what happens if the points are aligned and parallel to x or y axis?
        let (x_min, x_max) = match points.clone().map(|p| p.x).minmax() {
            MinMax(x_min, x_max) => (x_min, x_max),
            _ => panic!("Cannot construct a bounding box from less than two points"),
        };

        let (y_min, y_max) = match points.clone().map(|p| p.y).minmax() {
            MinMax(y_min, y_max) => (y_min, y_max),
            _ => unreachable!(),
        };

        let (z_min, z_max) = match points.map(|p| p.z).minmax() {
            MinMax(z_min, z_max) => (z_min, z_max),
            _ => unreachable!(),
        };

        Self::new(
            Point3D::new(x_min, y_min, z_min),
            Point3D::new(x_max, y_max, z_max),
        )
    }

    /// Constructs a new Aabb that is a sub-Aabb of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_aabb(&self, octant: Octant) -> Self {
        use self::Octant::*;
        let center = self.center();
        match octant {
            BottomLeftNear => Aabb3D::new(self.p_min, center),
            BottomLeftFar => Aabb3D::new(
                Point3D::new(self.p_min.x, self.p_min.y, center.z),
                Point3D::new(center.x, center.y, self.p_max.z),
            ),
            BottomRightNear => Aabb3D::new(
                Point3D::new(center.x, self.p_min.y, self.p_min.z),
                Point3D::new(self.p_max.x, center.y, center.z),
            ),
            BottomRightFar => Aabb3D::new(
                Point3D::new(center.x, self.p_min.y, center.z),
                Point3D::new(self.p_max.x, center.y, self.p_max.z),
            ),
            TopLeftNear => Aabb3D::new(
                Point3D::new(self.p_min.x, center.y, self.p_min.z),
                Point3D::new(center.x, self.p_max.y, center.z),
            ),
            TopLeftFar => Aabb3D::new(
                Point3D::new(self.p_min.x, center.y, center.z),
                Point3D::new(center.x, self.p_max.y, self.p_max.z),
            ),
            TopRightNear => Aabb3D::new(
                Point3D::new(center.x, center.y, self.p_min.z),
                Point3D::new(self.p_max.x, self.p_max.y, center.z),
            ),
            TopRightFar => Aabb3D::new(center, self.p_max),
        }
    }

    /// Computes the aspect ratio of the Aabb.
    ///
    /// It is defined as follows:
    /// `ratio = long_side / short_side`
    ///
    /// If `width` or `height` is equal to `0` then `ratio = 0`.
    /// Otherwise, `ratio >= 1`
    pub fn aspect_ratio(&self) -> f64 {
        let width = self.p_max.x - self.p_min.x;
        let height = self.p_max.y - self.p_min.y;
        let depth = self.p_max.z - self.p_min.z;

        let mut array = [width, height, depth];

        (&mut array[..]).sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());

        if array[0] < std::f64::EPSILON {
            0.
        } else {
            array[2] / array[0]
        }
    }

    /// Computes the distance between a point and the current Aabb.
    pub fn distance_to_point(&self, point: &Point3D) -> f64 {
        if !self.contains(point) {
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
            let clamped_z = match point.z {
                z if z > self.p_max.z => self.p_max.z,
                z if z < self.p_min.z => self.p_min.z,
                z => z,
            };
            Vector3::new(
                clamped_x - point.x,
                clamped_y - point.y,
                clamped_z - point.z,
            ).norm()
        } else {
            let center = self.center();
            let x_dist = if point.x > center.x {
                (self.p_max.x - point.x).abs()
            } else {
                (self.p_min.x - point.x).abs()
            };
            let y_dist = if point.y > center.y {
                (self.p_max.y - point.y).abs()
            } else {
                (self.p_min.y - point.y).abs()
            };
            let z_dist = if point.z > center.z {
                (self.p_max.z - point.z).abs()
            } else {
                (self.p_min.z - point.z).abs()
            };

            x_dist.max(y_dist).max(z_dist)
        }
    }

    /// Computes the center of the Aabb
    pub fn center(&self) -> Point3D {
        Point3D::new(
            0.5 * (self.p_min.x + self.p_max.x),
            0.5 * (self.p_min.y + self.p_max.y),
            0.5 * (self.p_min.z + self.p_max.z),
        )
    }

    /// Returns wheter or not the specified point is contained in the Aabb
    pub fn contains(&self, point: &Point3D) -> bool {
        let eps = 10. * ::std::f64::EPSILON;
        point.x < self.p_max.x + eps
            && point.x > self.p_min.x - eps
            && point.y < self.p_max.y + eps
            && point.y > self.p_min.y - eps
            && point.z < self.p_max.z + eps
            && point.z > self.p_min.z - eps
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// Returns `None` if the specified point is not contained in the Aabb
    pub fn octant(&self, point: &Point3D) -> Option<Octant> {
        use self::Octant::*;
        if !self.contains(point) {
            return None;
        }

        let center = self.center();

        match (point.x, point.y, point.z) {
            (x, y, z) if x < center.x && y < center.y && z < center.z => Some(BottomLeftNear),
            (x, y, z) if x < center.x && y < center.y && z > center.z => Some(BottomLeftFar),
            (x, y, z) if x > center.x && y < center.y && z < center.z => Some(BottomRightNear),
            (x, y, z) if x > center.x && y < center.y && z > center.z => Some(BottomRightFar),
            (x, y, z) if x < center.x && y > center.y && z < center.z => Some(TopLeftNear),
            (x, y, z) if x < center.x && y > center.y && z > center.z => Some(TopLeftFar),
            (x, y, z) if x > center.x && y > center.y && z < center.z => Some(TopRightNear),
            (x, y, z) if x > center.x && y > center.y && z > center.z => Some(TopRightFar),
            _ => unreachable!(),
        }
    }
}

/// A 3D Minimal bounding rectangle.
///
/// A MBR of a set of points is the smallest box that contains every element of the set.
///
/// As opposed to an Aabb, a Mbr is not always axis aligned.
#[derive(Debug, Copy, Clone)]
pub struct Mbr3D {
    aabb: Aabb3D,
    aabb_to_mbr: Matrix3<f64>, // base change matrix that maps the aabb to the mbr
    mbr_to_aabb: Matrix3<f64>, // base change matrix that maps the mbr to the aabb
}

impl Mbr3D {
    pub fn new(aabb: Aabb3D, aabb_to_mbr: Matrix3<f64>, mbr_to_aabb: Matrix3<f64>) -> Self {
        Self {
            aabb,
            aabb_to_mbr,
            mbr_to_aabb,
        }
    }

    pub fn aabb(&self) -> &Aabb3D {
        &self.aabb
    }

    /// Constructs a new `Mbr3D` from an iterator of `Point3D`.
    ///
    /// The resulting `Mbr3D` is the smallest Aabb that contains every points of the iterator.
    pub fn from_points<I, T>(points: I) -> Self
    where
        I: Iterator<Item = T> + Clone,
        T: ::std::ops::Deref<Target = Point3D>,
    {
        let weights = points.clone().map(|_| 1.);
        let mat = inertia_matrix_3d(
            &weights.collect::<Vec<_>>(),
            &points.clone().map(|v| *v).collect::<Vec<_>>(),
        );
        let vec = intertia_vector_3d(mat);
        let base_change = householder_reflection_3d(&vec);
        let mbr_to_aabb = base_change.try_inverse().unwrap();
        let mapped = points.map(|p| mbr_to_aabb * *p).collect::<Vec<_>>();
        let aabb = Aabb3D::from_points(mapped.iter());

        Self {
            aabb,
            aabb_to_mbr: base_change,
            mbr_to_aabb,
        }
    }

    /// Constructs a new Mbr that is a sub-Mbr of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_mbr(&self, octant: Octant) -> Self {
        Self {
            aabb: self.aabb.sub_aabb(octant),
            aabb_to_mbr: self.aabb_to_mbr,
            mbr_to_aabb: self.mbr_to_aabb,
        }
    }

    /// Computes the aspect ratio of the Mbr.
    ///
    /// It is defined as follows:
    /// `ratio = long_side / short_side`
    pub fn aspect_ratio(&self) -> f64 {
        self.aabb.aspect_ratio()
    }

    /// Computes the distance between a point and the current mbr.
    pub fn distance_to_point(&self, point: &Point3D) -> f64 {
        self.aabb.distance_to_point(&(self.mbr_to_aabb * *point))
    }

    /// Computes the center of the Mbr
    pub fn center(&self) -> Point3D {
        self.aabb_to_mbr * self.aabb.center()
    }

    /// Returns wheter or not the specified point is contained in the Mbr
    pub fn contains(&self, point: &Point3D) -> bool {
        self.aabb.contains(&(self.mbr_to_aabb * point))
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// A Mbr quadrant is defined as a quadrant of the associated Aabb.
    /// Returns `None` if the specified point is not contained in the Aabb.
    pub fn octant(&self, point: &Point3D) -> Option<Octant> {
        self.aabb.octant(&(self.mbr_to_aabb * *point))
    }

    /// Returns the rotated min and max points of the Aabb.
    pub fn minmax(&self) -> (Point3D, Point3D) {
        let min = self.aabb_to_mbr * self.aabb.p_min;
        let max = self.aabb_to_mbr * self.aabb.p_max;
        (min, max)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Octant {
    BottomLeftNear,
    BottomLeftFar,
    BottomRightNear,
    BottomRightFar,
    TopLeftNear,
    TopLeftFar,
    TopRightNear,
    TopRightFar,
}

// Computes the inertia matrix of a
// collection of weighted points.
pub fn inertia_matrix_2d(weights: &[f64], coordinates: &[Point2D]) -> Matrix2<f64> {
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

// Computes the inertia matrix of a
// collection of weighted points.
pub fn inertia_matrix_3d(weights: &[f64], coordinates: &[Point3D]) -> Matrix3<f64> {
    // We compute the centroid of the collection
    // of points which is required to construct
    // the inertia matrix.
    // centroid = (1 / sum(weights)) \sum_i weight_i * point_i
    let total_weight = weights.par_iter().sum::<f64>();
    let centroid = weights
        .par_iter()
        .zip(coordinates)
        .fold_with(Vector3::new(0., 0., 0.), |acc, (w, p)| {
            acc + p.map(|e| e * w)
        }).sum::<Vector3<_>>()
        / total_weight;

    // The inertia matrix is a 2x2 matrix (or a dxd matrix in dimension d)
    // it is defined as follows:
    // J = \sum_i (1/weight_i)*(point_i - centroid) * transpose(point_i - centroid)
    // It is by construction a symmetric matrix
    weights
        .par_iter()
        .zip(coordinates)
        .fold(Matrix3::zeros, |acc, (w, p)| {
            acc + ((p - centroid) * (p - centroid).transpose()).map(|e| e * w)
        }).sum()
}

// Computes an inertia vector of
// an inertia matrix. Is is defined to be
// an eigenvector of the smallest of the
// matrix eigenvalues
pub fn intertia_vector_3d(mat: Matrix3<f64>) -> Vector3<f64> {
    // by construction the inertia matrix is symmetric
    SymmetricEigen::new(mat)
        .eigenvectors
        .column(0)
        .clone_owned()
}

// Computes an inertia vector of
// an inertia matrix. Is is defined to be
// an eigenvector of the smallest of the
// matrix eigenvalues
pub fn intertia_vector_2d(mat: Matrix2<f64>) -> Vector2<f64> {
    // by construction the inertia matrix is symmetric
    SymmetricEigen::new(mat)
        .eigenvectors
        .column(0)
        .clone_owned()
}

pub fn inertia_matrix_nd(weights: &[f64], points: &[Point]) -> DMatrix<f64> {
    let dim = points[0].len();
    let total_weight = weights.par_iter().sum::<f64>();

    // TODO: refactor this ugly fold -> reduce into a single reduce
    let centroid: Point = weights
        .par_iter()
        .zip(points)
        .fold(
            || Point::from_row_slice(dim, &vec![0.; dim]),
            |acc, (w, p)| acc + p.map(|e| e * w),
        ).reduce(|| Point::from_row_slice(dim, &vec![0.; dim]), |a, b| a + b)
        / total_weight;

    let ret: DMatrix<f64> = weights
        .par_iter()
        .zip(points)
        .fold(
            || DMatrix::zeros(dim, dim),
            |acc, (w, p)| acc + ((p - &centroid) * (p - &centroid).transpose()).map(|e| e * w),
        ).reduce(|| DMatrix::zeros(dim, dim), |a, b| a + b);

    ret
}

pub fn intertia_vector_nd(mat: DMatrix<f64>) -> DVector<f64> {
    // by construction the inertia matrix is symmetric
    SymmetricEigen::new(mat)
        .eigenvectors
        .column(0)
        .clone_owned()
}

// Rotates each point of an angle (in radians) counter clockwise
pub(crate) fn rotate_vec(coordinates: Vec<Point2D>, angle: f64) -> Vec<Point2D> {
    // A rotation of angle theta is defined in 2D by a 2x2 matrix
    // |  cos(theta) sin(theta) |
    // | -sin(theta) cos(theta) |
    let rot_matrix = Matrix2::new(angle.cos(), angle.sin(), -angle.sin(), angle.cos());
    coordinates
        .into_par_iter()
        .map(|c| rot_matrix * c)
        .collect()
}

pub(crate) fn rotation(angle: f64) -> impl Fn((f64, f64)) -> (f64, f64) {
    let rot_matrix = Matrix2::new(angle.cos(), angle.sin(), -angle.sin(), angle.cos());
    move |(x, y)| {
        let ret = rot_matrix * Point2D::new(x, y);
        (ret.x, ret.y)
    }
}

pub fn center(points: &[Point2D]) -> Point2D {
    points.iter().fold(Point2D::new(0., 0.), |acc, val| {
        Point2D::new(acc.x + val.x, acc.y + val.y)
    }) / points.len() as f64
}

// The Householder reflexion algorithm.
// From a given vector v of dimension N, the algorithm will yield a square matrix Q of size N such that:
//  - The first colunm of Q is parallel to v.
//  - The columns of Q are an orthonormal basis of R^N.
// Complexity: O(N^2)
// Based from: https://math.stackexchange.com/questions/710103/algorithm-to-find-an-orthogonal-basis-orthogonal-to-a-given-vector
pub fn householder_reflection(element: &DVector<f64>) -> DMatrix<f64> {
    let dim = element.len();
    let e0 = canonical_vector(dim, 0);
    let sign = if element[0] > 0. { -1. } else { 1. };
    let w = element + sign * e0 * element.norm();
    let id = DMatrix::<f64>::identity(dim, dim);
    id - 2. * &w * w.transpose() / (w.transpose() * w)[0]
}

// This might actually be overkill for 3d but it's a copy/paste from nd implementation
pub fn householder_reflection_3d(element: &Vector3<f64>) -> Matrix3<f64> {
    let e0 = Vector3::new(1., 0., 0.);
    let sign = if element.x > 0. { -1. } else { 1. };
    let w: Vector3<f64> = element + sign * e0 * element.norm();
    let id = Matrix3::<f64>::identity();
    id - 2. * &w * w.transpose() / (w.transpose() * w)[0]
}

pub(crate) fn canonical_vector(dim: usize, nth: usize) -> DVector<f64> {
    let mut ret = DVector::<f64>::from_element(dim, 0.);
    ret[nth] = 1.0;
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_2d() {
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
    fn test_mbr_2d() {
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

        let mat = inertia_matrix_2d(&weights, &points);
        let expected = Matrix2::new(18., -18., -18., 18.);

        assert_ulps_eq!(mat, expected);
    }

    #[test]
    fn test_inertia_vector_2d() {
        let points = vec![
            Point2D::new(3., 0.),
            Point2D::new(0., 3.),
            Point2D::new(6., -3.),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix_2d(&weights, &points);
        let vec = intertia_vector_2d(mat);
        let vec = Vector3::new(vec.x, vec.y, 0.);
        let expected = Vector3::<f64>::new(1., -1., 0.);

        eprintln!("{}", vec);

        assert_ulps_eq!(expected.cross(&vec).norm(), 0.);
    }

    #[test]
    fn test_mbr_distance_to_point_2d() {
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

        relative_eq!(distances[0], 0.5);
        relative_eq!(distances[1], 2_f64.sqrt() / 2.);
        relative_eq!(distances[2], 1.);
    }

    #[test]
    fn test_mbr_center() {
        let points = vec![
            Point2D::new(0., 1.),
            Point2D::new(1., 0.),
            Point2D::new(5., 6.),
            Point2D::new(6., 5.),
        ];

        let mbr = Mbr2D::from_points(points.iter());

        let center = mbr.center();
        assert_ulps_eq!(center, Point2D::new(3., 3.))
    }

    #[test]
    fn test_mbr_contains() {
        let points = vec![
            Point2D::new(0., 1.),
            Point2D::new(1., 0.),
            Point2D::new(5., 6.),
            Point2D::new(6., 5.),
        ];

        let mbr = Mbr2D::from_points(points.iter());

        assert!(!mbr.contains(&Point2D::new(0., 0.)));
        assert!(mbr.contains(&mbr.center()));
        assert!(mbr.contains(&Point2D::new(5., 4.)));

        let (min, max) = mbr.minmax();

        assert!(mbr.contains(&min));
        assert!(mbr.contains(&max));
    }

    #[test]
    fn test_mbr_quadrant() {
        use super::Quadrant::*;
        let points = vec![
            Point2D::new(0., 1.),
            Point2D::new(1., 0.),
            Point2D::new(5., 6.),
            Point2D::new(6., 5.),
        ];

        let mbr = Mbr2D::from_points(points.iter());

        let none = mbr.quadrant(&Point2D::new(0., 0.));
        let q1 = mbr.quadrant(&Point2D::new(1.2, 1.2));
        let q2 = mbr.quadrant(&Point2D::new(5.8, 4.9));
        let q3 = mbr.quadrant(&Point2D::new(0.2, 1.1));
        let q4 = mbr.quadrant(&Point2D::new(5.1, 5.8));

        assert_eq!(none, None);
        assert_eq!(q1, Some(BottomLeft));
        assert_eq!(q2, Some(BottomRight));
        assert_eq!(q3, Some(TopLeft));
        assert_eq!(q4, Some(TopRight));
    }

    #[test]
    fn test_householder_reflexion() {
        let el = DVector::<f64>::new_random(11);
        let mat = householder_reflection(&el);

        // check that columns are of norm 1
        for col in 0..10 {
            assert_ulps_eq!(mat.column(col).norm(), 1.);
        }

        // check that columns are orthogonal
        for col1 in 0..10 {
            for col2 in 0..10 {
                if col1 != col2 {
                    relative_eq!(mat.column(col1).dot(&mat.column(col2)), 0.);
                }
            }
        }

        // check that first column is parallel to el
        let unit_el = &el / el.norm();
        let fst_col = mat.column(0).clone_owned();
        relative_eq!(&unit_el * unit_el.dot(&fst_col), fst_col);
    }

    #[test]
    fn test_aabb_3d() {
        let points = vec![
            Point3D::new(1., 2., 0.),
            Point3D::new(0., 0., 5.),
            Point3D::new(3., 1., 1.),
            Point3D::new(5., 4., -2.),
            Point3D::new(4., 5., 3.),
        ];

        let aabb = Aabb3D::from_points(points.iter());
        let aspect_ratio = aabb.aspect_ratio();

        assert_ulps_eq!(aabb.p_min, Point3D::new(0., 0., -2.));
        assert_ulps_eq!(aabb.p_max, Point3D::new(5., 5., 5.));
        assert_ulps_eq!(aspect_ratio, 7. / 5.);
    }

    #[test]
    fn test_mbr_3d() {
        let points = vec![
            Point3D::new(5., 3., 0.),
            Point3D::new(0., 0., 0.),
            Point3D::new(1., -1., 0.),
            Point3D::new(4., 4., 0.),
            Point3D::new(5., 3., 2.),
            Point3D::new(0., 0., 2.),
            Point3D::new(1., -1., 2.),
            Point3D::new(4., 4., 2.),
        ];

        let mbr = Mbr3D::from_points(points.iter());
        println!("mbr = {:?}", mbr);
        let aspect_ratio = mbr.aspect_ratio();

        relative_eq!(aspect_ratio, 4.);
    }

    #[test]
    fn test_inertia_vector_3d() {
        let points = vec![
            Point3D::new(3., 0., 0.),
            Point3D::new(0., 3., 3.),
            Point3D::new(6., -3., -3.),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix_3d(&weights, &points);
        let vec = intertia_vector_3d(mat);
        let expected = Vector3::<f64>::new(1., -1., -1.);

        eprintln!("{}", vec);

        relative_eq!(expected.cross(&vec).norm(), 0.);
    }

    #[test]
    fn test_mbr_distance_to_point_3d() {
        let points = vec![
            Point3D::new(0., 1., 0.),
            Point3D::new(1., 0., 0.),
            Point3D::new(5., 6., 0.),
            Point3D::new(6., 5., 0.),
            Point3D::new(0., 1., 4.),
            Point3D::new(1., 0., 4.),
            Point3D::new(5., 6., 4.),
            Point3D::new(6., 5., 4.),
        ];

        let mbr = Mbr3D::from_points(points.iter());

        let test_points = vec![
            Point3D::new(2., 2., 1.),
            Point3D::new(0., 0., 1.),
            Point3D::new(5., 7., 1.),
            Point3D::new(2., 2., 3.),
            Point3D::new(0., 0., 3.),
            Point3D::new(5., 7., 3.),
        ];

        let distances: Vec<_> = test_points
            .iter()
            .map(|p| mbr.distance_to_point(p))
            .collect();

        relative_eq!(distances[0], 0.5);
        relative_eq!(distances[1], 2_f64.sqrt() / 2.);
        relative_eq!(distances[2], 1.);
        relative_eq!(distances[3], 0.5);
        relative_eq!(distances[4], 2_f64.sqrt() / 2.);
        relative_eq!(distances[5], 1.);
    }

    #[test]
    fn test_mbr_octant() {
        let points = vec![
            Point3D::new(0., 1., 0.),
            Point3D::new(1., 0., 0.),
            Point3D::new(5., 6., 0.),
            Point3D::new(6., 5., 0.),
            Point3D::new(0., 1., 1.),
            Point3D::new(1., 0., 1.),
            Point3D::new(5., 6., 1.),
            Point3D::new(6., 5., 1.),
        ];

        let mbr = Mbr3D::from_points(points.iter());
        eprintln!("mbr = {:#?}", mbr);

        let none = mbr.octant(&Point3D::new(0., 0., 0.));
        let octants = vec![
            mbr.octant(&Point3D::new(1.2, 1.2, 0.3)),
            mbr.octant(&Point3D::new(5.8, 4.9, 0.3)),
            mbr.octant(&Point3D::new(0.2, 1.1, 0.3)),
            mbr.octant(&Point3D::new(5.1, 5.8, 0.3)),
            mbr.octant(&Point3D::new(1.2, 1.2, 0.7)),
            mbr.octant(&Point3D::new(5.8, 4.9, 0.7)),
            mbr.octant(&Point3D::new(0.2, 1.1, 0.7)),
            mbr.octant(&Point3D::new(5.1, 5.8, 0.7)),
        ];
        assert_eq!(none, None);
        assert!(octants.iter().all(|o| o.is_some()));
        // we cannot test if each octant is the right one because the orientation of the base
        // change is unspecified
        assert_eq!(8, octants.iter().unique().count());
    }
}
