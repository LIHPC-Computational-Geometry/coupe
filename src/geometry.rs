//! A few useful geometric types

use approx::Ulps;
use nalgebra::allocator::Allocator;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::DefaultAllocator;
use nalgebra::DimDiff;
use nalgebra::DimSub;
use nalgebra::SMatrix;
use nalgebra::SVector;
use rayon::prelude::*;

pub type Point2D = SVector<f64, 2>;
pub type Point3D = SVector<f64, 3>;
pub type PointND<const D: usize> = SVector<f64, D>;
pub type Matrix<const D: usize> = SMatrix<f64, D, D>;

/// Axis-aligned bounding box.
#[derive(Debug, Clone)]
pub struct BoundingBox<const D: usize> {
    pub p_min: PointND<D>,
    pub p_max: PointND<D>,
}

impl<const D: usize> BoundingBox<D> {
    /// The axis-aligned *minimum* bounding box.
    ///
    /// This is the smallest rectangle (rectangular cuboid in 3D) that both
    /// contains all given points and is aligned with the axises.
    ///
    /// Returns `None` iff the given iterator is empty.
    pub fn from_points<P>(points: P) -> Option<Self>
    where
        P: IntoParallelIterator<Item = PointND<D>>,
        P::Iter: IndexedParallelIterator,
    {
        let points = points.into_par_iter();
        if points.len() == 0 {
            return None;
        }
        let (p_min, p_max) = points
            .fold_with(
                (
                    PointND::<D>::from_element(std::f64::MAX),
                    PointND::<D>::from_element(std::f64::MIN),
                ),
                |(mut mins, mut maxs), vals| {
                    for ((min, max), val) in mins.iter_mut().zip(maxs.iter_mut()).zip(&vals) {
                        if *val < *min {
                            *min = *val;
                        }
                        if *max < *val {
                            *max = *val;
                        }
                    }
                    (mins, maxs)
                },
            )
            .reduce_with(|(mins_left, maxs_left), (mins_right, maxs_right)| {
                (
                    PointND::<D>::from_iterator(
                        mins_left
                            .into_iter()
                            .zip(&mins_right)
                            .map(|(left, right)| left.min(*right)),
                    ),
                    PointND::<D>::from_iterator(
                        maxs_left
                            .into_iter()
                            .zip(&maxs_right)
                            .map(|(left, right)| left.max(*right)),
                    ),
                )
            })
            .unwrap(); // fold_with yields at least one element.
        Some(Self { p_min, p_max })
    }

    pub fn center(&self) -> PointND<D> {
        (self.p_min + self.p_max) / 2.0
    }

    // region = bdim...b2b1b0 where bi are bits (0 or 1)
    // if bi is set (i.e. bi == 1) then the matching region has a i-th coordinates from center[i] to p_max[i]
    // otherwise, the matching region has a i-th coordinates from p_min[i] to center[i]
    pub fn sub_aabb(&self, region: u32) -> Self {
        assert!(
            region < 2u32.pow(D as u32),
            "Wrong region. Region should be composed of dim bits."
        );

        let center = self.center();

        let p_min =
            PointND::<D>::from_iterator(self.p_min.iter().zip(center.iter()).enumerate().map(
                |(i, (min, center))| {
                    if (region >> i) & 1 == 0 {
                        *min
                    } else {
                        *center
                    }
                },
            ));

        let p_max =
            PointND::<D>::from_iterator(center.iter().zip(self.p_max.iter()).enumerate().map(
                |(i, (center, max))| {
                    if (region >> i) & 1 == 0 {
                        *center
                    } else {
                        *max
                    }
                },
            ));

        Self { p_min, p_max }
    }

    pub fn contains(&self, point: &PointND<D>) -> bool {
        let eps = 10. * std::f64::EPSILON;
        self.p_min
            .iter()
            .zip(self.p_max.iter())
            .zip(point.iter())
            .all(|((min, max), point)| *point < *max + eps && *point > *min - eps)
    }

    pub fn distance_to_point(&self, point: &PointND<D>) -> f64 {
        if !self.contains(point) {
            let clamped = PointND::<D>::from_iterator(
                self.p_min
                    .iter()
                    .zip(self.p_max.iter())
                    .zip(point.iter())
                    .map(|((min, max), point)| {
                        if point > max {
                            *max
                        } else if point < min {
                            *min
                        } else {
                            *point
                        }
                    }),
            );
            clamped.norm()
        } else {
            let center = self.center();

            self.p_min
                .iter()
                .zip(self.p_max.iter())
                .zip(point.iter())
                .zip(&center)
                .map(|(((min, max), point), center)| {
                    if point > center {
                        (max - point).abs()
                    } else {
                        (min - point).abs()
                    }
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    pub fn region(&self, point: &PointND<D>) -> Option<u32> {
        if !self.contains(point) {
            return None;
        }

        let mut ret: u32 = 0;
        let center = self.center();
        for (i, (point, center)) in point.iter().zip(center.into_iter()).enumerate() {
            if point > center {
                ret |= 1 << i;
            }
        }
        Some(ret)
    }
}

/// Oriented bounding box.
///
/// Similar to a [BoundingBox] except it is not necessarily parallel to the
/// axes.
#[derive(Debug, Clone)]
pub(crate) struct OrientedBoundingBox<const D: usize> {
    aabb: BoundingBox<D>,
    aabb_to_obb: Matrix<D>,
    obb_to_aabb: Matrix<D>,
}

impl<const D: usize> OrientedBoundingBox<D> {
    /// The underlying axis-aligned bounding box.
    pub fn aabb(&self) -> &BoundingBox<D> {
        &self.aabb
    }

    /// Transforms a point with the transformation which maps the
    /// arbitrarily-oriented bounding box to the underlying axis-aligned
    /// bounding box.
    pub fn obb_to_aabb(&self, point: &PointND<D>) -> PointND<D> {
        self.obb_to_aabb * point
    }

    /// The arbitrarily-oriented *minimum* bounding box.
    ///
    /// The smallest box that contains all given points.
    pub fn from_points(points: &[PointND<D>]) -> Option<Self>
    where
        Const<D>: DimSub<Const<1>>,
        DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
            + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    {
        let mat = inertia_matrix(points);
        let vec = inertia_vector(mat);
        let aabb_to_obb = householder_reflection(&vec);
        let obb_to_aabb = aabb_to_obb.try_inverse().unwrap();
        let mapped = points.par_iter().map(|p| obb_to_aabb * p);
        let aabb = BoundingBox::from_points(mapped)?;

        Some(Self {
            aabb,
            aabb_to_obb,
            obb_to_aabb,
        })
    }

    /// Constructs a new Mbr that is a sub-Mbr of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_mbr(&self, region: u32) -> Self {
        Self {
            aabb: self.aabb.sub_aabb(region),
            aabb_to_obb: self.aabb_to_obb,
            obb_to_aabb: self.obb_to_aabb,
        }
    }

    /// Computes the distance between a point and the current bounding box.
    pub fn distance_to_point(&self, point: &PointND<D>) -> f64 {
        self.aabb.distance_to_point(&(self.obb_to_aabb * point))
    }

    /// Computes the center of the Mbr
    #[allow(unused)]
    pub fn center(&self) -> PointND<D> {
        self.aabb_to_obb * self.aabb.center()
    }

    /// Returns wheter or not the specified point is contained in the Mbr
    #[allow(unused)]
    pub fn contains(&self, point: &PointND<D>) -> bool {
        self.aabb.contains(&(self.obb_to_aabb * point))
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// A Mbr quadrant is defined as a quadrant of the associated Aabb.
    /// Returns `None` if the specified point is not contained in the Aabb.
    pub fn region(&self, point: &PointND<D>) -> Option<u32> {
        self.aabb.region(&(self.obb_to_aabb * point))
    }

    /// Returns the rotated min and max points of the Aabb.
    #[allow(unused)]
    pub fn minmax(&self) -> (PointND<D>, PointND<D>) {
        let min = self.aabb_to_obb * self.aabb.p_min;
        let max = self.aabb_to_obb * self.aabb.p_max;
        (min, max)
    }
}

fn inertia_matrix<const D: usize>(points: &[PointND<D>]) -> Matrix<D> {
    let centroid: PointND<D> = points.par_iter().sum();
    let centroid: PointND<D> = centroid / points.len() as f64;

    points
        .par_iter()
        .map(|point| {
            let offset = point - centroid;
            offset * offset.transpose()
        })
        .sum()
}

pub(crate) fn inertia_vector<const D: usize>(mat: Matrix<D>) -> PointND<D>
where
    Const<D>: DimSub<Const<1>>,
    DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
        + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
{
    let sym_eigen = mat.symmetric_eigen();
    let mut indices = (0..D).collect::<Vec<_>>();

    // sort indices in decreasing order
    indices.as_mut_slice().sort_unstable_by(|a, b| {
        sym_eigen.eigenvalues[*b]
            .partial_cmp(&sym_eigen.eigenvalues[*a])
            .unwrap()
    });

    sym_eigen.eigenvectors.column(indices[0]).into()
}

pub fn householder_reflection<const D: usize>(element: &PointND<D>) -> Matrix<D> {
    let e0 = canonical_vector(0);

    // if element is parallel to e0, then the reflection is not valid.
    // return identity (canonical basis)
    let norm = element.norm();
    if Ulps::default().eq(&(element / norm), &e0) {
        return Matrix::identity();
    }

    let sign = if element[0] > 0. { -1. } else { 1. };
    let w = element + sign * e0 * element.norm();
    let id = Matrix::identity();
    id - 2. * &w * w.transpose() / (w.transpose() * w)[0]
}

pub(crate) fn canonical_vector<const D: usize>(nth: usize) -> PointND<D> {
    let mut ret = PointND::from_element(0.);
    ret[nth] = 1.0;
    ret
}

pub(crate) fn center<const D: usize>(points: &[PointND<D>]) -> PointND<D> {
    assert!(!points.is_empty());
    let total = points.len() as f64;
    points.par_iter().sum::<PointND<D>>() / total
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use approx::assert_ulps_eq;
    use itertools::Itertools as _;
    use nalgebra::Matrix2;

    #[test]
    fn test_aabb_2d() {
        let points = [
            Point2D::from([1., 2.]),
            Point2D::from([0., 0.]),
            Point2D::from([3., 1.]),
            Point2D::from([5., 4.]),
            Point2D::from([4., 5.]),
        ];

        let aabb = BoundingBox::from_points(points).unwrap();

        assert_ulps_eq!(aabb.p_min, Point2D::from([0., 0.]));
        assert_ulps_eq!(aabb.p_max, Point2D::from([5., 5.]));
    }

    #[test]
    #[should_panic]
    fn test_aabb2d_invalid_input() {
        BoundingBox::<2>::from_points([]).unwrap();
    }

    #[test]
    fn test_inertia_matrix() {
        let points = [
            Point2D::from([3., 0.]),
            Point2D::from([0., 3.]),
            Point2D::from([6., -3.]),
        ];

        let mat = inertia_matrix(&points);
        let expected = Matrix2::new(18., -18., -18., 18.);

        assert_ulps_eq!(mat, expected);
    }

    #[test]
    fn test_inertia_vector_2d() {
        let points = [
            Point2D::from([3., 0.]),
            Point2D::from([0., 3.]),
            Point2D::from([6., -3.]),
        ];

        let mat = inertia_matrix(&points);
        let vec = inertia_vector(mat);
        let vec = Point3D::from([vec.x, vec.y, 0.]);
        let expected = Point3D::from([1., -1., 0.]);

        eprintln!("{}", vec);

        assert_ulps_eq!(expected.cross(&vec).norm(), 0.);
    }

    #[test]
    fn test_obb_center() {
        let points = [
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let obb = OrientedBoundingBox::from_points(&points).unwrap();

        let center = obb.center();
        assert_ulps_eq!(center, Point2D::from([3., 3.]))
    }

    #[test]
    fn test_obb_contains() {
        let points = [
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let obb = OrientedBoundingBox::from_points(&points).unwrap();

        assert!(!obb.contains(&Point2D::from([0., 0.])));
        assert!(obb.contains(&obb.center()));
        assert!(obb.contains(&Point2D::from([5., 4.])));

        let (min, max) = obb.minmax();

        assert!(obb.contains(&min));
        assert!(obb.contains(&max));
    }

    #[test]
    fn test_obb_quadrant() {
        let points = [
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let obb = OrientedBoundingBox::from_points(&points).unwrap();

        let none = obb.region(&Point2D::from([0., 0.]));
        let q1 = obb.region(&Point2D::from([1.2, 1.2]));
        let q2 = obb.region(&Point2D::from([5.8, 4.9]));
        let q3 = obb.region(&Point2D::from([0.2, 1.1]));
        let q4 = obb.region(&Point2D::from([5.1, 5.8]));

        println!("q1 = {:?}", q1);
        println!("q2 = {:?}", q2);
        println!("q3 = {:?}", q3);
        println!("q4 = {:?}", q4);

        assert!(none.is_none());
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_some());
        assert!(q4.is_some());
    }

    #[test]
    fn test_householder_reflexion() {
        let el = PointND::<6>::new_random();
        let mat = householder_reflection(&el);

        // check that columns are of norm 1
        for col in 0..6 {
            assert_ulps_eq!(mat.column(col).norm(), 1.);
        }

        // check that columns are orthogonal
        for col1 in 0..6 {
            for col2 in 0..6 {
                if col1 != col2 {
                    assert_relative_eq!(
                        mat.column(col1).dot(&mat.column(col2)),
                        0.,
                        epsilon = 1e-14,
                    );
                }
            }
        }

        // check that first column is parallel to el
        let unit_el = el / el.norm();
        let fst_col = mat.column(0).clone_owned();
        assert_relative_eq!(unit_el * unit_el.dot(&fst_col), fst_col, epsilon = 1e-14);
    }

    #[test]
    fn test_aabb_3d() {
        let points = [
            Point3D::from([1., 2., 0.]),
            Point3D::from([0., 0., 5.]),
            Point3D::from([3., 1., 1.]),
            Point3D::from([5., 4., -2.]),
            Point3D::from([4., 5., 3.]),
        ];

        let aabb = BoundingBox::from_points(points).unwrap();

        assert_ulps_eq!(aabb.p_min, Point3D::from([0., 0., -2.]));
        assert_ulps_eq!(aabb.p_max, Point3D::from([5., 5., 5.]));
    }

    #[test]
    fn test_inertia_vector_3d() {
        let points = [
            Point3D::from([3., 0., 0.]),
            Point3D::from([0., 3., 3.]),
            Point3D::from([6., -3., -3.]),
        ];

        let mat = inertia_matrix(&points);
        let vec = inertia_vector(mat);
        let expected = Point3D::from([1., -1., -1.]);

        assert_relative_eq!(expected.cross(&vec).norm(), 0., epsilon = 1e-15);
    }

    #[test]
    fn test_obb_octant() {
        let points = [
            Point3D::from([0., 1., 0.]),
            Point3D::from([1., 0., 0.]),
            Point3D::from([5., 6., 0.]),
            Point3D::from([6., 5., 0.]),
            Point3D::from([0., 1., 1.]),
            Point3D::from([1., 0., 1.]),
            Point3D::from([5., 6., 1.]),
            Point3D::from([6., 5., 1.]),
        ];

        let obb = OrientedBoundingBox::from_points(&points).unwrap();

        let none = obb.region(&Point3D::from([0., 0., 0.]));
        let octants = [
            obb.region(&Point3D::from([1.2, 1.2, 0.3])),
            obb.region(&Point3D::from([5.8, 4.9, 0.3])),
            obb.region(&Point3D::from([0.2, 1.1, 0.3])),
            obb.region(&Point3D::from([5.1, 5.8, 0.3])),
            obb.region(&Point3D::from([1.2, 1.2, 0.7])),
            obb.region(&Point3D::from([5.8, 4.9, 0.7])),
            obb.region(&Point3D::from([0.2, 1.1, 0.7])),
            obb.region(&Point3D::from([5.1, 5.8, 0.7])),
        ];
        assert_eq!(none, None);
        assert!(octants.iter().all(|o| o.is_some()));
        // we cannot test if each octant is the right one because the orientation of the base
        // change is unspecified
        assert_eq!(8, octants.iter().unique().count());
    }
}
