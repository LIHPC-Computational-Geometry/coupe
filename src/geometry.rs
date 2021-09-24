//! A few useful geometric types

use approx::Ulps;
use itertools::Itertools;
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

#[derive(Debug, Clone)]
pub(crate) struct Aabb<const D: usize> {
    p_min: PointND<D>,
    p_max: PointND<D>,
}

impl<const D: usize> Aabb<D> {
    pub fn p_min(&self) -> &PointND<D> {
        &self.p_min
    }

    pub fn p_max(&self) -> &PointND<D> {
        &self.p_max
    }

    pub fn from_points(points: &[PointND<D>]) -> Self {
        if points.len() < 2 {
            panic!("Cannot create an Aabb from less than 2 points.");
        }

        let (min, max) = points
            .par_iter()
            .fold_with(
                (
                    PointND::<D>::from_element(std::f64::MAX),
                    PointND::<D>::from_element(std::f64::MIN),
                ),
                |(mut mins, mut maxs), vals| {
                    for ((min, max), val) in mins.iter_mut().zip(maxs.iter_mut()).zip(vals.iter()) {
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
                            .zip(mins_right.into_iter())
                            .map(|(left, right)| left.min(*right)),
                    ),
                    PointND::<D>::from_iterator(
                        maxs_left
                            .into_iter()
                            .zip(maxs_right.into_iter())
                            .map(|(left, right)| left.max(*right)),
                    ),
                )
            })
            .unwrap();

        Self {
            p_min: min,
            p_max: max,
        }
    }

    fn center(&self) -> PointND<D> {
        PointND::from_iterator(
            self.p_min
                .iter()
                .zip(self.p_max.iter())
                .map(|(min, max)| 0.5 * (min + max)),
        )
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

    pub fn aspect_ratio(&self) -> f64 {
        use itertools::MinMaxResult::*;

        let (min, max) = match self
            .p_min()
            .iter()
            .zip(self.p_max().iter())
            .map(|(min, max)| (max - min).abs())
            .minmax()
        {
            MinMax(min, max) => (min, max),
            _ => unimplemented!(),
        };

        // TODO: What if min == 0.0 ?
        max / min
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
                .zip(center.into_iter())
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

#[derive(Debug, Clone)]
pub(crate) struct Mbr<const D: usize> {
    aabb: Aabb<D>,
    aabb_to_mbr: Matrix<D>,
    mbr_to_aabb: Matrix<D>,
}

impl<const D: usize> Mbr<D> {
    pub fn aabb(&self) -> &Aabb<D> {
        &self.aabb
    }

    // Transform a point with the transformation which maps the Mbr to the underlying Aabb
    pub fn mbr_to_aabb(&self, point: &PointND<D>) -> PointND<D> {
        self.mbr_to_aabb * point
    }

    /// Constructs a new `Mbr` from a slice of `PointND`.
    ///
    /// The resulting `Mbr` is the smallest Aabb that contains every points of the slice.
    pub fn from_points(points: &[PointND<D>]) -> Self
    where
        Const<D>: DimSub<Const<1>>,
        DefaultAllocator: Allocator<f64, Const<D>, Const<D>, Buffer = ArrayStorage<f64, D, D>>
            + Allocator<f64, DimDiff<Const<D>, Const<1>>>,
    {
        let weights = points.par_iter().map(|_| 1.).collect::<Vec<_>>();
        let mat = inertia_matrix(&weights, points);
        let vec = inertia_vector(mat);
        let base_change = householder_reflection(&vec);
        let mbr_to_aabb = base_change.try_inverse().unwrap();
        let mapped = points
            .par_iter()
            .map(|p| mbr_to_aabb * p)
            .collect::<Vec<_>>();
        let aabb = Aabb::from_points(&mapped);

        Self {
            aabb,
            aabb_to_mbr: base_change,
            mbr_to_aabb,
        }
    }

    /// Constructs a new Mbr that is a sub-Mbr of the current one.
    /// More precisely, it bounds exactly the specified quadrant.
    pub fn sub_mbr(&self, region: u32) -> Self {
        Self {
            aabb: self.aabb.sub_aabb(region),
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
    pub fn distance_to_point(&self, point: &PointND<D>) -> f64 {
        self.aabb.distance_to_point(&(self.mbr_to_aabb * point))
    }

    /// Computes the center of the Mbr
    #[allow(unused)]
    pub fn center(&self) -> PointND<D> {
        self.aabb_to_mbr * self.aabb.center()
    }

    /// Returns wheter or not the specified point is contained in the Mbr
    #[allow(unused)]
    pub fn contains(&self, point: &PointND<D>) -> bool {
        self.aabb.contains(&(self.mbr_to_aabb * point))
    }

    /// Returns the quadrant of the Aabb in which the specified point is.
    /// A Mbr quadrant is defined as a quadrant of the associated Aabb.
    /// Returns `None` if the specified point is not contained in the Aabb.
    pub fn region(&self, point: &PointND<D>) -> Option<u32> {
        self.aabb.region(&(self.mbr_to_aabb * point))
    }

    /// Returns the rotated min and max points of the Aabb.
    #[allow(unused)]
    pub fn minmax(&self) -> (PointND<D>, PointND<D>) {
        let min = self.aabb_to_mbr * self.aabb.p_min;
        let max = self.aabb_to_mbr * self.aabb.p_max;
        (min, max)
    }
}

pub(crate) fn inertia_matrix<const D: usize>(weights: &[f64], points: &[PointND<D>]) -> Matrix<D> {
    let total_weight = weights.par_iter().sum::<f64>();

    let centroid: PointND<D> = weights
        .par_iter()
        .zip(points)
        .fold_with(PointND::<D>::from_element(0.), |acc, (w, p)| {
            acc + p.map(|e| e * w)
        })
        .reduce_with(|a, b| a + b)
        .unwrap()
        / total_weight;

    weights
        .par_iter()
        .zip(points)
        .fold_with(Matrix::from_element(0.), |acc, (w, p)| {
            acc + ((p - centroid) * (p - centroid).transpose()).map(|e| e * w)
        })
        .reduce_with(|a, b| a + b)
        .unwrap()
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
    use approx::*;
    use nalgebra::Matrix2;

    #[test]
    fn test_aabb_2d() {
        let points = vec![
            Point2D::from([1., 2.]),
            Point2D::from([0., 0.]),
            Point2D::from([3., 1.]),
            Point2D::from([5., 4.]),
            Point2D::from([4., 5.]),
        ];

        let aabb = Aabb::from_points(&points);
        let aspect_ratio = aabb.aspect_ratio();

        assert_ulps_eq!(aabb.p_min, Point2D::from([0., 0.]));
        assert_ulps_eq!(aabb.p_max, Point2D::from([5., 5.]));
        assert_ulps_eq!(aspect_ratio, 1.);
    }

    #[test]
    #[should_panic]
    fn test_aabb2d_invalid_input_1() {
        let points: Vec<Point2D> = vec![];
        let _aabb = Aabb::from_points(&points);
    }

    #[test]
    #[should_panic]
    fn test_aabb2d_invalid_input_2() {
        let points = vec![Point2D::from([5., -9.2])];
        let _aabb = Aabb::from_points(&points);
    }

    //#[test] // TODO
    fn test_mbr_2d() {
        let points = vec![
            Point2D::from([5., 3.]),
            Point2D::from([0., 0.]),
            Point2D::from([1., -1.]),
            Point2D::from([4., 4.]),
        ];

        let mbr = Mbr::from_points(&points);
        let aspect_ratio = mbr.aspect_ratio();

        assert_relative_eq!(aspect_ratio, 4.);
    }

    #[test]
    fn test_inertia_matrix() {
        let points = vec![
            Point2D::from([3., 0.]),
            Point2D::from([0., 3.]),
            Point2D::from([6., -3.]),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix(&weights, &points);
        let expected = Matrix2::new(18., -18., -18., 18.);

        assert_ulps_eq!(mat, expected);
    }

    #[test]
    fn test_inertia_vector_2d() {
        let points = vec![
            Point2D::from([3., 0.]),
            Point2D::from([0., 3.]),
            Point2D::from([6., -3.]),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix(&weights, &points);
        let vec = inertia_vector(mat);
        let vec = Point3D::from([vec.x, vec.y, 0.]);
        let expected = Point3D::from([1., -1., 0.]);

        eprintln!("{}", vec);

        assert_ulps_eq!(expected.cross(&vec).norm(), 0.);
    }

    //#[test] // TODO
    fn test_mbr_distance_to_point_2d() {
        let points = vec![
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let mbr = Mbr::from_points(&points);

        let test_points = vec![
            Point2D::from([2., 2.]),
            Point2D::from([0., 0.]),
            Point2D::from([5., 7.]),
        ];

        let distances: Vec<_> = test_points
            .iter()
            .map(|p| mbr.distance_to_point(p))
            .collect();

        assert_relative_eq!(distances[0], 0.5);
        assert_relative_eq!(distances[1], 2_f64.sqrt() / 2.);
        assert_relative_eq!(distances[2], 1.);
    }

    #[test]
    fn test_mbr_center() {
        let points = vec![
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let mbr = Mbr::from_points(&points);

        let center = mbr.center();
        assert_ulps_eq!(center, Point2D::from([3., 3.]))
    }

    #[test]
    fn test_mbr_contains() {
        let points = vec![
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let mbr = Mbr::from_points(&points);

        assert!(!mbr.contains(&Point2D::from([0., 0.])));
        assert!(mbr.contains(&mbr.center()));
        assert!(mbr.contains(&Point2D::from([5., 4.])));

        let (min, max) = mbr.minmax();

        assert!(mbr.contains(&min));
        assert!(mbr.contains(&max));
    }

    #[test]
    fn test_mbr_quadrant() {
        let points = vec![
            Point2D::from([0., 1.]),
            Point2D::from([1., 0.]),
            Point2D::from([5., 6.]),
            Point2D::from([6., 5.]),
        ];

        let mbr = Mbr::from_points(&points);

        let none = mbr.region(&Point2D::from([0., 0.]));
        let q1 = mbr.region(&Point2D::from([1.2, 1.2]));
        let q2 = mbr.region(&Point2D::from([5.8, 4.9]));
        let q3 = mbr.region(&Point2D::from([0.2, 1.1]));
        let q4 = mbr.region(&Point2D::from([5.1, 5.8]));

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
                    assert_relative_eq!(mat.column(col1).dot(&mat.column(col2)), 0.);
                }
            }
        }

        // check that first column is parallel to el
        let unit_el = el / el.norm();
        let fst_col = mat.column(0).clone_owned();
        assert_relative_eq!(unit_el * unit_el.dot(&fst_col), fst_col);
    }

    #[test]
    fn test_aabb_3d() {
        let points = vec![
            Point3D::from([1., 2., 0.]),
            Point3D::from([0., 0., 5.]),
            Point3D::from([3., 1., 1.]),
            Point3D::from([5., 4., -2.]),
            Point3D::from([4., 5., 3.]),
        ];

        let aabb = Aabb::from_points(&points);
        let aspect_ratio = aabb.aspect_ratio();

        assert_ulps_eq!(aabb.p_min, Point3D::from([0., 0., -2.]));
        assert_ulps_eq!(aabb.p_max, Point3D::from([5., 5., 5.]));
        assert_ulps_eq!(aspect_ratio, 7. / 5.);
    }

    //#[test] // TODO
    fn test_mbr_3d() {
        let points = vec![
            Point3D::from([5., 3., 0.]),
            Point3D::from([0., 0., 0.]),
            Point3D::from([1., -1., 0.]),
            Point3D::from([4., 4., 0.]),
            Point3D::from([5., 3., 2.]),
            Point3D::from([0., 0., 2.]),
            Point3D::from([1., -1., 2.]),
            Point3D::from([4., 4., 2.]),
        ];

        let mbr = Mbr::from_points(&points);
        println!("mbr = {:?}", mbr);
        let aspect_ratio = mbr.aspect_ratio();

        assert_relative_eq!(aspect_ratio, 4.);
    }

    //#[test] // TODO
    fn test_inertia_vector_3d() {
        let points = vec![
            Point3D::from([3., 0., 0.]),
            Point3D::from([0., 3., 3.]),
            Point3D::from([6., -3., -3.]),
        ];

        let weights = vec![1.; 3];

        let mat = inertia_matrix(&weights, &points);
        let vec = inertia_vector(mat);
        let expected = Point3D::from([1., -1., -1.]);

        eprintln!("{}", vec);

        assert_relative_eq!(expected.cross(&vec).norm(), 0.);
    }

    //#[test] // TODO
    fn test_mbr_distance_to_point_3d() {
        let points = vec![
            Point3D::from([0., 1., 0.]),
            Point3D::from([1., 0., 0.]),
            Point3D::from([5., 6., 0.]),
            Point3D::from([6., 5., 0.]),
            Point3D::from([0., 1., 4.]),
            Point3D::from([1., 0., 4.]),
            Point3D::from([5., 6., 4.]),
            Point3D::from([6., 5., 4.]),
        ];

        let mbr = Mbr::from_points(&points);

        let test_points = vec![
            Point3D::from([2., 2., 1.]),
            Point3D::from([0., 0., 1.]),
            Point3D::from([5., 7., 1.]),
            Point3D::from([2., 2., 3.]),
            Point3D::from([0., 0., 3.]),
            Point3D::from([5., 7., 3.]),
        ];

        let distances: Vec<_> = test_points
            .iter()
            .map(|p| mbr.distance_to_point(p))
            .collect();

        assert_relative_eq!(distances[0], 0.5);
        assert_relative_eq!(distances[1], 2_f64.sqrt() / 2.);
        assert_relative_eq!(distances[2], 1.);
        assert_relative_eq!(distances[3], 0.5);
        assert_relative_eq!(distances[4], 2_f64.sqrt() / 2.);
        assert_relative_eq!(distances[5], 1.);
    }

    #[test]
    fn test_mbr_octant() {
        let points = vec![
            Point3D::from([0., 1., 0.]),
            Point3D::from([1., 0., 0.]),
            Point3D::from([5., 6., 0.]),
            Point3D::from([6., 5., 0.]),
            Point3D::from([0., 1., 1.]),
            Point3D::from([1., 0., 1.]),
            Point3D::from([5., 6., 1.]),
            Point3D::from([6., 5., 1.]),
        ];

        let mbr = Mbr::from_points(&points);
        eprintln!("mbr = {:#?}", mbr);

        let none = mbr.region(&Point3D::from([0., 0., 0.]));
        let octants = vec![
            mbr.region(&Point3D::from([1.2, 1.2, 0.3])),
            mbr.region(&Point3D::from([5.8, 4.9, 0.3])),
            mbr.region(&Point3D::from([0.2, 1.1, 0.3])),
            mbr.region(&Point3D::from([5.1, 5.8, 0.3])),
            mbr.region(&Point3D::from([1.2, 1.2, 0.7])),
            mbr.region(&Point3D::from([5.8, 4.9, 0.7])),
            mbr.region(&Point3D::from([0.2, 1.1, 0.7])),
            mbr.region(&Point3D::from([5.1, 5.8, 0.7])),
        ];
        assert_eq!(none, None);
        assert!(octants.iter().all(|o| o.is_some()));
        // we cannot test if each octant is the right one because the orientation of the base
        // change is unspecified
        assert_eq!(8, octants.iter().unique().count());
    }
}
