use crate::point_nd::{Point2D, PointND};

use cust::DeviceCopy;
use itertools::Itertools;
use itertools::MinMaxResult::{MinMax, NoElements, OneElement};

#[derive(Clone, Copy, DeviceCopy, PartialEq, PartialOrd)]
pub struct BoundingBox<const D: usize> {
    p_min: PointND<D>,
    p_max: PointND<D>,
}

pub type BBox2D = BoundingBox<2>;
impl BBox2D {
    pub fn from_points(points: &[Point2D]) -> Option<Self> {
        match points.iter().minmax() {
            NoElements => None,
            OneElement(p) => Some(Self {
                p_min: *p,
                p_max: *p,
            }),
            MinMax(p_min, p_max) => Some(Self {
                p_min: *p_min,
                p_max: *p_max,
            }),
        }
    }

    pub fn _p_min(&self) -> &Point2D {
        &self.p_min
    }

    pub fn _p_max(&self) -> &Point2D {
        &self.p_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bb_from_points() {
        let points = vec![
            Point2D::new(-1, 2),
            Point2D::new(0, 3),
            Point2D::new(1, 1),
            Point2D::new(3, 2),
        ];
        let bb = BBox2D::from_points(&points).unwrap();

        assert_eq!(bb._p_min(), &Point2D::new(-1, 2));
        assert_eq!(bb._p_max(), &Point2D::new(3, 2));
    }
}
