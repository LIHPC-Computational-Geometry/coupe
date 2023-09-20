pub(crate) use crate::point_nd::{Point2D, PointND};

// use cust::DeviceCopy;
use itertools::Itertools;
use itertools::MinMaxResult::{MinMax, NoElements, OneElement};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BoundingBox<const D: usize> {
    p_min: PointND<D>,
    p_max: PointND<D>,
}

impl<const D: usize> BoundingBox<D> {
    pub fn p_min(&self) -> &PointND<D> {
        &self.p_min
    }

    pub fn p_max(&self) -> &PointND<D> {
        &self.p_max
    }
}

pub type BBox2D = BoundingBox<2>;
impl BBox2D {
    pub fn from_coord(p_min: Point2D, p_max: Point2D) -> Self {
        Self { p_min, p_max }
    }

    pub fn from_points(points: &[Point2D]) -> Option<Self> {
        let x_minmax = match points.iter().map(|p| p.x()).minmax() {
            NoElements => None,
            OneElement(x) => Some((x, x)),
            MinMax(x_min, x_max) => Some((x_min, x_max)),
        };
        let y_minmax = match points.iter().map(|p| p.y()).minmax() {
            NoElements => None,
            OneElement(y) => Some((y, y)),
            MinMax(y_min, y_max) => Some((y_min, y_max)),
        };

        let (x_min, x_max) = x_minmax.unwrap();
        let (y_min, y_max) = y_minmax.unwrap();

        Some(Self {
            p_min: Point2D::new(x_min, y_min),
            p_max: Point2D::new(x_max, y_max),
        })
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

        assert_eq!(bb.p_min(), &Point2D::new(-1, 1));
        assert_eq!(bb.p_max(), &Point2D::new(3, 3));
    }
}
