use cust::memory::DeviceCopy;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy, PartialEq, PartialOrd, Debug)]
pub struct PointND<const D: usize> {
    coords: [i32; D],
}

pub type Point2D = PointND<2>;
impl Point2D {
    pub fn new(x: i32, y: i32) -> Self {
        Self { coords: [x, y] }
    }

    pub fn x(&self) -> i32 {
        self.coords[0]
    }

    pub fn y(&self) -> i32 {
        self.coords[1]
    }
}

impl<const D: usize> std::ops::Index<usize> for PointND<D> {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < D, "PointND only has {D} dimensions");
        &self.coords[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_point2d() {
        let p = Point2D::new(-1, 2);
        assert_eq!(p[0], p.x());
        assert_eq!(p[1], p.y());
    }
}
