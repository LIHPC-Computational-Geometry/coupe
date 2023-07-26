use crate::{bounding_box::BoundingBox, point_nd::PointND};

use std::{
    cmp,
    ops::{Add, AddAssign},
};

pub trait RcbWeight: Clone + Default + Add + AddAssign {}
impl RcbWeight for i32 {}

struct Grid<const D: usize, W: RcbWeight> {
    bounding_box: BoundingBox<D>,
    ncols: usize,
    nrows: usize,
    cells: Vec<Cell<W>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell<W: RcbWeight> {
    pub offset: usize,
    pub npoints: usize,
    pub weight_sum: W,
}

impl<const D: usize, W: RcbWeight + std::fmt::Debug> Grid<D, W> {
    pub fn from_points(
        bounding_box: BoundingBox<D>,
        ncols: usize,
        nrows: usize,
        points: &[PointND<D>],
        weights: &[W],
    ) -> Self {
        let mut cells: Vec<Cell<W>> = vec![Cell::default(); ncols * nrows];

        for (p, w) in points.iter().zip(weights.iter()) {
            let grid_x = cmp::min(cmp::max(p[0], 0), ncols as i32 - 1);
            let grid_y = cmp::min(cmp::max(p[1], 0), nrows as i32 - 1);
            let cell_idx = grid_x + ncols as i32 * grid_y;
            cells[cell_idx as usize].npoints += 1;
            cells[cell_idx as usize].weight_sum += w.clone();
        }

        let mut offset = 0;
        for c in cells.iter_mut() {
            c.offset = offset;
            offset += c.npoints;
        }

        Self {
            bounding_box,
            ncols,
            nrows,
            cells,
        }
    }

    pub fn cells(&self) -> &[Cell<W>] {
        &self.cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BBox2D, Point2D};

    #[test]
    fn build_from_points() {
        let points = vec![
            Point2D::new(-8, -2),
            Point2D::new(-9, 1),
            Point2D::new(0, -8),
            Point2D::new(-9, 5),
            Point2D::new(-2, -6),
            Point2D::new(3, -2),
            Point2D::new(6, 10),
            Point2D::new(0, 1),
            Point2D::new(-8, 3),
            Point2D::new(0, 7),
            Point2D::new(0, 9),
            Point2D::new(1, 5),
            Point2D::new(10, 0),
            Point2D::new(-8, 7),
            Point2D::new(0, -7),
            Point2D::new(5, -4),
        ];
        let weights = [1, 1, 1, 1, 2, 2, 2, 1, 4, 4, 2, 1, 8, 4, 2, 1];
        let ncols = 3;
        let nrows = 2;
        let bbox = BBox2D::from_points(&points);

        let grid: Grid<2, i32> = Grid::from_points(bbox.unwrap(), ncols, nrows, &points, &weights);

        assert_eq!(grid.cells[0].offset, 0);
        assert_eq!(grid.cells[0].npoints, 4);
        assert_eq!(grid.cells[0].weight_sum, 6);

        assert_eq!(grid.cells[1].offset, 4);
        assert_eq!(grid.cells[1].npoints, 0);
        assert_eq!(grid.cells[1].weight_sum, 0);

        assert_eq!(grid.cells[2].offset, 4);
        assert_eq!(grid.cells[2].npoints, 3);
        assert_eq!(grid.cells[2].weight_sum, 11);

        assert_eq!(grid.cells[3].offset, 7);
        assert_eq!(grid.cells[3].npoints, 7);
        assert_eq!(grid.cells[3].weight_sum, 17);

        assert_eq!(grid.cells[4].offset, 14);
        assert_eq!(grid.cells[4].npoints, 1);
        assert_eq!(grid.cells[4].weight_sum, 1);

        assert_eq!(grid.cells[5].offset, 15);
        assert_eq!(grid.cells[5].npoints, 1);
        assert_eq!(grid.cells[5].weight_sum, 2);
    }
}
