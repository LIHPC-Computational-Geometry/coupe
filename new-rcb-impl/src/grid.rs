use crate::{
    bounding_box::{BBox2D, BoundingBox},
    point_nd::Point2D,
};

use std::{
    cmp,
    ops::{Add, AddAssign},
};

/// Marker trait for a point's weight.
pub trait RcbWeight: Clone + Copy + Default + Add + AddAssign {}

impl RcbWeight for i32 {}

/// Cartesian grid laid over a random set of points.
#[derive(Debug)]
struct Grid<const D: usize, W: RcbWeight + Copy> {
    /// Imaginary bounding box of the grid that encapsulates all points within it.
    bounding_box: BoundingBox<D>,
    /// Number of columns of the grid.
    ncols: usize,
    /// Number of rows of the grid.
    nrows: usize,
    /// List of cells composing the grid.
    cells: Vec<Cell<W>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell<W: RcbWeight> {
    /// Offset of the cell in the linear arrays of reordered points and weights.
    pub offset: usize,
    /// Number of points in the cell.
    pub npoints: usize,
    /// Total weight of the points in the cell.
    pub weight_sum: W,
}

impl<const D: usize, W: RcbWeight + std::fmt::Debug> Grid<D, W> {
    pub fn cells(&self) -> &[Cell<W>] {
        &self.cells
    }

    pub fn mut_cells(&mut self) -> &mut [Cell<W>] {
        &mut self.cells
    }
}

impl<W: RcbWeight + std::fmt::Debug> Grid<2, W> {
    /// Obtain the index of the cell in which a given point is located.
    fn cell_idx(point: &Point2D, ncols: usize, nrows: usize) -> usize {
        // Safety: conversion to `usize` of the maximum is safe because it is always at least 0
        let grid_x = cmp::min(cmp::max(point.x(), 0) as usize, ncols - 1);
        let grid_y = cmp::min(cmp::max(point.y(), 0) as usize, nrows - 1);
        grid_x + ncols * grid_y
    }

    /// Creates a `Grid` from a given set of points and their weights.
    pub fn from_points(
        bounding_box: BBox2D,
        ncols: usize,
        nrows: usize,
        (points, weights): (&[Point2D], &[W]),
    ) -> Self {
        let mut cells: Vec<Cell<W>> = vec![Cell::default(); ncols * nrows];

        for (p, w) in points.iter().zip(weights.iter()) {
            let idx = Grid::<2, W>::cell_idx(p, ncols, nrows);
            cells[idx].npoints += 1;
            cells[idx].weight_sum += *w;
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

    pub fn reordered_points_and_weights(
        &self,
        (points, weights): (&[Point2D], &[W]),
    ) -> (Vec<Point2D>, Vec<W>) {
        let mut new_points = vec![Point2D::default(); points.len()];
        let mut new_weights = vec![W::default(); weights.len()];

        let mut npoints_in_cell: Vec<usize> = self.cells.iter().map(|c| c.npoints).collect();
        for (p, w) in points.iter().zip(weights.iter()) {
            let idx = Grid::<2, W>::cell_idx(p, self.ncols, self.nrows);
            let offset = self.cells[idx].offset;
            npoints_in_cell[idx] -= 1;
            new_points[offset + npoints_in_cell[idx]] = *p;
            new_weights[offset + npoints_in_cell[idx]] = *w;
        }

        (new_points, new_weights)
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
        let bbox = BBox2D::from_points(&points).unwrap();
        assert_eq!(bbox.p_min(), &Point2D::new(-9, -8));
        assert_eq!(bbox.p_max(), &Point2D::new(10, 10));

        let grid = Grid::<2, i32>::from_points(bbox, ncols, nrows, (&points, &weights));
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

    #[test]
    fn reorder_points_and_weights() {
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

        let grid = Grid::<2, i32>::from_points(bbox.unwrap(), ncols, nrows, (&points, &weights));

        let (new_points, _) = grid.reordered_points_and_weights((&points, &weights));

        for (i, p) in new_points.iter().enumerate() {
            let idx = Grid::<2, i32>::cell_idx(p, ncols, nrows);
            let cell = grid.cells()[idx];

            // Assert that the reordered points are in the correct grid cell
            assert!(i >= cell.offset && i < cell.offset + cell.npoints);
        }
    }
}
