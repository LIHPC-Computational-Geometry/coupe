//! An implementation of the Z-order space filling curve.
//! It aims to reorder a set of points with spacial hashing.
//!
//! First, a Minimal Bounding rectangle is constructed from a given set of points.
//! The Mbr is then recursively split following a quadtree refinement scheme until each cell
//! contains at most one point. Then, a spatial binary hash is determined for each point as follows.
//!
//! The hash of a point is initially 0 and for each new split required to isolate it,
//! the new hash is computed by `new_hash = previous_hash << 2 + b` where `0 <= b <= 3`.
//! `b` is chosen by looking up in which quadrant of the current Mbr the point is. The mapping is defined as follows:
//!
//!   - `BottomLeft => 0b00`
//!   - `BottomRight => 0b01`
//!   - `TopLeft => 0b10`
//!   - `TopRight => 0b11`
//!
//! Finally, the points are reordered according to the order of their hash.

use geometry::{Mbr2D, Point2D, Quadrant};

/// A quadtree construct to help with computing spatial hashes. It is a rectangle that can be recursively split
/// to yield sub quadtrees. Here, the ZCurveQuadtree is defined as a bounding box of a set of points.
///
/// Note that it may not be axis-aligned.
pub struct ZCurveQuadtree {
    mbr: Mbr2D,
    points: Vec<Point2D>,
}

impl ZCurveQuadtree {
    /// Constructs a new `ZCurveQuadtree` from a set of points.
    pub fn from_points(points: Vec<Point2D>) -> Self {
        Self {
            mbr: Mbr2D::from_points(points.iter()),
            points,
        }
    }

    /// Compute Z hashes of the points contained in the quadtree and reorders them.
    pub fn reorder(self) -> Vec<Point2D> {
        let mut with_hashes = self.compute_hashes();
        with_hashes
            .as_mut_slice()
            .sort_unstable_by_key(|(_, hash)| *hash);

        let (points, _): (Vec<_>, Vec<_>) = with_hashes.into_iter().unzip();
        points
    }

    fn from_points_with_mbr(points: Vec<Point2D>, mbr: Mbr2D) -> Self {
        Self { mbr, points }
    }

    fn compute_hashes(&self) -> Vec<(Point2D, u32)> {
        self.compute_hashes_impl(0)
    }

    fn compute_hashes_impl(&self, current_hash: u32) -> Vec<(Point2D, u32)> {
        use self::Quadrant::*;

        // Construct a mapping from each quadrant of the current mbr
        // to the set of points that are contained in it.
        let points_map = self
            .points
            .iter()
            .map(|point| (self.mbr.quadrant(point).unwrap(), point))
            .collect::<Vec<_>>();

        // Filter points based on which quadrant they are in
        let bottom_lefts = points_map
            .iter()
            .filter(|(q, _)| *q == BottomLeft)
            .collect::<Vec<_>>();
        let bottom_rights = points_map
            .iter()
            .filter(|(q, _)| *q == BottomRight)
            .collect::<Vec<_>>();
        let top_lefts = points_map
            .iter()
            .filter(|(q, _)| *q == TopLeft)
            .collect::<Vec<_>>();
        let top_rights = points_map
            .iter()
            .filter(|(q, _)| *q == TopRight)
            .collect::<Vec<_>>();

        // In each quadrant, three cases are possible:
        //   - Only one point is contained. The current hash is updated and is final.
        //   - The quadrant is empty. Nothing is done.
        //   - The quadrant contains at least to points. The current hash is updated, the quadrant is split again
        //     and the algorithm is called on each new quadrant.
        let bottom_lefts = if bottom_lefts.len() > 1 {
            let mbr = self.mbr.sub_mbr(BottomLeft);
            Self::from_points_with_mbr(
                bottom_lefts
                    .iter()
                    .map(|(_, point)| **point)
                    .collect::<Vec<_>>(),
                mbr,
            ).compute_hashes_impl(current_hash << 2)
        } else {
            bottom_lefts
                .iter()
                .map(|(_, point)| (**point, current_hash << 2))
                .collect()
        };

        let bottom_rights = if bottom_rights.len() > 1 {
            let mbr = self.mbr.sub_mbr(BottomRight);
            Self::from_points_with_mbr(
                bottom_rights
                    .iter()
                    .map(|(_, point)| **point)
                    .collect::<Vec<_>>(),
                mbr,
            ).compute_hashes_impl(current_hash << (2 + 0b01))
        } else {
            bottom_rights
                .iter()
                .map(|(_, point)| (**point, current_hash << (2 + 0b01)))
                .collect()
        };

        let top_lefts = if top_lefts.len() > 1 {
            let mbr = self.mbr.sub_mbr(TopLeft);
            Self::from_points_with_mbr(
                top_lefts
                    .iter()
                    .map(|(_, point)| **point)
                    .collect::<Vec<_>>(),
                mbr,
            ).compute_hashes_impl(current_hash << (2 + 0b10))
        } else {
            top_lefts
                .iter()
                .map(|(_, point)| (**point, current_hash << (2 + 0b10)))
                .collect()
        };

        let top_rights = if top_rights.len() > 1 {
            let mbr = self.mbr.sub_mbr(TopRight);
            Self::from_points_with_mbr(
                top_rights
                    .iter()
                    .map(|(_, point)| **point)
                    .collect::<Vec<_>>(),
                mbr,
            ).compute_hashes_impl(current_hash << (2 + 0b11))
        } else {
            top_rights
                .iter()
                .map(|(_, point)| (**point, current_hash << (2 + 0b11)))
                .collect()
        };

        // Stick back all the points together
        bottom_lefts
            .into_iter()
            .chain(bottom_rights)
            .chain(top_lefts)
            .chain(top_rights)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reorder() {
        let points = vec![
            Point2D::new(0., 0.),
            Point2D::new(20., 10.),
            Point2D::new(0., 10.),
            Point2D::new(20., 0.),
            Point2D::new(14., 7.),
            Point2D::new(4., 7.),
            Point2D::new(14., 2.),
            Point2D::new(4., 2.),
        ];

        let qt = ZCurveQuadtree::from_points(points);
        let reordered = qt.reorder();

        assert_ulps_eq!(reordered[0], Point2D::new(0., 0.));
        assert_ulps_eq!(reordered[1], Point2D::new(4., 2.));
        assert_ulps_eq!(reordered[2], Point2D::new(14., 2.));
        assert_ulps_eq!(reordered[3], Point2D::new(20., 0.));
        assert_ulps_eq!(reordered[4], Point2D::new(4., 7.));
        assert_ulps_eq!(reordered[5], Point2D::new(0., 10.));
        assert_ulps_eq!(reordered[6], Point2D::new(14., 7.));
        assert_ulps_eq!(reordered[7], Point2D::new(20., 10.));
    }
}
