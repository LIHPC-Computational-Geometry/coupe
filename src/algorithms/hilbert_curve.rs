//! An implementation of the Hilbert space filling curve.
//!
//! With this technique, a set of 2D points (p0, ..., pn) is mapped to a set of numbers (i1, ..., in)
//! used to reorder the set of points. How the mapping is defined follows how encoding the Hilbert curve is
//! described in "Encoding and Decoding the Hilbert Order" by XIAN LIU and GÜNTHER SCHRACK
//!
//! The hilbert curve depends on a grid resolution called `order`. Basically,
//! the minimal bounding rectangle of the set of points is split in 2^(2*order) cells.
//! All the points in a given cell will have the same encoding.
//!
//! The complexity of encoding a point is O(order)

use geometry::{self, Mbr2D, Point2D};
use rayon::prelude::*;

pub fn hilbert_curve_reorder(
    mut points: Vec<Point2D>,
    mut weights: Vec<f64>,
    order: usize,
) -> (Vec<Point2D>, Vec<f64>) {
    let compute_hilbert_index = hilbert_index_computer(&points, order);

    let mut zipped = points
        .par_iter()
        .cloned()
        .zip(weights.par_iter().cloned())
        .zip(points.par_iter().map(|p| compute_hilbert_index((p.x, p.y))))
        .collect::<Vec<_>>();

    zipped.as_mut_slice().par_sort_by_key(|(_, idx)| *idx);

    let (still_zipped, _): (Vec<_>, Vec<_>) = zipped.into_par_iter().unzip();

    still_zipped
        .into_par_iter()
        .unzip_into_vecs(&mut points, &mut weights);

    (points, weights)
}

fn hilbert_index_computer(points: &[Point2D], order: usize) -> impl Fn((f64, f64)) -> i64 {
    let mbr = Mbr2D::from_points(points.iter());
    let rotation = mbr.rotation();
    let aabb = mbr.aabb();

    let ax = (aabb.p_min().x, aabb.p_max().x);
    let ay = (aabb.p_min().y, aabb.p_max().y);

    let rotate = geometry::rotation(rotation);

    let x_mapping = segment_to_segment(ax.0, ax.1, 0., order as f64);
    let y_mapping = segment_to_segment(ay.0, ay.1, 0., order as f64);

    move |p| {
        let (x, y) = rotate(p);
        encode(x_mapping(x) as i64, y_mapping(y) as i64, order)
    }
}

fn encode(x: i64, y: i64, order: usize) -> i64 {
    assert!(
        0 <= x && x < 2i64.pow(order as u32), 
        format!(
            "Cannot encode the point {:?} on an hilbert curve of order {} because x < 0 or x >= 2^order.",
            (x, y), 
            order
        )
    );
    assert!(
        0 <= y && y < 2i64.pow(order as u32), 
        format!(
            "Cannot encode the point {:?} on an hilbert curve of order {} because y < 0 or y >= 2^order.",
            (x, y),
            order
        )
    );

    let mask = (1 << order) - 1;
    let h_even = x ^ y;
    let not_x = !x & mask;
    let not_y = !y & mask;
    let temp = not_x ^ y;

    let mut v0 = 0;
    let mut v1 = 0;

    for _ in 1..order {
        v1 = ((v1 & h_even) | ((v0 ^ not_y) & temp)) >> 1;
        v0 = ((v0 & (v1 ^ not_x)) | (!v0 & (v1 ^ not_y))) >> 1;
    }

    let h_odd = (!v0 & (v1 ^ x)) | (v0 & (v1 ^ not_y));

    interleave_bits(h_odd, h_even)
}

fn interleave_bits(odd: i64, even: i64) -> i64 {
    let mut val = 0;
    let mut max: i32 = odd.max(even) as i32;
    let mut n = 0;
    while max > 0 {
        n += 1;
        max >>= 1;
    }

    for i in 0..n {
        let mask = 1 << i;
        let a = if (even & mask) > 0 { 1 << (2 * i) } else { 0 };
        let b = if (odd & mask) > 0 {
            1 << (2 * i + 1)
        } else {
            0
        };
        val += a + b;
    }

    val
}

// Compute a mapping from [a_min; a_max] to [b_min; b_max]
fn segment_to_segment(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> impl Fn(f64) -> f64 {
    assert!(
        a_min <= a_max,
        format!(
            "Cannot construct a segment to segment mapping because a_max < a_min. a_min = {}, a_max = {}.",
            a_min,
            a_max
        )
    );
    assert!(
        b_min <= b_max,
        format!(
            "Cannot construct a segment to segment mapping because b_max < b_min. b_min = {}, b_max = {}.",
            b_min,
            b_max
        )
    );

    let da = a_min - a_max;
    let db = b_min - b_max;
    let alpha = db / da;
    let beta = b_min - a_min * alpha;

    move |x| {
        assert!(
            a_min <= x && x <= a_max,
            format!(
                "Called a mapping from [{}, {}] to [{}, {}] with the invalid value {}.",
                a_min, a_max, b_min, b_max, x
            )
        );
        alpha * x + beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_to_segment() {
        let a_min = -1.;
        let a_max = 1.;
        let b_min = 0.;
        let b_max = 10.;

        let mapping = segment_to_segment(a_min, a_max, b_min, b_max);
        let inverse = segment_to_segment(b_min, b_max, a_min, a_max);

        assert_ulps_eq!(mapping(a_min), b_min);
        assert_ulps_eq!(mapping(a_max), b_max);
        assert_ulps_eq!(mapping(0.), 5.);
        assert_ulps_eq!(mapping(-0.5), 2.5);

        assert_ulps_eq!(inverse(b_min), a_min);
        assert_ulps_eq!(inverse(b_max), a_max);
        assert_ulps_eq!(inverse(5.), 0.);
        assert_ulps_eq!(inverse(2.5), -0.5);
    }

    #[test]
    #[should_panic]
    fn test_segment_to_segment_wrong_input() {
        let a_min = -1.;
        let a_max = 1.;
        let b_min = 0.;
        let b_max = 10.;

        let _mapping = segment_to_segment(a_max, a_min, b_min, b_max);
    }

    #[test]
    fn test_hilbert_curve_1() {
        let points = vec![(0, 0), (1, 1), (1, 0), (0, 1)];
        let indices = points
            .into_iter()
            .map(|(x, y)| encode(x, y, 1))
            .collect::<Vec<_>>();

        assert_eq!(indices, vec![0, 2, 3, 1]);
    }

    #[test]
    fn test_hilbert_curve_2() {
        let points = vec![
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (2, 0),
            (3, 0),
        ];

        let expected: Vec<_> = (0..16).collect();
        let indices = points
            .into_iter()
            .map(|(x, y)| encode(x, y, 2))
            .collect::<Vec<_>>();

        assert_eq!(indices, expected);
    }
}
