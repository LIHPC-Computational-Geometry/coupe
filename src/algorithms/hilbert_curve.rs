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

use crate::geometry::OrientedBoundingBox;
use crate::Average;
use crate::Point2D;
use crate::Point3D;
use crate::PointND;
use num_traits::AsPrimitive;
use num_traits::NumAssign;
use rayon::prelude::*;
use std::fmt;
use std::iter::Sum;
use std::mem;

/// Divide `points` into `n` parts of similar weights.
///
/// The result is an array of `n` elements, the ith element is the the ???est
/// value of the ith part.
fn weighted_quantiles<P, W>(points: &[P], weights: &[W], n: usize) -> Vec<P>
where
    P: 'static + Copy + PartialOrd + Send + Sync,
    P: NumAssign + Average,
    usize: AsPrimitive<P>,
    W: Send + Sync,
    W: NumAssign + Sum + AsPrimitive<f64>,
    usize: AsPrimitive<W>,
{
    debug_assert!(n > 0);

    const SPLIT_TOLERANCE: f64 = 0.05;

    let (min, max) = rayon::join(
        || *points.par_iter().min_by(crate::partial_cmp).unwrap(),
        || *points.par_iter().max_by(crate::partial_cmp).unwrap(),
    );

    #[derive(Clone)]
    struct Split<P> {
        position: P,
        min_bound: P,
        max_bound: P,
        settled: bool,
    }

    let mut splits: Vec<Split<P>> = (1..n)
        .map(|i| Split {
            position: min + AsPrimitive::<P>::as_(i) * (max - min) / AsPrimitive::<P>::as_(n),
            min_bound: min,
            max_bound: max,
            settled: false,
        })
        .collect();

    // Number of splits that need to be settled.
    let mut todo_split_count = n - 1;

    while todo_split_count > 0 {
        let part_weights = points
            .par_iter()
            .zip(weights)
            .fold(
                || vec![W::zero(); n],
                |mut part_weights, (p, w)| {
                    let (Ok(split) | Err(split)) =
                        splits.binary_search_by(|split| crate::partial_cmp(&split.position, p));
                    part_weights[split] += *w;
                    part_weights
                },
            )
            .reduce_with(|mut pw0, pw1| {
                for (pw0, pw1) in pw0.iter_mut().zip(pw1) {
                    *pw0 += pw1;
                }
                pw0
            })
            .unwrap();

        let total_weight: W = part_weights.iter().cloned().sum();
        let prefix_left_weights = part_weights
            .iter()
            .scan(W::zero(), |weight_sum, part_weight| {
                *weight_sum += *part_weight;
                Some(*weight_sum)
            });

        splits = splits
            .iter()
            .cloned()
            .zip(prefix_left_weights)
            .enumerate()
            .map(|(p, (mut split, left_weight))| {
                if split.settled {
                    return split;
                }
                let left_weight_ratio = left_weight.as_() / (p + 1) as f64;
                let right_weight_ratio = (total_weight - left_weight).as_() / (n - p - 1) as f64;
                if f64::abs(left_weight_ratio - right_weight_ratio) / total_weight.as_()
                    < SPLIT_TOLERANCE
                {
                    split.settled = true;
                    todo_split_count -= 1;
                    return split;
                }
                let expected_left_weight = (p + 1) as f64 * total_weight.as_() / n as f64;
                if left_weight_ratio < right_weight_ratio {
                    split.min_bound = split.position;
                    let mut pw = left_weight;
                    for q in p + 1..n - 1 {
                        pw += part_weights[q];
                        if approx::abs_diff_eq!(pw.as_(), expected_left_weight) {
                            split.min_bound = splits[q].position;
                            split.max_bound = splits[q].position;
                            break;
                        } else if expected_left_weight < pw.as_() {
                            if splits[q].position < split.max_bound {
                                split.max_bound = splits[q].position;
                            }
                            break;
                        } else if pw.as_() < expected_left_weight {
                            split.min_bound = splits[q].position;
                        }
                    }
                } else {
                    split.max_bound = split.position;
                    let mut pw = left_weight;
                    for q in (0..p).rev() {
                        pw -= part_weights[q + 1];
                        if approx::abs_diff_eq!(pw.as_(), expected_left_weight) {
                            split.min_bound = splits[q].position;
                            split.max_bound = splits[q].position;
                            break;
                        } else if pw.as_() < expected_left_weight {
                            if split.min_bound < splits[q].position {
                                split.min_bound = splits[q].position;
                            }
                            break;
                        } else if expected_left_weight < pw.as_() {
                            split.max_bound = splits[q].position;
                        }
                    }
                }
                let new_position = P::avg(split.min_bound, split.max_bound);
                if split.position == new_position {
                    split.settled = true;
                    todo_split_count -= 1;
                    return split;
                }
                split.position = new_position;
                split
            })
            .collect();
    }

    splits.into_iter().map(|split| split.position).collect()
}

fn partition_indexed<const D: usize>(
    partition: &mut [usize],
    points: &[PointND<D>],
    weights: &[f64],
    part_count: usize,
    index_fn: impl Fn(&PointND<D>) -> u64 + Send + Sync,
) {
    let span = tracing::info_span!("compute indices");
    let enter = span.enter();

    let hilbert_indices: Vec<u64> = points.par_iter().map(index_fn).collect();

    mem::drop(enter);
    let span = tracing::info_span!("computing split positions");
    let enter = span.enter();

    let split_positions = weighted_quantiles(&hilbert_indices, weights, part_count);

    mem::drop(enter);
    let span = tracing::info_span!("apply part ids");
    let _enter = span.enter();

    partition
        .par_iter_mut()
        .zip(hilbert_indices)
        .for_each(|(part, index)| {
            let (Ok(part_id) | Err(part_id)) = split_positions.binary_search(&index);
            *part = part_id;
        });
}

/// Compute a mapping from [min; max] to [0; 2**order-1]
fn segment_to_segment(min: f64, max: f64, order: usize) -> impl Fn(f64) -> u64 {
    debug_assert!(min <= max);

    let width = max - min;
    let n = (1_u64 << order) as f64;
    let mut f = n / width;

    // Map max to (2**order-1) and avoid u64 overflow.
    while n <= width * f {
        f = crate::nextafter(f, 0.0);
    }

    move |v| {
        debug_assert!(min <= v && v <= max, "{v} not in [{min};{max}]");
        (f * (v - min)) as u64
    }
}

/// Returns a function that maps 2D points to their hilbert curve index.
///
/// Panics if `points` is empty.
fn index_fn_2d(points: &[Point2D], order: usize) -> impl Fn(&Point2D) -> u64 {
    let mbr = OrientedBoundingBox::from_points(points).unwrap();
    let aabb = mbr.aabb();
    let x_mapping = segment_to_segment(aabb.p_min.x, aabb.p_max.x, order);
    let y_mapping = segment_to_segment(aabb.p_min.y, aabb.p_max.y, order);
    move |p| {
        let p = mbr.obb_to_aabb(p);
        encode_2d(x_mapping(p.x), y_mapping(p.y), order)
    }
}

/// Returns a function that maps 3D points to their hilbert curve index.
///
/// Panics if `points` is empty.
fn index_fn_3d(points: &[Point3D], order: usize) -> impl Fn(&Point3D) -> u64 {
    let mbr = OrientedBoundingBox::from_points(points).unwrap();
    let aabb = mbr.aabb();
    let x_mapping = segment_to_segment(aabb.p_min.x, aabb.p_max.x, order);
    let y_mapping = segment_to_segment(aabb.p_min.y, aabb.p_max.y, order);
    let z_mapping = segment_to_segment(aabb.p_min.z, aabb.p_max.z, order);
    move |p| {
        let p = mbr.obb_to_aabb(p);
        encode_3d(x_mapping(p.x), y_mapping(p.y), z_mapping(p.z), order)
    }
}

/// Slower version of [encode_2d], to build the lookup table for [encode_2d].
///
/// This version takes the initial configuration as argument and also returns
/// the final configuration.
///
/// Taken from Marot, Célestin. "Parallel tetrahedral mesh generation." Prom:
/// Remacle, Jean-François <http://hdl.handle.net/2078.1/240626>.
///
/// TODO: once const-fn are more mature, take the "order" argument into account,
/// though it is only set to 6 for the purpose of building the lookup table.
const fn encode_2d_slow(zorder: u64, _order: usize, mut config: usize) -> (u64, usize) {
    // BASE_PATTERN[i][j] is the hilbert index given:
    // - i: the current configuration,
    // - j: the quadrant in row-major order.
    const BASE_PATTERN: [[u64; 4]; 4] = [
        [0, 1, 3, 2], // config 0 = [1,2]   )
        [0, 3, 1, 2], // config 1 = [2,1]   n
        [2, 3, 1, 0], // config 2 = [-1,-2] (
        [2, 1, 3, 0], // config 3 = [-2,-1] U
    ];
    // CONFIGURATION[i][j] is the next configuration given:
    // - i: the current configuration,
    // - j: the quadrant in row-major order.
    const CONFIGURATION: [[usize; 4]; 4] = [
        [1, 0, 3, 0], // ) => U)
        //                    n)
        [0, 2, 1, 1], // n => nn
        //                    )(
        [2, 1, 2, 3], // ( => (U
        //                    (n
        [3, 3, 0, 2], // U => )(
                      //      UU
    ];

    let mut hilbert = 0;

    // TODO replace unrolled loop by "for i in (0..order).rev()" once for loops
    // are allowed in const fns.
    let mut i = 5;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];
    i -= 1;
    let quadrant = (zorder >> (2 * i)) as usize & 3;
    hilbert = (hilbert << 2) | BASE_PATTERN[config][quadrant];
    config = CONFIGURATION[config][quadrant];

    (hilbert, config)
}

fn encode_2d(x: u64, y: u64, order: usize) -> u64 {
    debug_assert!(order < 64);
    debug_assert!(
        x < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because x >= 2^order.",
        (x, y),
        order,
    );
    debug_assert!(
        y < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because y >= 2^order.",
        (x, y),
        order,
    );

    const LUT: [u16; 16_384] = {
        let mut lut = [0; 16_384];
        let mut i: usize = 0;
        while i < 16_384 {
            let zorder = (i & 0xfff) as u64;
            let config = i >> 12;
            let (hilbert_order, config) = encode_2d_slow(zorder, 6, config);
            lut[i] = (config << 12) as u16 | hilbert_order as u16;
            i += 1;
        }
        lut
    };

    let zorder = unsafe {
        std::arch::x86_64::_pdep_u64(x, 0x5555_5555_5555_5555 << 1)
            | std::arch::x86_64::_pdep_u64(y, 0x5555_5555_5555_5555)
    };

    let mut config: u16 = 0;
    let mut hilbert: u64 = 0;
    let mut shift: i64 = 2 * order as i64 - 12;
    while shift > 0 {
        config = LUT[((config & !0xfff) | ((zorder >> shift) & 0xfff) as u16) as usize];
        hilbert = (hilbert << 12) | (config & 0xfff) as u64;
        shift -= 12;
    }

    config = LUT[((config & !0xfff) | ((zorder << (-shift) as u64) & 0xfff) as u16) as usize];
    hilbert = (hilbert << 12) | (config & 0xfff) as u64;

    hilbert >> -shift
}

fn encode_3d(x: u64, y: u64, z: u64, order: usize) -> u64 {
    debug_assert!(order < 64);
    debug_assert!(
        x < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because x >= 2^order.",
        (x, y, z),
        order,
    );
    debug_assert!(
        y < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because y >= 2^order.",
        (x, y, z),
        order,
    );
    debug_assert!(
        z < (1 << order),
        "Cannot encode the point {:?} on an hilbert curve of order {} because z >= 2^order.",
        (x, y, z),
        order,
    );

    #[allow(clippy::unusual_byte_groupings)]
    #[rustfmt::skip]
    const LUT: [u8; 96] = [
        0b0110_000, 0b0100_001, 0b0011_011, 0b0100_010, 0b0101_111, 0b1001_110, 0b0011_100, 0b1001_101,
        0b1000_010, 0b0011_101, 0b0110_011, 0b0110_100, 0b1000_001, 0b0011_110, 0b1001_000, 0b0111_111,
        0b1001_100, 0b1011_111, 0b1001_011, 0b0011_000, 0b0110_101, 0b0110_110, 0b1010_010, 0b1010_001,
        0b0010_010, 0b0000_011, 0b0010_001, 0b1010_000, 0b0111_101, 0b0000_100, 0b0111_110, 0b0001_111,
        0b0000_000, 0b0111_011, 0b1000_111, 0b0111_100, 0b0110_001, 0b0110_010, 0b1010_110, 0b1010_101,
        0b1010_100, 0b1010_011, 0b0000_101, 0b1011_010, 0b1001_111, 0b0111_000, 0b0000_110, 0b1011_001,
        0b0100_000, 0b0010_111, 0b0000_001, 0b1011_110, 0b0001_011, 0b0001_100, 0b0000_010, 0b1011_101,
        0b0101_010, 0b0101_001, 0b0001_101, 0b0001_110, 0b0100_011, 0b1011_000, 0b0100_100, 0b0011_111,
        0b1011_100, 0b0100_101, 0b1010_111, 0b0100_110, 0b1011_011, 0b1001_010, 0b0001_000, 0b1001_001,
        0b0101_110, 0b0101_101, 0b0001_001, 0b0001_010, 0b0000_111, 0b0010_100, 0b1000_000, 0b0010_011,
        0b1000_110, 0b0011_001, 0b0100_111, 0b0010_000, 0b1000_101, 0b0011_010, 0b0101_100, 0b0101_011,
        0b0010_110, 0b0110_111, 0b0010_101, 0b1000_100, 0b0111_001, 0b0101_000, 0b0111_010, 0b1000_011,
    ];

    let zorder = unsafe {
        std::arch::x86_64::_pdep_u64(x, 0x9249_2492_4924_9249 << 2)
            | std::arch::x86_64::_pdep_u64(y, 0x9249_2492_4924_9249 << 1)
            | std::arch::x86_64::_pdep_u64(z, 0x9249_2492_4924_9249)
    };

    let mut config = 0;
    let mut hilbert = 0;

    for i in (0..order).rev() {
        config = LUT[(config | ((zorder >> (3 * i)) & 7)) as usize] as u64;
        hilbert = (hilbert << 3) | (config & 7);
        config &= !7;
    }

    hilbert
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid space filling curve order.
    InvalidOrder { max: u32, actual: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidOrder { max, actual } => {
                write!(
                    f,
                    "given hilbert curve order too high. Got {}, max={}.",
                    actual, max
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// # Hilbert space-filling curve algorithm
///
/// Projects points on the hilbert curve and splits this curve into a given
/// amount of parts.
///
/// The hilbert curve depends on a grid resolution called `order`. Basically,
/// the minimal bounding rectangle of the set of points is split into
/// `2^order * 2^order` cells.  All the points in a given cell will have the
/// same encoding.
///
/// The complexity of encoding a point is `O(order)`.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::HilbertCurveError> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// let points: &[Point2D] = &[
///     Point2D::new(0., 0.),
///     Point2D::new(1., 1.),
///     Point2D::new(0., 10.),
///     Point2D::new(1., 9.),
///     Point2D::new(9., 1.),
///     Point2D::new(10., 0.),
///     Point2D::new(10., 10.),
///     Point2D::new(9., 9.),
/// ];
/// let weights = [1.0; 8];
/// let mut partition = [0; 8];
///
/// // generate a partition of 4 parts
/// coupe::HilbertCurve { part_count: 4, order: 5 }
///     .partition(&mut partition, (points, weights))?;
///
/// assert_eq!(partition[0], partition[1]);
/// assert_eq!(partition[2], partition[3]);
/// assert_eq!(partition[4], partition[5]);
/// assert_eq!(partition[6], partition[7]);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// Marot, Célestin. *Parallel tetrahedral mesh generation*. Prom.: Remacle,
/// Jean-François <http://hdl.handle.net/2078.1/240626>.
///
/// rawunprotected. *LUT-based 3D Hilbert curves*.
/// <http://threadlocalmutex.com/?p=149>
///
/// Deveci, M. et al., 2016. Multi-Jagged: A Scalable Parallel Spatial
/// Partitioning Algorithm. *IEEE Transactions on Parallel and Distributed
/// Systems*, 27(3), pp. 803–817. <https://doi.org/10.1109/TPDS.2015.2412545>
#[derive(Clone, Copy, Debug)]
pub struct HilbertCurve {
    pub part_count: usize,
    pub order: u32,
}

impl Default for HilbertCurve {
    fn default() -> Self {
        Self {
            part_count: 2,
            order: 12,
        }
    }
}

impl<W> crate::Partition<(&[Point2D], W)> for HilbertCurve
where
    W: AsRef<[f64]>,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&[Point2D], W),
    ) -> Result<Self::Metadata, Self::Error> {
        const MAX_ORDER: u32 = 32;
        if self.order > MAX_ORDER {
            return Err(Error::InvalidOrder {
                max: MAX_ORDER,
                actual: self.order,
            });
        }
        if part_ids.is_empty() {
            return Ok(());
        }
        let index_fn = index_fn_2d(points, self.order as usize);
        partition_indexed(
            part_ids,
            points,
            weights.as_ref(),
            self.part_count,
            index_fn,
        );
        Ok(())
    }
}

impl<W> crate::Partition<(&[Point3D], W)> for HilbertCurve
where
    W: AsRef<[f64]>,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (points, weights): (&[Point3D], W),
    ) -> Result<Self::Metadata, Self::Error> {
        const MAX_ORDER: u32 = 21;
        if self.order > MAX_ORDER {
            return Err(Error::InvalidOrder {
                max: MAX_ORDER,
                actual: self.order,
            });
        }
        if part_ids.is_empty() {
            return Ok(());
        }
        let index_fn = index_fn_3d(points, self.order as usize);
        partition_indexed(
            part_ids,
            points,
            weights.as_ref(),
            self.part_count,
            index_fn,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_to_segment() {
        let mapping = segment_to_segment(0.0, 8.0, 3);

        assert_eq!(mapping(0.0), 0);
        assert_eq!(mapping(1.0), 0);
        assert_eq!(mapping(2.0), 1);
        assert_eq!(mapping(3.0), 2);
        assert_eq!(mapping(4.0), 3);
        assert_eq!(mapping(5.0), 4);
        assert_eq!(mapping(6.0), 5);
        assert_eq!(mapping(7.0), 6);
        assert_eq!(mapping(8.0), 7);

        assert_eq!(mapping(crate::nextafter(1.0, f64::INFINITY)), 1);
        assert_eq!(mapping(crate::nextafter(2.0, f64::INFINITY)), 2);
        assert_eq!(mapping(crate::nextafter(3.0, f64::INFINITY)), 3);
        assert_eq!(mapping(crate::nextafter(4.0, f64::INFINITY)), 4);
        assert_eq!(mapping(crate::nextafter(5.0, f64::INFINITY)), 5);
        assert_eq!(mapping(crate::nextafter(6.0, f64::INFINITY)), 6);
        assert_eq!(mapping(crate::nextafter(7.0, f64::INFINITY)), 7);
    }

    #[test]
    fn test_hilbert_curve_1() {
        let points = vec![(0, 0), (1, 1), (1, 0), (0, 1)];
        let indices = points
            .into_iter()
            .map(|(x, y)| encode_2d(x, y, 1))
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
            .map(|(x, y)| encode_2d(x, y, 2))
            .collect::<Vec<_>>();

        assert_eq!(indices, expected);
    }
}
