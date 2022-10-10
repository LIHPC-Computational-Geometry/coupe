use num_traits::AsPrimitive;
use num_traits::NumAssign;
use num_traits::One;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::iter::Sum;

/// Divide `points` into `n` parts of similar weights.
///
/// The result is an array of `n` elements, the ith element is the the ???est
/// value of the ith part.
pub fn weighted_quantiles<P, W>(points: &[P], weights: &[W], n: usize) -> Vec<P>
where
    P: 'static + Copy + PartialOrd + Send + Sync,
    P: NumAssign + One,
    usize: AsPrimitive<P>,
    W: Send + Sync,
    W: NumAssign + Sum + AsPrimitive<f64>,
    usize: AsPrimitive<W>,
{
    debug_assert!(n > 0);

    const SPLIT_TOLERANCE: f64 = 0.05;
    let _2_p = {
        let _1_p = P::one();
        _1_p + _1_p
    };

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
                let new_position = (split.min_bound + split.max_bound) / _2_p;
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
