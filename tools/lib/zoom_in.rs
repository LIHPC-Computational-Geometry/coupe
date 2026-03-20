use coupe::num_traits::NumAssign;
use coupe::num_traits::PrimInt;
use coupe::num_traits::ToPrimitive;
use itertools::Itertools;
use std::iter::Sum;

/// Scales and offsets the input weights into the bandwidth of type `O`.
///
/// This function ensures two things:
///
/// - output weights are strictly greater than zero,
/// - the sum of each criterion is in `O`.
///
/// If all weights are the same, returns `None`.
pub fn zoom_in<I, O, W>(inputs: I) -> Option<Vec<O>>
where
    I: IntoIterator,
    I::IntoIter: Clone + ExactSizeIterator,
    I::Item: IntoIterator<Item = W>,
    W: Sum + Clone + PartialOrd + ToPrimitive + NumAssign,
    O: PrimInt,
{
    let inputs = inputs.into_iter();
    let criterion_count = match inputs.clone().next() {
        Some(v) => v.into_iter().count(),
        None => return Some(Vec::new()),
    };
    let count = inputs.clone().count();

    // Compensated summation of each criterion.
    let (mut sums, compensations) = inputs.clone().fold(
        (vec![0.0; criterion_count], vec![0.0; criterion_count]),
        |(mut sum, mut compensation), w| {
            for ((s, w), c) in sum.iter_mut().zip(w).zip(&mut compensation) {
                let w = w.to_f64().unwrap();
                let t = *s + w;
                if f64::abs(w) <= f64::abs(*s) {
                    *c += (*s - t) + w;
                } else {
                    *c += (w - t) + *s;
                }
                *s = t;
            }
            (sum, compensation)
        },
    );
    for (s, c) in sums.iter_mut().zip(compensations) {
        *s += c;
    }

    if !sums.iter().all(|s| s.is_finite()) {
        // At least one of the sum is either infinite or NaN. In this case, just
        // abort. This can happen when some weights aren't finite, or when their
        // sum is infinite.
        // TODO handle the case where all weights are finite but their sum is infinite
        // TODO return an error somehow, so that the user knows the data is corrupt
        return None;
    }

    let sum = sums
        .into_iter()
        // won't panic because sums are ensured to be finite
        .max_by(|a, b| f64::partial_cmp(a, b).unwrap())
        .unwrap();

    let (min_input, max_input) = inputs.clone().flatten().minmax().into_option().unwrap();
    let min_input = min_input.to_f64().unwrap();
    let max_input = max_input.to_f64().unwrap();

    if !f64::is_normal(max_input - min_input) {
        // Input weight are all (about) the same, don't scale them.
        // Otherwise 1/(max_input-min_input) is not finite.
        return None;
    }

    // Actually scale from [0,max] to [1,INTMAX], because it makes
    // more sense in the case of load balancing.
    let min_input = f64::min(min_input, 0.0);

    let max_output = ((max_input - min_input) * O::max_value().to_f64().unwrap() + sum
        - max_input * count as f64)
        / (sum - min_input * count as f64);

    let scale = (max_output - 1.0) / (max_input - min_input);
    let offset = (max_input - max_output * min_input) / (max_input - min_input);

    let scaled_inputs: Vec<O> = inputs
        .flat_map(move |v| {
            v.into_iter().map(move |v| {
                let v = v.to_f64().unwrap();
                O::from(f64::mul_add(v, scale, offset)).unwrap()
            })
        })
        .collect();

    let Some(first_input) = scaled_inputs.first() else {
        return None;
    };
    if scaled_inputs.iter().all(|i| i == first_input) {
        None
    } else {
        Some(scaled_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zoom_in() {
        let z = zoom_in::<_, i32, _>(vec![[1.0 - f64::EPSILON], [1.0], [1.0 + f64::EPSILON]]);
        assert!(z.is_none());

        let z: Vec<i32> = zoom_in(vec![[0.0], [1.0]]).unwrap();
        assert_eq!(z, vec![1, i32::MAX - 1]);

        let z: Vec<i32> = zoom_in(vec![[0.0], [1.0], [1.0]]).unwrap();
        assert_eq!(z, vec![1, i32::MAX / 2, i32::MAX / 2]);

        let z = zoom_in::<_, i32, _>(vec![[i32::MAX - 2], [i32::MAX - 1], [i32::MAX]]);
        assert!(z.is_none());

        let z: Vec<i32> = zoom_in(vec![[i32::MAX - 6], [i32::MAX - 3], [i32::MAX]]).unwrap();
        assert_eq!(z, vec![i32::MAX / 3 - 1, i32::MAX / 3, i32::MAX / 3 + 1]);
    }
}
