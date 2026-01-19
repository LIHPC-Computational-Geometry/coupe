// Adapted from the float_next_after crate, under MIT.
// See <https://gitlab.com/bronsonbdevost/next_afterf/-/blob/9c96e8416f07298e6b6e69940b012a2908187287/LICENSE>

pub fn nextafter(from: f64, to: f64) -> f64 {
    if from == to {
        to
    } else if from.is_nan() || to.is_nan() {
        f64::NAN
    } else if from >= f64::INFINITY {
        f64::INFINITY
    } else if from <= f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else if from == 0_f64 {
        f64::copysign(f64::from_bits(1), to)
    } else {
        let ret = if (from < to) == (0_f64 < from) {
            f64::from_bits(from.to_bits() + 1)
        } else {
            f64::from_bits(from.to_bits() - 1)
        };
        if ret == 0_f64 {
            f64::copysign(ret, from)
        } else {
            ret
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POS_INF: f64 = f64::INFINITY;
    const NEG_INF: f64 = f64::NEG_INFINITY;
    const POS_ZERO: f64 = 0.0;
    const NEG_ZERO: f64 = -0.0;

    // Note: Not the same as f64::MIN_POSITIVE, because that is only the min *normal* number.
    const SMALLEST_POS: f64 = 5e-324;
    const SMALLEST_NEG: f64 = -5e-324;
    const LARGEST_POS: f64 = f64::MAX;
    const LARGEST_NEG: f64 = f64::MIN;

    const POS_ONE: f64 = 1.0;
    const NEG_ONE: f64 = -1.0;
    const NEXT_LARGER_THAN_ONE: f64 = 1.0 + f64::EPSILON;
    const NEXT_SMALLER_THAN_ONE: f64 = 0.999_999_999_999_999_9;

    const SEQUENCE_BIG_NUM: (f64, f64) = (16_237_485_966.000_004, 16_237_485_966.000_006);

    const NAN: f64 = f64::NAN;

    fn is_pos_zero(x: f64) -> bool {
        x.to_bits() == POS_ZERO.to_bits()
    }

    fn is_neg_zero(x: f64) -> bool {
        x.to_bits() == NEG_ZERO.to_bits()
    }

    #[test]
    fn next_larger_than_0() {
        assert_eq!(nextafter(POS_ZERO, POS_INF), SMALLEST_POS);
        assert_eq!(nextafter(NEG_ZERO, POS_INF), SMALLEST_POS);
    }

    #[test]
    fn next_smaller_than_0() {
        assert_eq!(nextafter(POS_ZERO, NEG_INF), SMALLEST_NEG);
        assert_eq!(nextafter(NEG_ZERO, NEG_INF), SMALLEST_NEG);
    }

    #[test]
    fn step_towards_zero() {
        // For steps towards zero, the sign of the zero reflects the direction
        // from where zero was approached.
        assert!(is_pos_zero(nextafter(SMALLEST_POS, POS_ZERO)));
        assert!(is_pos_zero(nextafter(SMALLEST_POS, NEG_ZERO)));
        assert!(is_pos_zero(nextafter(SMALLEST_POS, NEG_INF)));
        assert!(is_neg_zero(nextafter(SMALLEST_NEG, NEG_ZERO)));
        assert!(is_neg_zero(nextafter(SMALLEST_NEG, POS_ZERO)));
        assert!(is_neg_zero(nextafter(SMALLEST_NEG, POS_INF)));
    }

    #[test]
    fn special_case_signed_zeros() {
        // For a non-zero dest, stepping away from either POS_ZERO or NEG_ZERO
        // has a non-zero result. Only if the destination itself points to the
        // "other zero", the next_after call performs a zero sign switch.
        assert!(is_neg_zero(nextafter(POS_ZERO, NEG_ZERO)));
        assert!(is_pos_zero(nextafter(NEG_ZERO, POS_ZERO)));
    }

    #[test]
    fn nextafter_around_one() {
        assert_eq!(nextafter(POS_ONE, POS_INF), NEXT_LARGER_THAN_ONE);
        assert_eq!(nextafter(POS_ONE, NEG_INF), NEXT_SMALLER_THAN_ONE);
        assert_eq!(nextafter(NEG_ONE, NEG_INF), -NEXT_LARGER_THAN_ONE);
        assert_eq!(nextafter(NEG_ONE, POS_INF), -NEXT_SMALLER_THAN_ONE);
    }

    #[test]
    fn nextafter_for_big_pos_number() {
        let (lo, hi) = SEQUENCE_BIG_NUM;
        assert_eq!(nextafter(lo, POS_INF), hi);
        assert_eq!(nextafter(hi, NEG_INF), lo);
        assert_eq!(nextafter(lo, hi), hi);
        assert_eq!(nextafter(hi, lo), lo);
    }

    #[test]
    fn nextafter_for_big_neg_number() {
        let (lo, hi) = SEQUENCE_BIG_NUM;
        let (lo, hi) = (-lo, -hi);
        assert_eq!(nextafter(lo, NEG_INF), hi);
        assert_eq!(nextafter(hi, POS_INF), lo);
        assert_eq!(nextafter(lo, hi), hi);
        assert_eq!(nextafter(hi, lo), lo);
    }

    #[test]
    fn step_to_largest_is_possible() {
        let smaller = nextafter(LARGEST_POS, NEG_INF);
        assert_eq!(nextafter(smaller, POS_INF), LARGEST_POS);
        let smaller = nextafter(LARGEST_NEG, POS_INF);
        assert_eq!(nextafter(smaller, NEG_INF), LARGEST_NEG);
    }

    #[test]
    fn jump_to_infinity() {
        // Incrementing the max representable number has to go to infinity.
        assert_eq!(nextafter(LARGEST_POS, POS_INF), POS_INF);
        assert_eq!(nextafter(LARGEST_NEG, NEG_INF), NEG_INF);
    }

    #[test]
    fn stays_at_infinity() {
        // Once infinity is reached, there is not going back to normal numbers
        assert_eq!(nextafter(POS_INF, NEG_INF), POS_INF);
        assert_eq!(nextafter(NEG_INF, POS_INF), NEG_INF);
    }

    #[test]
    fn returns_nan_for_any_nan_involved() {
        assert!(nextafter(NAN, POS_ONE).is_nan());
        assert!(nextafter(POS_ONE, NAN).is_nan());
        assert!(nextafter(NAN, NAN).is_nan());
    }

    #[test]
    fn returns_identity_for_equal_dest() {
        let values = [
            POS_ZERO,
            NEG_ZERO,
            POS_ONE,
            NEG_ONE,
            SEQUENCE_BIG_NUM.0,
            SEQUENCE_BIG_NUM.1,
            POS_INF,
            NEG_INF,
            SMALLEST_POS,
            SMALLEST_NEG,
            LARGEST_POS,
            LARGEST_NEG,
        ];
        for x in values {
            assert_eq!(nextafter(x, x), x);
        }
    }

    #[test]
    fn roundtrip() {
        let values = [
            POS_ONE,
            NEG_ONE,
            SEQUENCE_BIG_NUM.0,
            SEQUENCE_BIG_NUM.1,
            SMALLEST_POS,
            SMALLEST_NEG,
        ];
        for orig in values {
            assert_eq!(nextafter(nextafter(orig, POS_INF), NEG_INF), orig);
            assert_eq!(nextafter(nextafter(orig, NEG_INF), POS_INF), orig);

            let upper = nextafter(orig, POS_INF);
            let lower = nextafter(orig, NEG_INF);

            assert_eq!(nextafter(nextafter(orig, upper), lower), orig);
            assert_eq!(nextafter(nextafter(orig, lower), upper), orig);
        }
    }
}
