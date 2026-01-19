/// Compute the average of two values without overflow.
pub trait Average {
    fn avg(a: Self, b: Self) -> Self;
}

impl Average for f64 {
    fn avg(a: Self, b: Self) -> Self {
        (a + b) / 2.0
    }
}

macro_rules! impl_int {
    ( $t:ty ) => {
        impl Average for $t {
            /// Ref: <http://aggregate.org/MAGIC/#Average%20of%20Integers>
            fn avg(a: Self, b: Self) -> Self {
                (a & b) + (a ^ b) / 2
            }
        }
    };
}

impl_int!(i64);
impl_int!(u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int() {
        assert_eq!(i64::avg(i64::MAX, i64::MAX - 2), i64::MAX - 1);
        assert_eq!(u64::avg(u64::MAX, u64::MAX - 2), u64::MAX - 1);
    }

    #[test]
    fn test_float() {
        assert_eq!(f64::avg(1e308, 1e307), 5.5e307);
    }
}
