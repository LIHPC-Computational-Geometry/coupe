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
