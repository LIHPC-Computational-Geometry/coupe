pub trait Two {
    fn two() -> Self;
}

impl Two for f64 {
    fn two() -> Self {
        2.0
    }
}

pub trait Sqrt {
    fn sqrt(&self) -> Self;
}

macro_rules! sqrt {
    ( $($t:ty,)* ) => { $(
        impl Sqrt for $t {
            fn sqrt(&self) -> Self {
                self.sqrt()
            }
        }
    )*};
}

sqrt! {
    f32, f64,
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
}
