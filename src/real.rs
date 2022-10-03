use std::cmp::Ordering;
use std::{fmt, iter, ops};

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

/// A floating-point value that cannot be NAN nor infinity.
///
/// It implements every trait [f64] implements, plus [std::cmp::Ord] and [std::cmp::Eq].
///
/// # Example
///
/// ```
/// # use coupe::Real;
/// let a = Real::from(7.0);
/// let b = Real::from(5.0);
/// println!("{} / {} = {}", a, b, a / b);
/// ```
#[repr(transparent)]
#[derive(PartialEq, Copy, Clone, Default)]
pub struct Real(f64);

impl Real {
    pub const EPSILON: Real = Real(f64::EPSILON);

    pub fn abs(self) -> Real {
        Real(f64::abs(self.0))
    }
}

impl Eq for Real {}

impl FromPrimitive for Real {
    fn from_i64(n: i64) -> Option<Real> {
        f64::from_i64(n).filter(|f| f.is_finite()).map(Real)
    }

    fn from_u64(n: u64) -> Option<Real> {
        f64::from_u64(n).filter(|f| f.is_finite()).map(Real)
    }
}

impl ToPrimitive for Real {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
}

impl Zero for Real {
    fn zero() -> Self {
        f64::zero().into()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for Real {
    fn one() -> Real {
        Real::from(1.0)
    }
}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Real) -> Option<Ordering> {
        Some(
            self.0
                .partial_cmp(&other.0)
                .expect("cannot compare with NaN"),
        )
    }
}

impl Ord for Real {
    fn cmp(&self, other: &Real) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("cannot compare with NaN")
    }
}

impl From<&f64> for Real {
    fn from(&v: &f64) -> Real {
        assert!(v.is_finite());
        Real(v)
    }
}

impl From<f64> for Real {
    fn from(v: f64) -> Real {
        assert!(v.is_finite());
        Real(v)
    }
}

impl From<&usize> for Real {
    fn from(&v: &usize) -> Real {
        Real(v as f64)
    }
}

impl From<usize> for Real {
    fn from(v: usize) -> Real {
        Real(v as f64)
    }
}

impl From<&Real> for f64 {
    fn from(v: &Real) -> f64 {
        v.0
    }
}

impl From<Real> for f64 {
    fn from(v: Real) -> f64 {
        v.0
    }
}

impl fmt::Debug for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::UpperExp for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.0, f)
    }
}

impl<'a> iter::Product<&'a Real> for Real {
    fn product<I>(iter: I) -> Real
    where
        I: Iterator<Item = &'a Real>,
    {
        iter.fold(Real::from(1.0), ops::Mul::mul)
    }
}

impl iter::Product<Real> for Real {
    fn product<I>(iter: I) -> Real
    where
        I: Iterator<Item = Real>,
    {
        iter.fold(Real::from(1.0), ops::Mul::mul)
    }
}

impl<'a> iter::Sum<&'a Real> for Real {
    fn sum<I>(iter: I) -> Real
    where
        I: Iterator<Item = &'a Real>,
    {
        iter.fold(Real::from(0.0), ops::Add::add)
    }
}

impl iter::Sum<Real> for Real {
    fn sum<I>(iter: I) -> Real
    where
        I: Iterator<Item = Real>,
    {
        iter.fold(Real::from(0.0), ops::Add::add)
    }
}

impl ops::Add<&Real> for &Real {
    type Output = Real;
    fn add(self, other: &Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl ops::Add<&Real> for Real {
    type Output = Real;
    fn add(self, other: &Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl ops::Add<Real> for &Real {
    type Output = Real;
    fn add(self, other: Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl ops::Add<Real> for Real {
    type Output = Real;
    fn add(self, other: Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl ops::AddAssign<&Real> for Real {
    fn add_assign(&mut self, other: &Real) {
        self.0 += other.0;
    }
}

impl ops::AddAssign<Real> for Real {
    fn add_assign(&mut self, other: Real) {
        self.0 += other.0;
    }
}

impl ops::Div<&Real> for &Real {
    type Output = Real;
    fn div(self, other: &Real) -> Real {
        Real(self.0 / other.0)
    }
}

impl ops::Div<&Real> for Real {
    type Output = Real;
    fn div(self, other: &Real) -> Real {
        Real(self.0 / other.0)
    }
}

impl ops::Div<Real> for &Real {
    type Output = Real;
    fn div(self, other: Real) -> Real {
        Real(self.0 / other.0)
    }
}

impl ops::Div<Real> for Real {
    type Output = Real;
    fn div(self, other: Real) -> Real {
        Real(self.0 / other.0)
    }
}

impl ops::DivAssign<&Real> for Real {
    fn div_assign(&mut self, other: &Real) {
        self.0 /= other.0;
    }
}

impl ops::DivAssign<Real> for Real {
    fn div_assign(&mut self, other: Real) {
        self.0 /= other.0;
    }
}

impl ops::Mul<&Real> for &Real {
    type Output = Real;
    fn mul(self, other: &Real) -> Real {
        Real(self.0 * other.0)
    }
}

impl ops::Mul<&Real> for Real {
    type Output = Real;
    fn mul(self, other: &Real) -> Real {
        Real(self.0 * other.0)
    }
}

impl ops::Mul<Real> for &Real {
    type Output = Real;
    fn mul(self, other: Real) -> Real {
        Real(self.0 * other.0)
    }
}

impl ops::Mul<Real> for Real {
    type Output = Real;
    fn mul(self, other: Real) -> Real {
        Real(self.0 * other.0)
    }
}

impl ops::MulAssign<&Real> for Real {
    fn mul_assign(&mut self, other: &Real) {
        self.0 *= other.0;
    }
}

impl ops::MulAssign<Real> for Real {
    fn mul_assign(&mut self, other: Real) {
        self.0 *= other.0;
    }
}

impl ops::Neg for &Real {
    type Output = Real;
    fn neg(self) -> Real {
        Real(-self.0)
    }
}

impl ops::Neg for Real {
    type Output = Real;
    fn neg(self) -> Real {
        Real(-self.0)
    }
}

impl ops::Rem<&Real> for &Real {
    type Output = Real;
    fn rem(self, other: &Real) -> Real {
        Real(self.0 % other.0)
    }
}

impl ops::Rem<&Real> for Real {
    type Output = Real;
    fn rem(self, other: &Real) -> Real {
        Real(self.0 % other.0)
    }
}

impl ops::Rem<Real> for &Real {
    type Output = Real;
    fn rem(self, other: Real) -> Real {
        Real(self.0 % other.0)
    }
}

impl ops::Rem<Real> for Real {
    type Output = Real;
    fn rem(self, other: Real) -> Real {
        Real(self.0 % other.0)
    }
}

impl ops::RemAssign<&Real> for Real {
    fn rem_assign(&mut self, other: &Real) {
        self.0 %= other.0;
    }
}

impl ops::RemAssign<Real> for Real {
    fn rem_assign(&mut self, other: Real) {
        self.0 %= other.0;
    }
}

impl ops::Sub<&Real> for &Real {
    type Output = Real;
    fn sub(self, other: &Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl ops::Sub<&Real> for Real {
    type Output = Real;
    fn sub(self, other: &Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl ops::Sub<Real> for &Real {
    type Output = Real;
    fn sub(self, other: Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl ops::Sub<Real> for Real {
    type Output = Real;
    fn sub(self, other: Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl ops::SubAssign<&Real> for Real {
    fn sub_assign(&mut self, other: &Real) {
        self.0 -= other.0;
    }
}

impl ops::SubAssign<Real> for Real {
    fn sub_assign(&mut self, other: Real) {
        self.0 -= other.0;
    }
}
