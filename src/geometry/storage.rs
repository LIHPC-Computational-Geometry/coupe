use std::iter::Sum;
use std::mem;
use std::ops::Index;
use std::ops::Mul;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PointND<T, const D: usize>(pub [T; D]);

pub type Point2D<T> = PointND<T, 2>;
pub type Point3D<T> = PointND<T, 3>;

impl<T, const D: usize> PointND<T, D> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }
}

impl<T, const D: usize> PointND<T, D>
where
    T: crate::Sqrt + Sum,
    for<'e> &'e T: Mul<Output = T>,
{
    pub fn norm(&self) -> T {
        let sum: T = self.iter().map(|elem| elem * elem).sum();
        T::sqrt(&sum)
    }
}

impl<T, const D: usize> Index<usize> for PointND<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T, const D: usize> IntoIterator for PointND<T, D> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, const D: usize> std::iter::FromIterator<T> for PointND<T, D> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();

        let mut arr: [mem::MaybeUninit<T>; D] = unsafe { mem::MaybeUninit::uninit().assume_init() };
        let mut count = 0;
        for elem in (&mut iter).take(D) {
            arr[count].write(elem);
            count += 1;
        }

        assert_eq!(
            D, count,
            "PointND::<_, {}>::from_iter() called with an iterator of {} elements",
            D, count,
        );
        if cfg!(debug_assertions) {
            count += iter.count();
            debug_assert_eq!(
                D, count,
                "PointND::<_, {}>::from_iter() called with an iterator of {} elements",
                D, count,
            );
        }

        let arr = unsafe { mem::transmute::<_, [T; D]>(arr) };
        PointND(arr)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Matrix<T, const R: usize, const C: usize>(pub [[T; R]; C]);

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.0[row][col]
    }
}
