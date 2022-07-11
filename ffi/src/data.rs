use rayon::iter::IntoParallelIterator as _;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelExtend as _;
use rayon::iter::ParallelIterator as _;
use std::borrow::Cow;
use std::collections::TryReserveError;
use std::ffi::c_void;
use std::slice;

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub enum Type {
    Int,
    Int64,
    Double,
}

pub struct Constant {
    pub len: usize,
    pub type_: Type,
    pub value: *const c_void,
}

impl Constant {
    pub fn len(&self) -> usize {
        self.len
    }

    pub unsafe fn to_slice<T>(&self) -> Result<Vec<T>, TryReserveError>
    where
        T: Copy,
    {
        let mut v = Vec::new();
        v.try_reserve_exact(self.len)?;
        let value = *(self.value as *const T);
        v.resize(self.len, value);
        Ok(v)
    }

    pub unsafe fn iter<'a, T>(&'a self) -> impl Iterator<Item = T> + ExactSizeIterator + 'a
    where
        T: 'a + Copy,
    {
        let value = *(self.value as *const T);
        (0..self.len).map(move |_| value)
    }

    pub unsafe fn par_iter<'a, T>(
        &'a self,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = T> + Clone + 'a
    where
        T: 'a + Copy + Send + Sync,
    {
        let value = *(self.value as *const T);
        rayon::iter::repeatn(value, self.len)
    }
}

pub struct Array {
    pub len: usize,
    pub type_: Type,
    pub array: *const c_void,
}

impl Array {
    pub fn len(&self) -> usize {
        self.len
    }

    pub unsafe fn to_slice<'a, T>(&'a self) -> &'a [T]
    where
        T: 'a,
    {
        slice::from_raw_parts(self.array as *const T, self.len)
    }

    pub unsafe fn iter<'a, T>(&'a self) -> impl Iterator<Item = T> + ExactSizeIterator + 'a
    where
        T: 'a + Copy,
    {
        slice::from_raw_parts(self.array as *const T, self.len)
            .iter()
            .cloned()
    }

    pub unsafe fn par_iter<'a, T>(
        &'a self,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = T> + Clone + 'a
    where
        T: 'a + Copy + Send + Sync,
    {
        slice::from_raw_parts(self.array as *const T, self.len)
            .par_iter()
            .cloned()
    }
}

pub struct Fn {
    pub len: usize,
    pub type_: Type,
    pub context: *const c_void,
    pub i_th: extern "C" fn(*const c_void, usize) -> *const c_void,
}

unsafe impl Send for Fn {}
unsafe impl Sync for Fn {}

impl Fn {
    pub fn len(&self) -> usize {
        self.len
    }

    pub unsafe fn to_slice<T>(&self) -> Result<Vec<T>, TryReserveError>
    where
        T: Copy + Send,
    {
        let mut v = Vec::new();
        v.try_reserve_exact(self.len)?;
        v.par_extend(self.par_iter::<T>());
        Ok(v)
    }

    pub unsafe fn iter<'a, T>(&'a self) -> impl Iterator<Item = T> + ExactSizeIterator + 'a
    where
        T: 'a + Copy,
    {
        (0..self.len).map(|i| {
            let ptr = (self.i_th)(self.context, i) as *const T;
            *ptr
        })
    }

    pub unsafe fn par_iter<'a, T>(
        &'a self,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = T> + Clone + 'a
    where
        T: 'a + Copy + Send,
    {
        (0..self.len).into_par_iter().map(|i| {
            let ptr = (self.i_th)(self.context, i) as *const T;
            *ptr
        })
    }
}

pub enum Data {
    Array(Array),
    Constant(Constant),
    Fn(Fn),
}

impl Data {
    pub fn len(&self) -> usize {
        match self {
            Self::Array(iter) => iter.len(),
            Self::Constant(iter) => iter.len(),
            Self::Fn(iter) => iter.len(),
        }
    }

    pub unsafe fn to_slice<T>(&self) -> Result<Cow<'_, [T]>, TryReserveError>
    where
        T: Copy + Send,
    {
        Ok(match self {
            Self::Array(iter) => Cow::from(iter.to_slice()),
            Self::Constant(iter) => Cow::from(iter.to_slice()?),
            Self::Fn(iter) => Cow::from(iter.to_slice()?),
        })
    }

    pub fn type_(&self) -> Type {
        match self {
            Self::Array(iter) => iter.type_,
            Self::Constant(iter) => iter.type_,
            Self::Fn(iter) => iter.type_,
        }
    }
}

/// Replacement for `Data::iter` since you can't have `Box<dyn Trait1 + Trait2>`
/// yet.
///
/// TODO find a way to not use a macro
///
/// # Example
///
/// ```rust,ignore
/// let i = coupe_data_constant(5, 0xDEADBEEF);
/// let v = with_iter!(i, f64, {
///     i.map(|f| f.ceil()).collect::<Vec<f64>>()
/// });
/// println!("{v:?}");
/// ```
#[macro_export]
macro_rules! with_iter {
    ( $iter:ident, $t:ty, $do:block ) => {
        match $iter {
            $crate::data::Data::Array($iter) => {
                let $iter = $iter.iter::<$t>();
                $do
            }
            $crate::data::Data::Constant($iter) => {
                let $iter = $iter.iter::<$t>();
                $do
            }
            $crate::data::Data::Fn($iter) => {
                let $iter = $iter.iter::<$t>();
                $do
            }
        }
    };
    ( $iter:ident, $do:block ) => {
        match $iter.type_() {
            $crate::data::Type::Int => with_iter!($iter, std::os::raw::c_int, $do),
            $crate::data::Type::Int64 => with_iter!($iter, i64, $do),
            $crate::data::Type::Double => with_iter!($iter, f64, $do),
        }
    };
}

/// Replacement for `Data::par_iter` since
/// `rayon::iter::IndexedParallelIterator` cannot be made into a trait object.
///
/// TODO find a way to not use a macro
///
/// # Example
///
/// ```rust,ignore
/// let i = coupe_data_array(5, 0xDEADBEEF);
/// let v = with_par_iter!(i, f64, {
///     i.map(|f| f.ceil()).collect::<Vec<f64>>()
/// });
/// println!("{v:?}");
/// ```
#[macro_export]
macro_rules! with_par_iter {
    ( $iter:ident, $t:ty, $do:block ) => {
        match $iter {
            $crate::data::Data::Array($iter) => {
                let $iter = $iter.par_iter::<$t>();
                $do
            }
            $crate::data::Data::Constant($iter) => {
                let $iter = $iter.par_iter::<$t>();
                $do
            }
            $crate::data::Data::Fn($iter) => {
                let $iter = $iter.par_iter::<$t>();
                $do
            }
        }
    };
    ( $iter:ident, $do:block ) => {
        match $iter.type_() {
            $crate::data::Type::Int => with_par_iter!($iter, std::os::raw::c_int, $do),
            $crate::data::Type::Int64 => with_par_iter!($iter, i64, $do),
            $crate::data::Type::Double => with_par_iter!($iter, f64, $do),
        }
    };
}

#[macro_export]
macro_rules! with_slice {
    ( $iter:ident, $do:block ) => {
        match $iter.type_() {
            $crate::data::Type::Int => {
                let $iter = match $iter.to_slice::<std::os::raw::c_int>() {
                    Ok(v) => v,
                    Err(_) => return Error::Alloc,
                };
                $do
            }
            $crate::data::Type::Int64 => {
                let $iter = match $iter.to_slice::<i64>() {
                    Ok(v) => v,
                    Err(_) => return Error::Alloc,
                };
                $do
            }
            $crate::data::Type::Double => {
                let $iter = match $iter.to_slice::<f64>() {
                    Ok(v) => v,
                    Err(_) => return Error::Alloc,
                };
                $do
            }
        }
    };
}
