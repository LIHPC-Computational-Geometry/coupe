use std::collections::TryReserveError;

/// Extension trait to handle allocation failures on [Vec] operations.
pub trait VecExt<T> {
    fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError>
    where
        Self: Sized;

    /// Fallible equivalent of [`vec!`].
    fn try_filled(value: T, len: usize) -> Result<Self, TryReserveError>
    where
        Self: Sized,
        T: Clone;

    /// Fallible equivalent of [`Vec::push`].
    fn try_push(&mut self, value: T) -> Result<(), TryReserveError>;
}

impl<T> VecExt<T> for Vec<T> {
    fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError>
    where
        Self: Sized,
    {
        let mut v = Vec::new();
        v.try_reserve(capacity)?;
        Ok(v)
    }

    fn try_filled(value: T, len: usize) -> Result<Self, TryReserveError>
    where
        T: Clone,
    {
        let mut v = Vec::new();
        v.try_reserve_exact(len)?;
        v.resize(len, value);
        Ok(v)
    }

    fn try_push(&mut self, value: T) -> Result<(), TryReserveError> {
        self.try_reserve(1)?;
        self.push(value);
        Ok(())
    }
}

pub trait SliceExt<T> {
    fn try_to_vec(&self) -> Result<Vec<T>, TryReserveError>
    where
        Self: Sized,
        T: Clone;
}

impl<T> SliceExt<T> for &[T] {
    fn try_to_vec(&self) -> Result<Vec<T>, TryReserveError>
    where
        Self: Sized,
        T: Clone,
    {
        let mut v = Vec::new();
        v.try_reserve_exact(self.len())?;
        v.extend_from_slice(self);
        Ok(v)
    }
}
