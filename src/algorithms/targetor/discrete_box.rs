use crate::algorithms::targetor::BoxOneIndex;
use num_traits::Zero;
use std::ops::{Deref, DerefMut};

#[derive(PartialOrd, PartialEq, Ord, Eq, Debug)]
pub(crate) struct BoxIndex<const NUM_CRITERIA: usize>([BoxOneIndex; NUM_CRITERIA]);

impl<const NUM_CRITERIA: usize> Deref for BoxIndex<NUM_CRITERIA> {
    type Target = [BoxOneIndex; NUM_CRITERIA];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const NUM_CRITERIA: usize> DerefMut for BoxIndex<NUM_CRITERIA> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const NUM_CRITERIA: usize> From<[BoxOneIndex; NUM_CRITERIA]> for BoxIndex<NUM_CRITERIA> {
    fn from(value: [BoxOneIndex; NUM_CRITERIA]) -> Self {
        Self(value)
    }
}

impl<const NUM_CRITERIA: usize> TryFrom<&[u32]> for BoxIndex<NUM_CRITERIA> {
    type Error = std::array::TryFromSliceError;

    fn try_from(value: &[u32]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

impl<const NUM_CRITERIA: usize> Default for BoxIndex<NUM_CRITERIA> {
    fn default() -> Self {
        Self([BoxOneIndex::zero(); NUM_CRITERIA])
    }
}

pub(crate) struct IterBoxIndices<'a, const NUM_CRITERIA: usize> {
    pub inner: Box<dyn Iterator<Item = BoxIndex<NUM_CRITERIA>> + 'a>,
}

impl<'a, const NUM_CRITERIA: usize> Iterator for IterBoxIndices<'a, NUM_CRITERIA> {
    type Item = BoxIndex<NUM_CRITERIA>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
