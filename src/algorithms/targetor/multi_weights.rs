use num_traits::Zero;
use std::ops::{AddAssign, Deref, DerefMut, SubAssign};

#[derive(PartialOrd, PartialEq, Ord, Eq, Debug, Copy, Clone)]
pub(crate) struct MultiWeights<W, const NUM_CRITERIA: usize>([W; NUM_CRITERIA]);

impl<W, const NUM_CRITERIA: usize> Deref for MultiWeights<W, NUM_CRITERIA> {
    type Target = [W; NUM_CRITERIA];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<W, const NUM_CRITERIA: usize> DerefMut for MultiWeights<W, NUM_CRITERIA> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<W, const NUM_CRITERIA: usize> From<[W; NUM_CRITERIA]> for MultiWeights<W, NUM_CRITERIA> {
    fn from(value: [W; NUM_CRITERIA]) -> Self {
        Self(value)
    }
}

impl<'a, W, const NUM_CRITERIA: usize> TryFrom<&'a [W]> for MultiWeights<W, NUM_CRITERIA>
where
    [W; NUM_CRITERIA]: TryFrom<&'a [W]>,
{
    type Error = <[W; NUM_CRITERIA] as TryFrom<&'a [W]>>::Error;

    fn try_from(value: &'a [W]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

impl<W, const NUM_CRITERIA: usize> Default for MultiWeights<W, NUM_CRITERIA>
where
    W: Zero + Copy,
{
    fn default() -> Self {
        Self([W::zero(); NUM_CRITERIA])
    }
}

impl<W, const NUM_CRITERIA: usize> SubAssign for MultiWeights<W, NUM_CRITERIA>
where
    W: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(me, other)| *me -= *other);
    }
}

impl<W, const NUM_CRITERIA: usize> AddAssign for MultiWeights<W, NUM_CRITERIA>
where
    W: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(me, other)| *me += *other);
    }
}
