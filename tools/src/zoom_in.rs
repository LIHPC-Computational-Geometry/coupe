use coupe::num_traits::PrimInt;
use coupe::num_traits::ToPrimitive;
use coupe::num_traits::Zero;
use std::iter::Sum;
use std::ops::AddAssign;

pub fn zoom_in<I, O, W>(inputs: I) -> Vec<O>
where
    I: IntoIterator,
    I::IntoIter: Clone + ExactSizeIterator,
    I::Item: IntoIterator<Item = W>,
    W: Sum + AddAssign + Clone + Zero + PartialOrd + ToPrimitive,
    O: PrimInt,
{
    let inputs = inputs.into_iter();
    let criterion_count = match inputs.clone().next() {
        Some(v) => v.into_iter().count(),
        None => return Vec::new(),
    };
    let sum = inputs
        .clone()
        .fold(vec![W::zero(); criterion_count], |mut acc, w| {
            for (acc, w) in acc.iter_mut().zip(w) {
                *acc += w;
            }
            acc
        })
        .into_iter()
        .max_by(|a, b| W::partial_cmp(a, b).unwrap())
        .unwrap();
    let zoom_factor = (O::max_value() - O::from(inputs.len()).unwrap())
        .to_f64()
        .unwrap()
        / sum.to_f64().unwrap();
    let _1 = O::one();
    inputs
        .flat_map(move |v| {
            v.into_iter()
                .map(move |v| O::from(v.to_f64().unwrap() * zoom_factor).unwrap() + _1)
        })
        .collect()
}
