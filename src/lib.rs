#[cfg(test)]
#[macro_use]
extern crate approx;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate nalgebra;
extern crate radix;
extern crate snowflake;

pub mod algorithms;
pub mod analysis;
pub mod geometry;

#[cfg(test)]
mod tests;
