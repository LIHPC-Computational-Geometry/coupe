#[cfg(test)]
#[macro_use]
extern crate approx;
#[macro_use]
extern crate itertools;
extern crate nalgebra;
extern crate snowflake;

pub mod algorithms;
pub mod analysis;
pub mod geometry;

#[cfg(test)]
mod tests;
