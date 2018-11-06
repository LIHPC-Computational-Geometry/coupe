#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate itertools;
extern crate nalgebra;
extern crate rayon;
extern crate snowflake;

pub mod algorithms;
pub mod analysis;
pub mod geometry;

#[cfg(test)]
mod tests;
