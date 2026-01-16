//! Wrapper around ittapi to avoid spreading #[cfg] flags in the codebase.
//!
//! Version of the module where ittapi is disabled. See `ittapi.rs`.

pub struct Domain;
pub struct Task;

// This Drop impl is needed to silence `clippy::drop_non_drop`.
impl Drop for Task {
    fn drop(&mut self) {}
}

pub fn domain(_: &str) -> Domain {
    Domain
}

pub fn begin(_: &Domain, _: &str) -> Task {
    Task
}
