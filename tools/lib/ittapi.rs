//! Wrapper around ittapi to avoid spreading #[cfg] flags in the codebase.
//!
//! Version of the module where ittapi is enabled. See `ittapi_stub.rs`.

pub fn domain(s: &str) -> ittapi::Domain {
    ittapi::Domain::new(s)
}

pub fn begin<'a>(domain: &'a ittapi::Domain, name: &'a str) -> ittapi::Task<'a> {
    ittapi::Task::begin(domain, name)
}
