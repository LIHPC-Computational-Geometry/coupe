use super::Mesh;
use std::io;
use vtkio::Error;

pub fn parse_xml<const D: usize>(r: impl io::BufRead) -> Result<Mesh<D>, Error> {
    todo!()
}

pub fn parse_legacy_le<const D: usize>(r: impl io::BufRead) -> Result<Mesh<D>, Error> {
    todo!()
}

pub fn parse_legacy_be<const D: usize>(r: impl io::BufRead) -> Result<Mesh<D>, Error> {
    todo!()
}
