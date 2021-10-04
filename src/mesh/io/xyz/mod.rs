use crate::mesh::{Mesh, D3};
use pest::Parser;

use std::fmt;
use std::fs;
use std::io;
use std::num;
use std::path::Path;

#[derive(Parser)]
#[grammar = "xyz/xyz.pest"]
pub struct XYZParser;

#[derive(Debug)]
pub enum Error {
    Parse(pest::error::Error<Rule>),
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(err) => err.fmt(f),
            Error::Io(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<pest::error::Error<Rule>> for Error {
    fn from(err: pest::error::Error<Rule>) -> Error {
        Error::Parse(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

pub struct XYZMesh {
    vertices: Vec<(f64, f64, f64)>,
}

impl Mesh for XYZMesh {
    type Dim = D3;
    fn vertices(&self) -> Vec<(f64, f64, f64)> {
        self.vertices.clone()
    }

    fn elements_vertices(&self) -> Vec<Vec<usize>> {
        self.vertices
            .iter()
            .enumerate()
            .map(|(idx, _)| vec![idx])
            .collect()
    }
}

impl XYZMesh {
    pub fn from_string(data: impl Into<String>) -> Result<Self, Error> {
        let data = data.into();
        let record_pairs = XYZParser::parse(Rule::file, &data)?
            .next()
            .unwrap()
            .into_inner();

        let mut vertices = Vec::new();
        let (lb, ub) = record_pairs.size_hint();

        if let Some(ub) = ub {
            vertices.reserve(ub);
        } else {
            vertices.reserve(lb);
        }

        let to_parse_err =
            |err: num::ParseFloatError, span: pest::Span| -> pest::error::Error<Rule> {
                let variant = pest::error::ErrorVariant::CustomError {
                    message: err.to_string(),
                };
                pest::error::Error::new_from_span(variant, span)
            };

        for record in record_pairs {
            match record.as_rule() {
                Rule::record => {
                    let mut fields = record.into_inner();
                    let field1 = fields.next().unwrap();
                    let field2 = fields.next().unwrap();
                    let field3 = fields.next().unwrap();
                    vertices.push((
                        field1
                            .as_str()
                            .parse()
                            .map_err(|err| to_parse_err(err, field1.as_span()))?,
                        field2
                            .as_str()
                            .parse()
                            .map_err(|err| to_parse_err(err, field2.as_span()))?,
                        field3
                            .as_str()
                            .parse()
                            .map_err(|err| to_parse_err(err, field3.as_span()))?,
                    ));
                }
                Rule::EOI => (),
                _ => unreachable!(),
            }
        }

        Ok(Self { vertices })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let unparsed = fs::read_to_string(path)?;
        Self::from_string(unparsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_xyz_parser_basic() {
        let input = "
        0.15 +2.5 1.22
        1 5.9 -8
        ";

        let mesh = XYZMesh::from_string(input).unwrap();
        let vertices = mesh.vertices();

        assert_ulps_eq!(vertices[0].0, 0.15);
        assert_ulps_eq!(vertices[0].1, 2.5);
        assert_ulps_eq!(vertices[0].2, 1.22);
        assert_ulps_eq!(vertices[1].0, 1.0);
        assert_ulps_eq!(vertices[1].1, 5.9);
        assert_ulps_eq!(vertices[1].2, -8.);
    }

    #[test]
    fn test_xyz_parser_no_newline_before_eof() {
        let input = "
        0.15 +2.5 1.22
        1 5.9 -8";

        let mesh = XYZMesh::from_string(input).unwrap();
        let vertices = mesh.vertices();

        assert_ulps_eq!(vertices[0].0, 0.15);
        assert_ulps_eq!(vertices[0].1, 2.5);
        assert_ulps_eq!(vertices[0].2, 1.22);
        assert_ulps_eq!(vertices[1].0, 1.0);
        assert_ulps_eq!(vertices[1].1, 5.9);
        assert_ulps_eq!(vertices[1].2, -8.);
    }

    #[test]
    fn test_xyz_parser_with_spaces() {
        let input = "
        0.15     +2.5   1.22
        
        1             5.9 -8
        ";

        let mesh = XYZMesh::from_string(input).unwrap();
        let vertices = mesh.vertices();

        assert_ulps_eq!(vertices[0].0, 0.15);
        assert_ulps_eq!(vertices[0].1, 2.5);
        assert_ulps_eq!(vertices[0].2, 1.22);
        assert_ulps_eq!(vertices[1].0, 1.0);
        assert_ulps_eq!(vertices[1].1, 5.9);
        assert_ulps_eq!(vertices[1].2, -8.);
    }

    #[test]
    fn test_xyz_parser_wrong_syntax() {
        let input = "
        0.15 +2.5 1.22 5.3
        1 5.9 -8
        ";

        let mesh = XYZMesh::from_string(input);
        assert!(mesh.is_err());
    }
}
