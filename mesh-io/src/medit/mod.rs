//! This module allows to load Medit mesh files, as
//! described by Frey in
//! [MEDIT : An interactive Mesh visualization Software](https://hal.inria.fr/inria-00069921).

pub use parser::Error as ParseError;
pub use parser::parse_ascii;
pub use parser::parse_binary;
pub use parser::test_format_ascii;
pub use parser::test_format_binary;
pub use serializer::DisplayAscii;

mod parser;
mod serializer;
mod code {
    pub const DIMENSION: i64 = 3;
    pub const VERTEX: i64 = 4;
    pub const EDGE: i64 = 5;
    pub const TRIANGLE: i64 = 6;
    pub const QUAD: i64 = 7;
    pub const TETRAHEDRON: i64 = 8;
    pub const HEXAHEDRON: i64 = 9;
    pub const END: i64 = 54;
}
