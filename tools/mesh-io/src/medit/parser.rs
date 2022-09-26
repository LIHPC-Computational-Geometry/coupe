use super::code;
use crate::ElementType;
use crate::Mesh;
use std::cell;
use std::error;
use std::fmt;
use std::io;
use std::mem;
use std::num;
use std::str;

#[derive(Debug)]
pub enum ErrorKind {
    UnexpectedToken { expected: String, found: String },
    Io(io::Error),
    BadInteger(num::ParseIntError),
    BadFloat(num::ParseFloatError),
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    lineno: usize,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedToken { expected, found } => {
                write!(f, "expected token {:?}, found {}", expected, found)
            }
            ErrorKind::Io(err) => write!(f, "io error: {}", err),
            ErrorKind::BadInteger(err) => write!(f, "when parsing integer: {}", err),
            ErrorKind::BadFloat(err) => write!(f, "when parsing float: {}", err),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at line {}: {}", self.lineno, self.kind)
    }
}

impl error::Error for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error {
            kind: ErrorKind::Io(err),
            lineno: 0,
        }
    }
}

impl From<io::ErrorKind> for Error {
    fn from(err: io::ErrorKind) -> Error {
        Error {
            kind: ErrorKind::Io(io::Error::from(err)),
            lineno: 0,
        }
    }
}

impl From<num::ParseIntError> for Error {
    fn from(err: num::ParseIntError) -> Error {
        Error {
            kind: ErrorKind::BadInteger(err),
            lineno: 0,
        }
    }
}

impl From<num::ParseFloatError> for Error {
    fn from(err: num::ParseFloatError) -> Error {
        Error {
            kind: ErrorKind::BadFloat(err),
            lineno: 0,
        }
    }
}

impl str::FromStr for ElementType {
    type Err = ();

    fn from_str(s: &str) -> Result<ElementType, ()> {
        Ok(match s {
            "edges" => ElementType::Edge,
            "triangles" => ElementType::Triangle,
            "quadrilaterals" => ElementType::Quadrilateral,
            "quadrangles" => ElementType::Quadrangle,
            "tetrahedra" => ElementType::Tetrahedron,
            "hexahedra" => ElementType::Hexahedron,
            _ => return Err(()),
        })
    }
}

impl ElementType {
    fn from_code(code: i64) -> Option<Self> {
        Some(match code {
            code::EDGE => Self::Edge,
            code::TRIANGLE => Self::Triangle,
            code::QUAD => Self::Quadrilateral,
            code::TETRAHEDRON => Self::Tetrahedron,
            code::HEXAHEDRON => Self::Hexahedron,
            _ => return None,
        })
    }
}

/// a token separator
fn is_separator(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\r' || b == b'\n'
}

/// BufRead::consume/trim all separators found at the begining of the reader.
fn skip_separators<R: io::BufRead>(lineno: &mut usize, mut r: R) -> io::Result<()> {
    loop {
        let buf = r.fill_buf()?;
        let n = buf.len();
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }

        let num_separators = buf
            .iter()
            .enumerate()
            .inspect(|&(_i, &b)| {
                if b == b'\n' {
                    *lineno += 1;
                }
            })
            .find(|&(_i, &b)| !is_separator(b))
            .map_or(n, |(i, &_b)| i);
        r.consume(num_separators);
        if num_separators < n {
            // Reached a non-separator byte.
            break;
        }
    }

    Ok(())
}

/// Like BufRead::read_line, except it reads til a separator byte.
fn read_token<R: io::BufRead>(token: &mut String, mut r: R) -> io::Result<()> {
    let original_len = token.len();
    loop {
        let buf = r.fill_buf()?;
        let n = buf.len();
        if n == 0 {
            if original_len == token.len() {
                // Asked for a token, got EOF.
                return Err(io::ErrorKind::UnexpectedEof.into());
            } else {
                // This token is the last.
                return Ok(());
            }
        }

        let token_size = buf
            .iter()
            .enumerate()
            .find(|&(_i, &b)| is_separator(b))
            .map_or(n, |(i, &_b)| i);
        let buf = std::str::from_utf8(&buf[..token_size])
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err.to_string()))?;
        token.push_str(buf);
        r.consume(token_size);
        if token_size < n {
            // Reached a separator.
            break;
        }
    }

    Ok(())
}

fn with_lineno<E>(lineno: usize) -> impl Fn(E) -> Error
where
    E: Into<Error>,
{
    move |err: E| {
        let mut err = err.into();
        err.lineno = lineno;
        err
    }
}

pub fn parse_ascii<R: io::BufRead>(mut input: R) -> Result<Mesh, Error> {
    enum Read {
        T,
        L,
    }
    use Read::*;

    let token = cell::RefCell::new(String::new());
    let lineno = cell::RefCell::new(1);
    // the previous result of this function must be dropped before the next
    // call, otherwise it will panic.
    let mut read = |what: Read| -> Result<cell::Ref<'_, String>, Error> {
        token.borrow_mut().clear();
        skip_separators(&mut lineno.borrow_mut(), &mut input)?;
        match what {
            T => {
                read_token(&mut token.borrow_mut(), &mut input)?;
                token.borrow_mut().make_ascii_lowercase();
            }
            L => {
                input.read_line(&mut token.borrow_mut())?;
                *lineno.borrow_mut() += 1;
            }
        }
        Ok(token.borrow())
    };

    let header = read(T)?;
    if header.as_str() != "meshversionformatted" {
        return Err(Error {
            kind: ErrorKind::UnexpectedToken {
                found: header.to_owned(),
                expected: String::from("MeshVersionFormatted"),
            },
            lineno: *lineno.borrow(),
        });
    }
    mem::drop(header); // drop the borrow on token

    let _version_number = read(T)
        .map_err(with_lineno(*lineno.borrow()))?
        .parse::<usize>()
        .map_err(with_lineno(*lineno.borrow()))?;

    let dimension_keyword = read(T).map_err(with_lineno(*lineno.borrow()))?;
    if dimension_keyword.as_str() != "dimension" {
        return Err(Error {
            kind: ErrorKind::UnexpectedToken {
                found: dimension_keyword.to_owned(),
                expected: String::from("Dimension"),
            },
            lineno: *lineno.borrow(),
        });
    }
    mem::drop(dimension_keyword); // drop the borrow on token

    let dimension = read(T)
        .map_err(with_lineno(*lineno.borrow()))?
        .parse::<usize>()
        .map_err(with_lineno(*lineno.borrow()))?;

    let mut mesh = Mesh {
        dimension,
        ..Default::default()
    };

    loop {
        let section = read(T).map_err(with_lineno(*lineno.borrow()))?;

        match section.as_str() {
            "end" => break,
            "vertices" => {
                mem::drop(section); // drop the borrow on token

                let num_vertices = read(T)
                    .map_err(with_lineno(*lineno.borrow()))?
                    .parse::<usize>()
                    .map_err(with_lineno(*lineno.borrow()))?;
                let mut coords = Vec::with_capacity(dimension * num_vertices);
                let mut node_refs = Vec::with_capacity(num_vertices);

                for _ in 0..num_vertices {
                    let line = read(L).map_err(with_lineno(*lineno.borrow()))?;
                    let mut words = line.split_whitespace();
                    for _ in 0..dimension {
                        let c = words
                            .next()
                            .ok_or(io::ErrorKind::UnexpectedEof)?
                            .parse::<f64>()
                            .map_err(with_lineno(*lineno.borrow()))?;
                        coords.push(c);
                    }
                    if let Some(word) = words.next() {
                        let node_ref = word
                            .parse::<isize>()
                            .map_err(with_lineno(*lineno.borrow()))?;
                        node_refs.push(node_ref);
                    }
                    if let Some(word) = words.next() {
                        // TODO error type
                        return Err(Error {
                            kind: ErrorKind::UnexpectedToken {
                                found: String::from(word),
                                expected: String::from("newline"),
                            },
                            lineno: *lineno.borrow(),
                        });
                    }
                }

                mesh.coordinates = coords;
                mesh.node_refs = node_refs;
            }
            element_type if element_type.parse::<ElementType>().is_ok() => {
                let element_type = element_type.parse::<ElementType>().unwrap();
                mem::drop(section); // drop the borrow on token

                let prev_lineno = *lineno.borrow();
                let num_entries = loop {
                    let is_this_num_entries_yet = read(T).map_err(with_lineno(*lineno.borrow()))?;
                    match is_this_num_entries_yet.parse::<usize>() {
                        Ok(n) => break n,
                        Err(err) if prev_lineno != *lineno.borrow() => {
                            // This is on the next line, let's kindly ask the
                            // user to fix its own mess.
                            return Err(Error {
                                kind: ErrorKind::BadInteger(err),
                                lineno: prev_lineno,
                            });
                        }
                        Err(_) => {} // skip junk
                    }
                };

                let mut vertices = Vec::with_capacity(num_entries * element_type.node_count());
                let mut refs = Vec::with_capacity(num_entries);

                for _ in 0..num_entries {
                    let line = read(L).map_err(with_lineno(*lineno.borrow()))?;
                    let mut words = line.split_whitespace();
                    let prev_len = vertices.len();
                    for word in (&mut words).take(element_type.node_count()) {
                        let col = word.parse::<usize>();
                        let col = col.map_err(with_lineno(*lineno.borrow()))?;
                        vertices.push(col - 1);
                    }
                    if vertices.len() - prev_len < element_type.node_count() {
                        let mut err = Error::from(io::ErrorKind::UnexpectedEof);
                        err.lineno = *lineno.borrow();
                        return Err(err);
                    }
                    if let Some(word) = words.next() {
                        let element_ref = word
                            .parse::<isize>()
                            .map_err(with_lineno(*lineno.borrow()))?;
                        refs.push(element_ref);
                    }
                }

                mesh.topology.push((element_type, vertices, refs));
            }
            "corners" | "ridges" | "requiredvertices" => {
                mem::drop(section); // drop the borrow on token
                let num_entries = read(T)
                    .map_err(with_lineno(*lineno.borrow()))?
                    .parse::<usize>()
                    .map_err(with_lineno(*lineno.borrow()))?;
                for _ in 0..num_entries {
                    let _ = read(L).map_err(with_lineno(*lineno.borrow()))?; // skip lines
                }
            }
            unexpected_token => {
                return Err(Error {
                    kind: ErrorKind::UnexpectedToken {
                        found: unexpected_token.to_owned(),
                        expected: String::from("element"),
                    },
                    lineno: *lineno.borrow(),
                });
            }
        }
    }
    Ok(mesh)
}

pub fn test_format_ascii(header: &[u8]) -> bool {
    const HEADER: &str = "meshversionformatted";
    if header.len() < HEADER.len() {
        return false;
    }
    let header = match std::str::from_utf8(&header[..HEADER.len()]) {
        Ok(v) => v,
        Err(_) => return false,
    };
    header.eq_ignore_ascii_case(HEADER)
}

pub fn test_format_binary(header: &[u8]) -> bool {
    if header.len() < 4 {
        return false;
    }
    let binary_magic = &header[..4];
    binary_magic == [1, 0, 0, 0] || binary_magic == [0, 0, 0, 1]
}

// Taken from nschloe's meshio[0] implementation. It seems medit binary files
// are encoded like so:
//
//     file    := magic version *field
//     magic   := %x01  ; magic byte, to tell whether the file is encoded in
//                      ; little- or big-endian.
//     version := %x01-04  ; sets the size of ints, floats and bitpos
//     field   := code next *1( count )  values
//     code    := key    ; %x03 for dimension, %x04 for vertices, etc.
//     next    := bitpos ; the bit position of the next field code in the file
//     count   := int    ; the number of values that follows
//     values  := ...    ; depends on the field. For the dimension, it's a key,
//                       ; for vertices it's a list of floats and ints.
//     key     := i32       ; type for codes and the version
//     bitpos  := i32 / i64 ; depends on the file version
//     int     := i32 / i64 ; depends on the file version
//     float   := f32 / f64 ; depends on the file version
//
// [0] https://github.com/nschloe/meshio
pub fn parse_binary<R: io::BufRead>(mut input: R) -> Result<Mesh, Error> {
    let mut magic = [0; 4];
    input.read_exact(&mut magic)?;
    let magic = u32::from_le_bytes(magic);
    let little_endian = if magic == 1 {
        true
    } else if magic == 1 << 24 {
        false
    } else {
        return Err(Error {
            kind: ErrorKind::UnexpectedToken {
                found: magic.to_string(),
                expected: "1".to_string(),
            },
            lineno: 0,
        });
    };

    macro_rules! up {
        ( i32 ) => {
            i64
        };
        ( i64 ) => {
            i64
        };
        ( f32 ) => {
            f64
        };
        ( f64 ) => {
            f64
        };
    }
    macro_rules! read_fn {
        ( $little_endian:expr, $type_:ident ) => {
            if $little_endian {
                |input: &mut R| -> io::Result<up!($type_)> {
                    let mut buf = [0; std::mem::size_of::<$type_>()];
                    input.read_exact(&mut buf)?;
                    Ok($type_::from_le_bytes(buf) as up!($type_))
                }
            } else {
                |input: &mut R| -> io::Result<up!($type_)> {
                    let mut buf = [0; std::mem::size_of::<$type_>()];
                    input.read_exact(&mut buf)?;
                    Ok($type_::from_be_bytes(buf) as up!($type_))
                }
            }
        };
    }

    let read_key = read_fn!(little_endian, i32);

    let version = read_key(&mut input)?;
    let read_int;
    let read_float;
    let read_pos;
    match version {
        1 => {
            read_int = read_fn!(little_endian, i32);
            read_float = read_fn!(little_endian, f32);
            read_pos = read_fn!(little_endian, i32);
        }
        2 => {
            read_int = read_fn!(little_endian, i32);
            read_float = read_fn!(little_endian, f64);
            read_pos = read_fn!(little_endian, i32);
        }
        3 => {
            read_int = read_fn!(little_endian, i32);
            read_float = read_fn!(little_endian, f64);
            read_pos = read_fn!(little_endian, i64);
        }
        4 => {
            read_int = read_fn!(little_endian, i64);
            read_float = read_fn!(little_endian, f64);
            read_pos = read_fn!(little_endian, i64);
        }
        _ => {
            return Err(Error {
                kind: ErrorKind::UnexpectedToken {
                    found: version.to_string(),
                    expected: "4".to_string(),
                },
                lineno: 0,
            })
        }
    }

    let dimension_code = read_key(&mut input)?;
    if dimension_code != code::DIMENSION {
        return Err(Error {
            kind: ErrorKind::UnexpectedToken {
                found: dimension_code.to_string(),
                expected: "3".to_string(),
            },
            lineno: 0,
        });
    }

    let _ = read_pos(&mut input)?;

    let dimension = read_key(&mut input)? as usize;
    let mut mesh = Mesh {
        dimension,
        ..Mesh::default()
    };

    loop {
        let code = match read_key(&mut input) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        };
        match code {
            code::END => break,
            code::VERTEX => {
                let _ = read_pos(&mut input)?;
                let vertex_count = read_int(&mut input)? as usize;
                mesh.coordinates = Vec::with_capacity(vertex_count * dimension);
                mesh.node_refs = Vec::with_capacity(vertex_count);
                for _ in 0..vertex_count {
                    for _ in 0..dimension {
                        mesh.coordinates.push(read_float(&mut input)?);
                    }
                    mesh.node_refs.push(read_int(&mut input)? as isize);
                }
            }
            code => {
                let element_type = match ElementType::from_code(code) {
                    Some(v) => v,
                    None => {
                        return Err(Error {
                            kind: ErrorKind::UnexpectedToken {
                                found: code.to_string(),
                                expected: "54".to_string(),
                            },
                            lineno: 0,
                        })
                    }
                };
                let _ = read_pos(&mut input)?;
                let element_count = read_int(&mut input)? as usize;
                let nodes_per_element = element_type.node_count();
                let mut nodes = Vec::with_capacity(nodes_per_element * element_count);
                let mut refs = Vec::with_capacity(element_count);
                for _ in 0..element_count {
                    for _ in 0..nodes_per_element {
                        nodes.push(read_int(&mut input)? as usize - 1);
                    }
                    refs.push(read_int(&mut input)? as isize);
                }
                mesh.topology.push((element_type, nodes, refs));
            }
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let input = "MeshVersionFormatted 1
        Dimension 3
        Vertices
        4
        2.3 0.0 1 0
        1231 2.00 3.14 0
        -21.2 21 0.0001 0
        -0.2 -0.2 -0.2 0
        Triangles
        2
        1 2 3 0
        2 3 4 1
        End
        ";
        let _mesh = input.parse::<Mesh>().unwrap();
    }
}
