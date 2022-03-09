use super::{ElementType, Mesh};

use std::cell;
use std::error;
use std::fmt;
use std::fs;
use std::io;
use std::mem;
use std::num;
use std::path;
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

pub fn parse<R: io::BufRead>(mut input: R) -> Result<Mesh, Error> {
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

impl Mesh {
    /// Import from a medit mesh file.
    pub fn from_file(path: impl AsRef<path::Path>) -> Result<Mesh, Error> {
        let file = fs::File::open(path)?;
        parse(io::BufReader::new(file))
    }

    pub fn from_reader(r: impl io::BufRead) -> Result<Mesh, Error> {
        parse(r)
    }
}

impl str::FromStr for Mesh {
    type Err = Error;

    fn from_str(s: &str) -> Result<Mesh, Error> {
        parse(io::Cursor::new(s))
    }
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
