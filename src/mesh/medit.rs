use super::ElementType;
use super::Mesh;
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
    BadDimension { expected: usize, found: usize },
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
            ErrorKind::BadDimension { expected, found } => {
                write!(f, "expected dimension to be {}, found {}", expected, found)
            }
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

fn parse_element_type(s: &str) -> Option<ElementType> {
    Some(match s {
        "edges" => ElementType::Edge,
        "triangles" => ElementType::Triangle,
        "quadrilaterals" => ElementType::Quadrangle,
        "quadrangles" => ElementType::Quadrangle,
        "tetrahedra" => ElementType::Tetrahedron,
        "hexahedra" => ElementType::Hexahedron,
        _ => return None,
    })
}

fn format_element_type(element_type: ElementType) -> &'static str {
    match element_type {
        ElementType::Vertex => "Vertices",
        ElementType::Edge => "Edges",
        ElementType::Triangle => "Triangles",
        ElementType::Quadrangle => "Quadrangles",
        ElementType::Tetrahedron => "Tetrahedra",
        ElementType::Hexahedron => "Hexahedra",
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

pub fn parse<R: io::BufRead, const D: usize>(mut input: R) -> Result<Mesh<D>, Error> {
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

    if dimension != D {
        return Err(Error {
            kind: ErrorKind::BadDimension {
                expected: D,
                found: dimension,
            },
            lineno: *lineno.borrow(),
        });
    }

    let mut nodes = Vec::new();
    let mut elements = Vec::new();

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
                nodes = Vec::with_capacity(dimension * num_vertices);

                for _ in 0..num_vertices {
                    let line = read(L).map_err(with_lineno(*lineno.borrow()))?;
                    let mut words = line.split_whitespace();
                    let mut node = [0.0; D];
                    for coord in node.iter_mut() {
                        *coord = words
                            .next()
                            .ok_or(io::ErrorKind::UnexpectedEof)?
                            .parse::<f64>()
                            .map_err(with_lineno(*lineno.borrow()))?;
                    }
                    nodes.push(node);
                    if let Some(word) = words.next() {
                        let _vertex_ref = word
                            .parse::<usize>()
                            .map_err(with_lineno(*lineno.borrow()))?;
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
            }
            element_type if parse_element_type(element_type).is_some() => {
                let element_type = parse_element_type(element_type).unwrap();
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

                let mut vertices = Vec::with_capacity(num_entries * element_type.num_nodes());

                for _ in 0..num_entries {
                    let line = read(L).map_err(with_lineno(*lineno.borrow()))?;
                    let mut words = line.split_whitespace();
                    let prev_len = vertices.len();
                    for word in (&mut words).take(element_type.num_nodes()) {
                        let col = word.parse::<usize>();
                        let col = col.map_err(with_lineno(*lineno.borrow()))?;
                        vertices.push(col - 1);
                    }
                    if vertices.len() - prev_len < element_type.num_nodes() {
                        let mut err = Error::from(io::ErrorKind::UnexpectedEof);
                        err.lineno = *lineno.borrow();
                        return Err(err);
                    }
                    if let Some(word) = words.next() {
                        let _element_ref = word
                            .parse::<usize>()
                            .map_err(with_lineno(*lineno.borrow()))?;
                    }
                }

                elements.push((element_type, vertices));
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
                })
            }
        }
    }
    Ok(Mesh { nodes, elements })
}

pub struct Display<'a, const D: usize> {
    pub mesh: &'a Mesh<D>,
}

impl<const D: usize> fmt::Display for Display<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeshVersionFormatted 2\nDimension {}\n\nVertices\n\t{}\n",
            D,
            self.mesh.nodes.len(),
        )?;
        for vertex in &self.mesh.nodes {
            for coordinate in vertex {
                write!(f, " {}", coordinate)?;
            }
            writeln!(f, " 0")?;
        }
        for (element_type, nodes) in &self.mesh.elements {
            let num_elements = nodes.len() / element_type.num_nodes();
            write!(
                f,
                "\n{}\n\t{}\n",
                format_element_type(*element_type),
                num_elements,
            )?;
            for element in nodes.chunks(element_type.num_nodes()) {
                for node in element {
                    write!(f, " {}", node + 1)?;
                }
                writeln!(f, " 0")?;
            }
        }
        write!(f, "\nEnd")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let input = "MeshVersionFormatted 2
Dimension 3

Vertices
\t4
 2.3 0 1 0
 1231 2 3.14 0
 -21.2 21 0.0001 0
 -0.2 -0.2 -0.2 0

Triangles
\t2
 1 2 3 0
 2 3 4 0

End";
        let input = io::Cursor::new(input);
        let mesh = parse(input).unwrap();
        let output = format(&mesh);
        assert_eq!(input, output);
    }

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
        let _mesh = input.parse::<MeditMesh>().unwrap();
    }
}
