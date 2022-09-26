use super::code;
use crate::ElementType;
use crate::Mesh;
use std::fmt;
use std::io;

/// Deserialize a mesh into the ASCII MEDIT format.
#[derive(Debug)]
pub struct DisplayAscii<'a> {
    mesh: &'a Mesh,
}

impl Mesh {
    pub fn display_medit_ascii(&self) -> DisplayAscii<'_> {
        DisplayAscii { mesh: self }
    }
}

struct AsciiElementType(ElementType);

impl fmt::Display for AsciiElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ElementType::Vertex => write!(f, "Vertices"),
            ElementType::Edge => write!(f, "Edges"),
            ElementType::Triangle => write!(f, "Triangles"),
            ElementType::Quadrangle => write!(f, "Quadrangles"),
            ElementType::Quadrilateral => write!(f, "Quadrilaterals"),
            ElementType::Tetrahedron => write!(f, "Tetrahedra"),
            ElementType::Hexahedron => write!(f, "Hexahedra"),
        }
    }
}

impl ElementType {
    fn code(self) -> i64 {
        match self {
            ElementType::Vertex => code::VERTEX,
            ElementType::Edge => code::EDGE,
            ElementType::Triangle => code::TRIANGLE,
            ElementType::Quadrangle | ElementType::Quadrilateral => code::QUAD,
            ElementType::Tetrahedron => code::TETRAHEDRON,
            ElementType::Hexahedron => code::HEXAHEDRON,
        }
    }
}

impl fmt::Display for DisplayAscii<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeshVersionFormatted 2\nDimension {}\n\nVertices\n\t{}\n",
            self.mesh.dimension,
            self.mesh.node_count(),
        )?;
        for (coordinates, node_ref) in self.mesh.nodes() {
            for coordinate in coordinates {
                write!(f, " {}", coordinate)?;
            }
            writeln!(f, " {}", node_ref)?;
        }
        for (element_type, nodes, refs) in &self.mesh.topology {
            if *element_type == ElementType::Vertex {
                // Breaks MEDIT and meshio-py.
                continue;
            }
            let element_count = refs.len();
            write!(
                f,
                "\n{}\n\t{}\n",
                AsciiElementType(*element_type),
                element_count,
            )?;
            for (element, element_ref) in nodes.chunks(element_type.node_count()).zip(refs) {
                for node in element {
                    write!(f, " {}", node + 1)?;
                }
                writeln!(f, " {}", element_ref)?;
            }
        }
        write!(f, "\nEnd")
    }
}

impl Mesh {
    pub fn serialize_medit_binary<W: io::Write>(&self, mut w: W) -> io::Result<()> {
        // Header
        w.write_all(&i32::to_le_bytes(1))?; // magic code
        w.write_all(&i32::to_le_bytes(4))?; // version
        w.write_all(&i32::to_le_bytes(code::DIMENSION as i32))?;
        let mut bitpos = 4 + 4 + 4 + 8 + 4; // byte position of the vertices
        w.write_all(&i64::to_le_bytes(bitpos as i64))?;
        w.write_all(&i32::to_le_bytes(self.dimension as i32))?;

        // Nodes
        w.write_all(&i32::to_le_bytes(code::VERTEX as i32))?;
        let node_count = self.node_refs.len();
        bitpos += 8 * node_count * (self.dimension + 1); // +1 for the int ref
        bitpos += 4 + 8 + 8; // take into account the code, bitpos and node_count
        w.write_all(&i64::to_le_bytes(bitpos as i64))?;
        w.write_all(&i64::to_le_bytes(node_count as i64))?;
        for (coordinates, node_ref) in self.nodes() {
            for coordinate in coordinates {
                w.write_all(&f64::to_le_bytes(*coordinate))?;
            }
            w.write_all(&i64::to_le_bytes(node_ref as i64))?;
        }

        // Elements
        for (element_type, nodes, refs) in &self.topology {
            w.write_all(&i32::to_le_bytes(element_type.code() as i32))?;
            let element_count = refs.len();
            let nodes_per_element = element_type.node_count();
            bitpos += 8 * element_count * (nodes_per_element + 1); // +1 for the int ref
            bitpos += 4 + 8 + 8; // take into account the code, bitpos and element_count
            w.write_all(&i64::to_le_bytes(bitpos as i64))?;
            w.write_all(&i64::to_le_bytes(element_count as i64))?;
            for (element, element_ref) in nodes.chunks(nodes_per_element).zip(refs) {
                for node in element {
                    w.write_all(&i64::to_le_bytes(*node as i64 + 1))?;
                }
                w.write_all(&i64::to_le_bytes(*element_ref as i64))?;
            }
        }

        w.write_all(&i32::to_le_bytes(54))?; // End

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse_ascii;

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
        let mesh = parse_ascii(input.as_bytes()).unwrap();
        let output = mesh.display_medit_ascii().to_string();
        assert_eq!(input, output);
    }
}
