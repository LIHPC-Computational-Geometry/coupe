use super::{ElementType, MeditMesh};
use std::fmt;

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

impl fmt::Display for MeditMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeshVersionFormatted 2\nDimension {}\n\nVertices\n\t{}\n",
            self.dimension,
            self.num_nodes(),
        )?;
        for vertex in self.coordinates.chunks(self.dimension) {
            for coordinate in vertex {
                write!(f, " {}", coordinate)?;
            }
            writeln!(f, " 0")?;
        }
        for (element_type, nodes) in &self.topology {
            let num_elements = nodes.len() / element_type.num_nodes();
            write!(f, "\n{}\n\t{}\n", element_type, num_elements)?;
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
        let output = input.parse::<MeditMesh>().unwrap().to_string();
        assert_eq!(input, output);
    }
}
