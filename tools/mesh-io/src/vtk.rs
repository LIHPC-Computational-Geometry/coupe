use crate::ElementType;
use crate::Mesh;
use crate::Ref;
use std::io;
use vtkio::model::CellType;
use vtkio::model::DataSet;
use vtkio::model::Piece;
use vtkio::model::UnstructuredGridPiece;
use vtkio::model::VertexNumbers;
use vtkio::Error;
use vtkio::Vtk;

const DIMENSION: usize = 3;

pub fn test_format_legacy(header: &[u8]) -> bool {
    let header_start = match header.iter().position(|b| *b == b'#') {
        Some(v) => v,
        None => return false,
    };
    let junk = &header[..header_start];
    let header = &header[header_start..];

    if junk
        .iter()
        .any(|b| *b != b' ' || *b != b'\n' || *b != b'\r')
    {
        return false;
    }
    let header_end = match header.iter().position(|b| *b == b'\n') {
        Some(v) => v,
        None => return false,
    };
    let header = match std::str::from_utf8(&header[..header_end]) {
        Ok(v) => v,
        Err(_) => return false,
    };
    header.starts_with("# vtk DataFile")
}

impl From<CellType> for ElementType {
    fn from(ct: CellType) -> Self {
        match ct {
            CellType::Vertex => ElementType::Vertex,
            CellType::Line => ElementType::Edge,
            CellType::Triangle => ElementType::Triangle,
            CellType::Quad => ElementType::Quadrilateral,
            CellType::Tetra => ElementType::Tetrahedron,
            CellType::Hexahedron => ElementType::Hexahedron,
            _ => todo!(),
        }
    }
}

fn add_element<I>(mesh: &mut Mesh, element_type: ElementType, vertices: I, element_ref: Ref)
where
    I: IntoIterator<Item = usize>,
{
    let entry = match mesh
        .topology
        .iter_mut()
        .find(|(el_type, _, _)| *el_type == element_type)
    {
        Some(entry) => entry,
        None => {
            mesh.topology.push((element_type, Vec::new(), Vec::new()));
            mesh.topology.last_mut().unwrap()
        }
    };
    entry.1.extend(vertices);
    entry.2.push(element_ref);
}

fn add_piece(mesh: &mut Mesh, piece: UnstructuredGridPiece) {
    let vertex_offset = mesh.coordinates.len() / DIMENSION;
    mesh.coordinates.extend(piece.points.iter().unwrap());
    // TODO extract attributes
    mesh.node_refs
        .extend(std::iter::repeat(1).take(piece.num_points()));
    match piece.cells.cell_verts {
        VertexNumbers::Legacy {
            num_cells,
            vertices,
        } => {
            assert_eq!(num_cells as usize, piece.cells.types.len());
            let mut vertices = vertices.into_iter();
            for cell_type in piece.cells.types {
                let vertex_count = vertices.next().unwrap() as usize;
                let vertices = (&mut vertices)
                    .take(vertex_count)
                    .map(|v| v as usize + vertex_offset);
                let element_type = ElementType::from(cell_type);
                let element_ref = 1; // TODO
                add_element(mesh, element_type, vertices, element_ref)
            }
        }
        VertexNumbers::XML { .. } => todo!(),
    }
}

pub fn parse_legacy<R: io::BufRead>(input: R) -> Result<Mesh, Error> {
    let vtk = Vtk::parse_legacy_be(input)?;
    let mut mesh = Mesh::new(DIMENSION);
    match vtk.data {
        DataSet::ImageData { .. } => todo!(),
        DataSet::StructuredGrid { .. } => todo!(),
        DataSet::RectilinearGrid { .. } => todo!(),
        DataSet::UnstructuredGrid { pieces, .. } => {
            for piece in pieces {
                match piece {
                    Piece::Source(_, _) => todo!(),
                    Piece::Loaded(_) => todo!(),
                    Piece::Inline(piece) => add_piece(&mut mesh, *piece),
                }
            }
        }
        DataSet::PolyData { .. } => todo!(),
        DataSet::Field { .. } => todo!(),
    }
    Ok(mesh)
}
