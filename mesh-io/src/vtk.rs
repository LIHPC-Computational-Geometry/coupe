use crate::ElementType;
use crate::Mesh;
use crate::Ref;
use std::fmt;
use std::io;
use std::iter;
use vtkio::Error;
use vtkio::IOBuffer;
use vtkio::Vtk;
use vtkio::model::Attribute;
use vtkio::model::Attributes;
use vtkio::model::ByteOrder;
use vtkio::model::CellType;
use vtkio::model::Cells;
use vtkio::model::DataArray;
use vtkio::model::DataSet;
use vtkio::model::Piece;
use vtkio::model::UnstructuredGridPiece;
use vtkio::model::Version;
use vtkio::model::VertexNumbers;
use vtkio::writer::WriteVtk;

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

impl ElementType {
    fn into_cell_type(self) -> CellType {
        match self {
            Self::Vertex => CellType::Vertex,
            Self::Edge => CellType::Line,
            Self::Triangle => CellType::Triangle,
            Self::Quadrangle | Self::Quadrilateral => CellType::Quad,
            Self::Tetrahedron => CellType::Tetra,
            Self::Hexahedron => CellType::Hexahedron,
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
        .extend(std::iter::repeat_n(1, piece.num_points()));
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

impl Mesh {
    fn to_vtk(&self) -> Option<Vtk> {
        let dimension = self.dimension();
        if u32::try_from(self.node_count()).is_err()
            || u32::try_from(self.element_count()).is_err()
            || dimension > 3
        {
            return None;
        }

        let points = self
            .nodes()
            .flat_map(|(coords, _)| coords.iter().cloned().chain(iter::repeat(0.0)).take(3))
            .collect();
        let cell_verts = VertexNumbers::Legacy {
            num_cells: self.element_count() as u32,
            vertices: self
                .elements()
                .flat_map(|(element_type, nodes, _)| {
                    iter::once(element_type.node_count() as u32)
                        .chain(nodes.iter().map(|node| *node as u32))
                })
                .collect(),
        };
        let types = self
            .elements()
            .map(|(element_type, _, _)| element_type.into_cell_type())
            .collect();
        let point_refs = DataArray {
            name: String::from("node_refs"),
            elem: vtkio::model::ElementType::Scalars {
                num_comp: 1,
                lookup_table: None,
            },
            data: IOBuffer::I64(
                self.node_refs()
                    .iter()
                    .map(|node_ref| *node_ref as i64)
                    .collect(),
            ),
        };
        let cell_refs = DataArray {
            name: String::from("cell_refs"),
            elem: vtkio::model::ElementType::Scalars {
                num_comp: 1,
                lookup_table: None,
            },
            data: IOBuffer::I64(
                self.elements()
                    .map(|(_, _, element_ref)| element_ref as i64)
                    .collect(),
            ),
        };
        let piece = UnstructuredGridPiece {
            points: IOBuffer::F64(points),
            cells: Cells { cell_verts, types },
            data: Attributes {
                point: vec![Attribute::DataArray(point_refs)],
                cell: vec![Attribute::DataArray(cell_refs)],
            },
        };
        Some(Vtk {
            version: Version::new((2, 0)),
            title: String::new(),
            byte_order: ByteOrder::BigEndian,
            data: DataSet::UnstructuredGrid {
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
            file_path: None,
        })
    }
}

/// Deserialize a mesh into the ASCII, legacy VTK format.
///
/// This type implements [`Display`](fmt::Display).
#[derive(Debug)]
pub struct DisplayAscii<'a> {
    mesh: &'a Mesh,
}

impl Mesh {
    pub fn display_vtk_ascii(&self) -> DisplayAscii<'_> {
        DisplayAscii { mesh: self }
    }
}

impl fmt::Display for DisplayAscii<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vtk = self.mesh.to_vtk().ok_or(fmt::Error)?;
        vtkio::writer::AsciiWriter(f)
            .write_vtk_be(vtk)
            .map_err(|_| fmt::Error)?;
        Ok(())
    }
}

impl Mesh {
    pub fn serialize_vtk_binary<W>(&self, w: W) -> io::Result<()>
    where
        W: io::Write,
    {
        fn writer_err_to_io(err: vtkio::writer::Error) -> io::Error {
            match err {
                vtkio::writer::Error::IOError(err) => io::Error::from(err),
                _ => io::Error::other(err),
            }
        }

        let vtk = self.to_vtk().ok_or(io::ErrorKind::InvalidInput)?;
        vtkio::writer::BinaryWriter(w)
            .write_vtk_be(vtk)
            .map_err(writer_err_to_io)?;
        Ok(())
    }
}
