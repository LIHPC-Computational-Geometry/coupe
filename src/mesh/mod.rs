//! Mesh I/O - Read and write meshes to files and al.
//!
//! See the [Mesh] struct for more details.

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::str;

mod medit;
mod vtk;

/// Type of a mesh element (also called cell).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ElementType {
    Edge,
    Triangle,
    Quadrangle,
    Tetrahedron,
    Hexahedron,
}

impl ElementType {
    /// The number of nodes involved in the definition of this type of element.
    pub fn num_nodes(self) -> usize {
        match self {
            ElementType::Edge => 2,
            ElementType::Triangle => 3,
            ElementType::Quadrangle | ElementType::Tetrahedron => 4,
            ElementType::Hexahedron => 8,
        }
    }
}

/// An element (triangle, hexahedron, ...) of a mesh.
///
/// - `D` is the dimension of the mesh.
/// - `N` is the number of nodes in the element.
pub struct Element<'a, const D: usize, const N: usize> {
    /// Reference to the mesh, for the nodes' coordinates.
    mesh: &'a Mesh<D>,
    node_ids: [usize; N],
}

impl<const D: usize, const N: usize> Element<'_, D, N> {
    pub fn as_node_ids(&self) -> &[usize; N] {
        &self.node_ids
    }

    pub fn into_node_ids(self) -> [usize; N] {
        self.node_ids
    }

    pub fn nodes(&self) -> impl Iterator<Item = [f64; D]> + ExactSizeIterator + '_ {
        self.node_ids
            .into_iter()
            .map(move |&node_id| self.mesh.node(node_id))
    }

    pub fn center(&self) -> [f64; D] {
        let mut center = [0.0; D];
        for node in self.nodes() {
            for (c, n) in center.iter_mut().zip(node) {
                *c += n;
            }
        }
        for c in center.iter_mut() {
            *c /= N as f64;
        }
        center
    }
}

pub struct Edge<'a, const D: usize>(pub Element<'a, D, 2>);

pub struct Triangle<'a, const D: usize>(pub Element<'a, D, 3>);

impl<'a, const D: usize> Triangle<'a, D> {
    pub fn edges(&self) -> [Edge<'a, D>; 3] {
        let [a, b, c] = self.0.node_ids;
        [
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [a, b],
            }),
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [b, c],
            }),
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [c, a],
            }),
        ]
    }
}

pub struct Quadrangle<'a, const D: usize>(pub Element<'a, D, 4>);

impl<'a, const D: usize> Quadrangle<'a, D> {
    pub fn edges(&self) -> [Edge<'a, D>; 4] {
        let [a, b, c, d] = self.0.node_ids;
        [
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [a, b],
            }),
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [b, c],
            }),
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [c, d],
            }),
            Edge(Element {
                mesh: self.0.mesh,
                node_ids: [d, a],
            }),
        ]
    }
}

pub struct Tetrahedron<'a, const D: usize>(pub Element<'a, D, 4>);

impl<'a, const D: usize> Tetrahedron<'a, D> {
    pub fn edges(&self) -> [Edge<'a, D>; 6] {
        // TODO
        todo!()
    }

    pub fn faces(&self) -> [Triangle<'a, D>; 4] {
        // TODO
        todo!()
    }
}

pub struct Hexahedron<'a, const D: usize>(pub Element<'a, D, 8>);

impl<'a, const D: usize> Hexahedron<'a, D> {
    pub fn edges(&self) -> [Edge<'a, D>; 8] {
        // TODO
        todo!()
    }

    pub fn faces(&self) -> [Quadrangle<'a, D>; 6] {
        // TODO
        todo!()
    }
}

/// Main struct for a mesh.
pub struct Mesh<const D: usize> {
    nodes: Vec<[f64; D]>,
    elements: Vec<(ElementType, Vec<usize>)>,
}

/// Data methods.
impl<const D: usize> Mesh<D> {
    /// Iterator over the nodes of the mesh.
    pub fn nodes(&self) -> impl Iterator<Item = [f64; D]> + ExactSizeIterator + '_ {
        self.nodes.iter().cloned()
    }

    /// A single node of the mesh.
    ///
    /// # Panics
    ///
    /// This function will panic iff `node_id` is greater or equal to the
    /// number of nodes.
    pub fn node(&self, node_id: usize) -> [f64; D] {
        self.nodes[node_id]
    }

    /// An iterator over the triangles present in the mesh.
    pub fn edges(&self) -> Option<impl Iterator<Item = Edge<'_, D>> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Edge)?;
        debug_assert_eq!(node_ids.len() % 2, 0);
        Some(node_ids.array_chunks().cloned().map(move |node_ids| {
            Edge(Element {
                mesh: self,
                node_ids,
            })
        }))
    }

    /// An iterator over the triangles present in the mesh.
    pub fn triangles(
        &self,
    ) -> Option<impl Iterator<Item = Triangle<'_, D>> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Triangle)?;
        debug_assert_eq!(node_ids.len() % 3, 0);
        Some(node_ids.array_chunks().cloned().map(move |node_ids| {
            Triangle(Element {
                mesh: self,
                node_ids,
            })
        }))
    }

    /// An iterator over the quadrangles present in the mesh.
    pub fn quadrangles(
        &self,
    ) -> Option<impl Iterator<Item = Quadrangle<'_, D>> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Quadrangle)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned().map(move |node_ids| {
            Quadrangle(Element {
                mesh: self,
                node_ids,
            })
        }))
    }

    /// An iterator over the tetrahedra present in the mesh.
    pub fn tetrahedra(
        &self,
    ) -> Option<impl Iterator<Item = Tetrahedron<'_, D>> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Tetrahedron)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned().map(move |node_ids| {
            Tetrahedron(Element {
                mesh: self,
                node_ids,
            })
        }))
    }

    /// An iterator over the hexahedra present in the mesh.
    pub fn hexahedra(
        &self,
    ) -> Option<impl Iterator<Item = Hexahedron<'_, D>> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Hexahedron)?;
        debug_assert_eq!(node_ids.len() % 8, 0);
        Some(node_ids.array_chunks().cloned().map(move |node_ids| {
            Hexahedron(Element {
                mesh: self,
                node_ids,
            })
        }))
    }

    /// An iterator over the elements of the given type present in the mesh.
    fn elements_of_type(&self, el_type: ElementType) -> Option<&Vec<usize>> {
        self.elements
            .iter()
            .find(|(et, _node_ids)| et == &el_type)
            .map(|(_et, node_ids)| node_ids)
    }
}

/// Parsing methods.
impl<const D: usize> Mesh<D> {
    /// Read and parse a mesh from a stream of bytes.
    ///
    /// The first format that succeeds to parse is chosen.
    pub fn from_reader(mut r: impl io::BufRead + io::Seek) -> io::Result<Mesh<D>> {
        use std::io::SeekFrom;

        let start_pos = r.stream_position()?;

        if let Ok(mesh) = medit::parse(&mut r) {
            return Ok(mesh);
        }

        let offset = (r.stream_position()? - start_pos) as i64;
        r.seek(SeekFrom::Current(-offset))?;

        if let Ok(mesh) = vtk::parse_xml(&mut r) {
            return Ok(mesh);
        }

        let offset = (r.stream_position()? - start_pos) as i64;
        r.seek(SeekFrom::Current(-offset))?;

        if let Ok(mesh) = vtk::parse_legacy_le(&mut r) {
            return Ok(mesh);
        }

        Err(io::ErrorKind::InvalidData.into())
    }

    /// Read and parse a mesh using the ASCII Medit format.
    pub fn from_reader_medit(r: impl io::BufRead) -> io::Result<Mesh<D>> {
        medit::parse(r).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    /// Read and parse a mesh using the ASCII Medit format.
    pub fn from_str_medit(s: &str) -> io::Result<Mesh<D>> {
        let r = io::Cursor::new(s);
        Mesh::from_reader_medit(r)
    }

    /// Read and parse a mesh using the XML VTK format.
    pub fn from_reader_vtk_xml(r: impl io::BufRead) -> io::Result<Mesh<D>> {
        vtk::parse_xml(r).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    /// Read and parse a mesh using the XML VTK format.
    pub fn from_str_vtk_xml(s: &str) -> io::Result<Mesh<D>> {
        let r = io::Cursor::new(s);
        Mesh::from_reader_vtk_xml(r)
    }

    /// Read and parse a mesh from a file.
    ///
    /// The file extension is used to detect the underlying format.
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Mesh<D>> {
        let path = path.as_ref();
        let extension = path.extension().and_then(std::ffi::OsStr::to_str);

        let file = fs::File::open(path)?;
        let r = io::BufReader::new(file);

        match extension {
            Some("mesh") => Mesh::from_reader_medit(r),
            Some("vtk") => Mesh::from_reader_vtk_xml(r), // TODO get the extension right
            _ => Mesh::from_reader(r),
        }
    }
}

impl<const D: usize> str::FromStr for Mesh<D> {
    type Err = io::Error;

    /// Read and parse a mesh from a string.
    ///
    /// The first format that succeeds to parse is chosen.
    fn from_str(s: &str) -> io::Result<Mesh<D>> {
        Mesh::from_reader(io::Cursor::new(s))
    }
}

/// Formatting methods.
impl<const D: usize> Mesh<D> {
    /// Serialize the mesh using the ASCII Medit format.
    ///
    /// # Example
    ///
    /// ```
    /// # use coupe::mesh::Mesh;
    /// let mesh = Mesh::<2>::from_str("MeshVersionFormatted 1\nDimension 2\n")
    ///     .unwrap();
    /// println!("{}", mesh.fmt_medit());
    /// ```
    pub fn fmt_medit(&self) -> impl fmt::Display + '_ {
        medit::Display { mesh: self }
    }

    // TODO fmt_vtk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_usage() {
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
        let mesh: Mesh<3> = input.parse().unwrap();
        assert_eq!(mesh.triangles().unwrap().len(), 2);
        let t = mesh.triangles().unwrap().next().unwrap();
        assert_eq!(t.0.as_node_ids(), &[0, 1, 2]);
    }
}
