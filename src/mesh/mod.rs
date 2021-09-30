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
    Vertex,
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
            ElementType::Vertex => 1,
            ElementType::Edge => 2,
            ElementType::Triangle => 3,
            ElementType::Quadrangle | ElementType::Tetrahedron => 4,
            ElementType::Hexahedron => 8,
        }
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
    /// This function will panic iff `node_idx` is greater or equal to the
    /// number of nodes.
    pub fn node(&self, node_idx: usize) -> [f64; D] {
        self.nodes[node_idx]
    }

    /// An iterator over the triangles present in the mesh.
    pub fn triangles(&self) -> Option<impl Iterator<Item = [usize; 3]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Triangle)?;
        debug_assert_eq!(node_ids.len() % 3, 0);
        Some(node_ids.array_chunks().cloned())
    }

    /// An iterator over the quadrangles present in the mesh.
    pub fn quadrangles(&self) -> Option<impl Iterator<Item = [usize; 4]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Quadrangle)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned())
    }

    /// An iterator over the tetrahedra present in the mesh.
    pub fn tetrahedra(&self) -> Option<impl Iterator<Item = [usize; 4]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Tetrahedron)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned())
    }

    /// An iterator over the hexahedra present in the mesh.
    pub fn hexahedra(&self) -> Option<impl Iterator<Item = [usize; 8]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Hexahedron)?;
        debug_assert_eq!(node_ids.len() % 8, 0);
        Some(node_ids.array_chunks().cloned())
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

    /// Read and parse a mesh using the XML VTK format.
    pub fn from_reader_vtk_xml(r: impl io::BufRead) -> io::Result<Mesh<D>> {
        vtk::parse_xml(r).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
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
