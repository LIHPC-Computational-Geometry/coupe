use itertools::{self, Itertools};
use vtkio::{self, model::DataSet};

use std::fmt;
use std::path::{Path, PathBuf};

use crate::mesh;

#[derive(Debug)]
pub enum Error {
    Vtkio(vtkio::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Vtkio(err) => write!(f, "vtkio: {}", err),
        }
    }
}

impl std::error::Error for Error {}

impl From<vtkio::Error> for Error {
    fn from(err: vtkio::Error) -> Error {
        Error::Vtkio(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, PartialEq, Clone)]
pub struct VtkMesh {
    data_set: DataSet,
}

impl VtkMesh {
    /// Reads a vtk format mesh from the specified path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data_set = vtkio::import(path.as_ref())?;
        Ok(VtkMesh {
            data_set: data_set.data,
        })
    }

    /// Saves the current mesh in a vtk file.
    ///
    /// Data is exported in binary format.
    ///
    /// If `None`, the version will default to `4.1`.
    pub fn save(
        self,
        path: impl AsRef<Path>,
        title: impl Into<String>,
        version: impl Into<Option<(u8, u8)>>,
    ) -> Result<(), Error> {
        let vtk = vtkio::model::Vtk {
            version: vtkio::model::Version::new(version.into().unwrap_or((4, 1))),
            title: title.into(),
            data: self.data_set,
        };
        vtkio::export(vtk, &PathBuf::from(path.as_ref()))?;
        Ok(())
    }

    /// Constructs an iterator over all the points in the mesh.
    /// The mesh is always represented in 3D. The points are stored
    /// in a single buffer in which x, y, z coordinates are interleaved.
    /// E.g. points = [x1, y1, z1, x2, y2, z2, ...]
    pub fn points(&self) -> impl Iterator<Item = &f64> {
        use vtkio::model::DataSet::*;
        match self.data_set {
            UnstructuredGrid { ref points, .. } => points.iter().unwrap(),
            StructuredGrid { ref points, .. } => points.iter().unwrap(),
            _ => unimplemented!(),
        }
    }

    /// Construcs an iterator over the x coordinate of each
    /// point in the mesh.
    /// E.g. xs = [x1, x2, x3, x4, ...]
    pub fn xs(&self) -> impl Iterator<Item = &f64> {
        self.points().step_by(3)
    }

    /// Construcs an iterator over the x coordinate of each
    /// point in the mesh.
    /// E.g. ys = [y1, y2, y3, y4, ...]
    pub fn ys(&self) -> impl Iterator<Item = &f64> {
        self.points().skip(1).step_by(3)
    }

    /// Construcs an iterator over the x coordinate of each
    /// point in the mesh.
    /// E.g. zs = [z1, z2, z3, z4, ...]
    pub fn zs(&self) -> impl Iterator<Item = &f64> {
        self.points().skip(2).step_by(3)
    }

    /// Constructs an iterator over the [x|y|z] coordinate of
    /// the center of each cell of the mesh, matching the
    /// `CartesianAxis` input variant.
    pub fn centers(&self, axis: CartesianAxis) -> impl Iterator<Item = f64> + '_ {
        use vtkio::model::DataSet::*;
        match self.data_set {
            UnstructuredGrid {
                cells: vtkio::model::Cells { ref vertices, .. },
                ..
            } => {
                let coords: Vec<_> = match axis {
                    CartesianAxis::X => self.xs().collect(),
                    CartesianAxis::Y => self.ys().collect(),
                    CartesianAxis::Z => self.zs().collect(),
                };

                itertools::unfold(vertices.iter(), move |vertices| {
                    if let Some(&n_vertices) = vertices.next() {
                        let vertices_indices = vertices.take(n_vertices as usize);
                        let mut mean_x = 0.;
                        for idx in vertices_indices {
                            mean_x += coords[*idx as usize];
                        }
                        Some(mean_x / (f64::from(n_vertices)))
                    } else {
                        None
                    }
                })
            }
            _ => unimplemented!(),
        }
    }

    /// Returns the number of cells present in the mesh.
    /// If the mesh is not of type `UnstructuredGrid`, then
    /// the notion of cell is meaningless for the current mesh and
    /// this function returns `0`.      
    pub fn num_cells(&self) -> usize {
        use vtkio::model::DataSet::*;
        match self.data_set {
            UnstructuredGrid {
                cells: vtkio::model::Cells { num_cells, .. },
                ..
            } => num_cells as usize,
            _ => 0,
        }
    }

    pub fn is_structured_points(&self) -> bool {
        matches!(self.data_set, DataSet::StructuredPoints { .. })
    }

    pub fn is_structured_grid(&self) -> bool {
        matches!(self.data_set, DataSet::StructuredGrid { .. })
    }

    pub fn is_unstructured_grid(&self) -> bool {
        matches!(self.data_set, DataSet::UnstructuredGrid { .. })
    }

    pub fn is_rectilinear_grid(&self) -> bool {
        matches!(self.data_set, DataSet::RectilinearGrid { .. })
    }

    pub fn is_poly_data(&self) -> bool {
        matches!(self.data_set, DataSet::PolyData { .. })
    }
}

impl mesh::Mesh for VtkMesh {
    type Dim = mesh::D3;

    fn vertices(&self) -> Vec<<Self::Dim as mesh::PointType>::Point>
    where
        Self::Dim: mesh::PointType,
    {
        self.points()
            .chunks(3)
            .into_iter()
            .map(|mut chunk| {
                (
                    *chunk.next().unwrap(),
                    *chunk.next().unwrap(),
                    *chunk.next().unwrap(),
                )
            })
            .collect()
    }

    fn elements_vertices(&self) -> Vec<Vec<usize>> {
        use vtkio::model::DataSet::*;
        match self.data_set {
            UnstructuredGrid {
                cells: vtkio::model::Cells { ref vertices, .. },
                ..
            } => itertools::unfold(vertices.iter(), move |vertices| {
                if let Some(&n_vertices) = vertices.next() {
                    Some(
                        vertices
                            .take(n_vertices as usize)
                            .cloned()
                            .map(|v| v as usize)
                            .collect::<Vec<usize>>(),
                    )
                } else {
                    None
                }
            })
            .collect(),
            _ => unimplemented!(),
        }
    }
}
