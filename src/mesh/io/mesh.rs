//! An abstract interface to the different supported file formats

/// An abstract representation of a mesh.
///
/// It is a layer of abstraction over how the different file
/// formats represent a mesh.
///
/// A Mesh is represented by a collection of vertices and
/// a collection of elements which links vertices together
pub trait Mesh {
    /// The dimension of the mesh: either 1D, 2D or 3D
    ///
    /// Dimensions are reprensented by types, respectively D1, D2 and D3
    /// and each dimension should have an associated type to represent a vertex in that dimension
    type Dim;

    /// Required function.
    ///
    /// Returns an array of vertices of type defined by Dim.
    fn vertices(&self) -> Vec<<Self::Dim as PointType>::Point>
    where
        Self::Dim: PointType;

    /// Required function.
    ///
    /// Returns an array of arrays of indices. Each subarray
    /// contains the indices of the vertices (in `self.vertices()`)
    /// that form a single element.
    fn elements_vertices(&self) -> Vec<Vec<usize>>;
}

pub trait PointType {
    type Point;
    const N_COORD: usize;
}

/// Represents 1D
pub struct D1;

/// Represents 2D
pub struct D2;

/// Represents 3D
pub struct D3;

impl PointType for D1 {
    type Point = f64;
    const N_COORD: usize = 1;
}

impl PointType for D2 {
    type Point = (f64, f64);
    const N_COORD: usize = 2;
}

impl PointType for D3 {
    type Point = (f64, f64, f64);
    const N_COORD: usize = 3;
}
