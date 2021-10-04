//! This module allows to load Medit mesh files, as
//! described by Frey in
//! [MEDIT : An interactive Mesh visualization Software](https://hal.inria.fr/inria-00069921).

use std::collections::HashMap;

pub use parser::Error as ParseError;

mod parser;
mod serializer;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ElementType {
    Vertex,
    Edge,
    Triangle,
    Quadrangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
}

impl ElementType {
    pub fn num_nodes(self) -> usize {
        match self {
            ElementType::Vertex => 1,
            ElementType::Edge => 2,
            ElementType::Triangle => 3,
            ElementType::Quadrangle | ElementType::Quadrilateral | ElementType::Tetrahedron => 4,
            ElementType::Hexahedron => 8,
        }
    }
}

/// MeditMesh data structure.
///
/// It stores main mesh informations about geometric coordinates and topology.
#[derive(Default, Debug)]
pub struct MeditMesh {
    dimension: usize,
    coordinates: Vec<f64>,
    topology: Vec<(ElementType, Vec<usize>)>,
}

impl MeditMesh {
    /// Returns the dimension of the mesh (2D, 3D, ...)
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a slice of f64 representing the coordinates of the points in the mesh.
    /// e.g. [x1, y1, x2, y2, ...]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    /// Returns a slice of matrices containing the connectivity for each mesh element
    pub fn topology(&self) -> &[(ElementType, Vec<usize>)] {
        &self.topology
    }

    /// Returns the number of nodes (vertices) of the mesh.
    pub fn num_nodes(&self) -> usize {
        self.coordinates.len() / self.dimension
    }

    pub fn num_elements(&self) -> usize {
        self.topology
            .iter()
            .map(|(element_type, elements)| elements.len() / element_type.num_nodes())
            .sum()
    }
}

impl MeditMesh {
    /// Split each element in N smaller elements of the same type, where N is
    /// the number of nodes of the element.
    pub fn refine(&self) -> MeditMesh {
        assert!(
            self.dimension == 2 || self.dimension == 3,
            "Only 2D and 3D meshes are supported",
        );

        let mut middles: HashMap<(usize, usize), usize> = HashMap::new();
        let mut middle = vec![0.0; self.dimension];
        let mut middle = move |coordinates: &mut Vec<f64>, mut p0: usize, mut p1: usize| -> usize {
            let id = coordinates.len() / self.dimension;
            if p1 < p0 {
                std::mem::swap(&mut p0, &mut p1);
            }
            *middles.entry((p0, p1)).or_insert_with(|| {
                let p0 = &coordinates[p0 * self.dimension..(p0 + 1) * self.dimension];
                let p1 = &coordinates[p1 * self.dimension..(p1 + 1) * self.dimension];
                for ((m, e0), e1) in middle.iter_mut().zip(p0.iter()).zip(p1.iter()) {
                    *m = (e0 + e1) / 2.0;
                }
                coordinates.extend_from_slice(&middle);
                id
            })
        };
        let mut new_coordinates = self.coordinates.clone();
        let mut new_topology: Vec<(ElementType, Vec<usize>)> =
            Vec::with_capacity(self.topology.len());

        for (element_type, elements) in &self.topology {
            let mut new_elements = Vec::new();
            match element_type {
                ElementType::Vertex | ElementType::Edge => continue,
                ElementType::Triangle => {
                    new_elements.reserve(4 * elements.len());
                    for vertices in elements.chunks(element_type.num_nodes()) {
                        let m01 = middle(&mut new_coordinates, vertices[0], vertices[1]);
                        let m02 = middle(&mut new_coordinates, vertices[0], vertices[2]);
                        let m12 = middle(&mut new_coordinates, vertices[1], vertices[2]);
                        new_elements.extend_from_slice(&[vertices[0], m01, m02]);
                        new_elements.extend_from_slice(&[vertices[1], m01, m12]);
                        new_elements.extend_from_slice(&[vertices[2], m02, m12]);
                        new_elements.extend_from_slice(&[m01, m02, m12]);
                    }
                }
                ElementType::Quadrangle | ElementType::Quadrilateral => {
                    new_elements.reserve(4 * elements.len());
                    for vertices in elements.chunks(element_type.num_nodes()) {
                        let v0 = vertices[0];
                        let v1 = vertices[1];
                        let v2 = vertices[2];
                        let v3 = vertices[3];
                        let m01 = middle(&mut new_coordinates, v0, v1);
                        let m12 = middle(&mut new_coordinates, v1, v2);
                        let m23 = middle(&mut new_coordinates, v2, v3);
                        let m30 = middle(&mut new_coordinates, v3, v0);
                        let mm = middle(&mut new_coordinates, m01, m23);
                        new_elements.extend_from_slice(&[v0, m01, mm, m30]);
                        new_elements.extend_from_slice(&[v1, m12, mm, m01]);
                        new_elements.extend_from_slice(&[v2, m23, mm, m12]);
                        new_elements.extend_from_slice(&[v3, m30, mm, m23]);
                    }
                }
                _ => {
                    // TODO
                    todo!();
                }
            }
            new_topology.push((*element_type, new_elements));
        }

        MeditMesh {
            dimension: self.dimension,
            coordinates: new_coordinates,
            topology: new_topology,
        }
    }
}
