//! This module allows to load Medit mesh files, as
//! described by Frey in
//! [MEDIT : An interactive Mesh visualization Software](https://hal.inria.fr/inria-00069921).

use std::collections::HashMap;

pub use parser::Error as ParseError;

mod parser;
mod serializer;

type Ref = isize;

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
    pub fn dimension(self) -> usize {
        match self {
            ElementType::Vertex => 0,
            ElementType::Edge => 1,
            ElementType::Triangle | ElementType::Quadrangle | ElementType::Quadrilateral => 2,
            ElementType::Tetrahedron | ElementType::Hexahedron => 3,
        }
    }

    pub fn node_count(self) -> usize {
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
pub struct Mesh {
    dimension: usize,
    coordinates: Vec<f64>,
    node_refs: Vec<Ref>,
    topology: Vec<(ElementType, Vec<usize>, Vec<Ref>)>,
}

impl Mesh {
    /// Returns the dimension of the mesh (2D, 3D, ...)
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a slice of f64 representing the coordinates of the points in the mesh.
    /// e.g. [x1, y1, x2, y2, ...]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    pub fn node(&self, idx: usize) -> &[f64] {
        &self.coordinates[idx * self.dimension..(idx + 1) * self.dimension]
    }

    pub fn nodes(&self) -> impl Iterator<Item = (&[f64], Ref)> {
        self.coordinates
            .chunks_exact(self.dimension)
            .zip(self.node_refs.iter().cloned())
    }

    pub fn nodes_mut(&mut self) -> impl Iterator<Item = (&mut [f64], &mut Ref)> {
        self.coordinates
            .chunks_exact_mut(self.dimension)
            .zip(self.node_refs.iter_mut())
    }

    /// Returns a slice of matrices containing the connectivity for each mesh element
    pub fn topology(&self) -> &[(ElementType, Vec<usize>, Vec<Ref>)] {
        &self.topology
    }

    pub fn elements(&self) -> impl Iterator<Item = (ElementType, &[usize], Ref)> {
        self.topology
            .iter()
            .flat_map(|(element_type, nodes, node_refs)| {
                nodes
                    .chunks_exact(element_type.node_count())
                    .zip(node_refs)
                    .map(|(node_chunk, node_ref)| (*element_type, node_chunk, *node_ref))
            })
    }

    pub fn elements_mut(&mut self) -> impl Iterator<Item = (ElementType, &mut [usize], &mut Ref)> {
        self.topology
            .iter_mut()
            .flat_map(|(element_type, nodes, node_refs)| {
                nodes
                    .chunks_exact_mut(element_type.node_count())
                    .zip(node_refs)
                    .map(|(node_chunk, node_ref)| (*element_type, node_chunk, node_ref))
            })
    }

    /// Returns the number of nodes (vertices) of the mesh.
    pub fn node_count(&self) -> usize {
        self.node_refs.len()
    }

    pub fn element_count(&self) -> usize {
        self.topology
            .iter()
            .map(|(_element_type, _elements, refs)| refs.len())
            .sum()
    }
}

impl Mesh {
    /// Split each element in N smaller elements of the same type, where N is
    /// the number of nodes of the element.
    pub fn refine(&self) -> Mesh {
        assert!(
            self.dimension == 2 || self.dimension == 3,
            "Only 2D and 3D meshes are supported",
        );

        let mut middles: HashMap<(usize, usize), usize> = HashMap::new();
        let mut middle = vec![0.0; self.dimension];
        let mut middle = move |coordinates: &mut Vec<f64>,
                               node_refs: &mut Vec<Ref>,
                               mut p0: usize,
                               mut p1: usize|
              -> usize {
            let id = coordinates.len() / self.dimension;
            if p1 < p0 {
                std::mem::swap(&mut p0, &mut p1);
            }
            *middles.entry((p0, p1)).or_insert_with(|| {
                node_refs.push((node_refs[p0] + node_refs[p1]) / 2);
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
        let mut new_node_refs = self.node_refs.clone();
        let mut new_topology = Vec::with_capacity(self.topology.len());

        for (element_type, elements, refs) in &self.topology {
            let mut new_elements = Vec::new();
            let mut new_refs = Vec::new();
            match element_type {
                ElementType::Vertex | ElementType::Edge => continue,
                ElementType::Triangle => {
                    new_elements.reserve(4 * elements.len());
                    for (vertices, reference) in elements.chunks_exact(3).zip(refs) {
                        let [v0, v1, v2] = <[usize; 3]>::try_from(vertices).unwrap();
                        let m01 = middle(&mut new_coordinates, &mut new_node_refs, v0, v1);
                        let m02 = middle(&mut new_coordinates, &mut new_node_refs, v0, v2);
                        let m12 = middle(&mut new_coordinates, &mut new_node_refs, v1, v2);
                        new_elements.extend_from_slice(&[v0, m01, m02]);
                        new_elements.extend_from_slice(&[v1, m01, m12]);
                        new_elements.extend_from_slice(&[v2, m02, m12]);
                        new_elements.extend_from_slice(&[m01, m02, m12]);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                    }
                }
                ElementType::Quadrangle | ElementType::Quadrilateral => {
                    new_elements.reserve(4 * elements.len());
                    for (vertices, reference) in elements.chunks_exact(4).zip(refs) {
                        let [v0, v1, v2, v3] = <[usize; 4]>::try_from(vertices).unwrap();
                        let m01 = middle(&mut new_coordinates, &mut new_node_refs, v0, v1);
                        let m12 = middle(&mut new_coordinates, &mut new_node_refs, v1, v2);
                        let m23 = middle(&mut new_coordinates, &mut new_node_refs, v2, v3);
                        let m30 = middle(&mut new_coordinates, &mut new_node_refs, v3, v0);
                        let mm = middle(&mut new_coordinates, &mut new_node_refs, m01, m23);
                        new_elements.extend_from_slice(&[v0, m01, mm, m30]);
                        new_elements.extend_from_slice(&[v1, m12, mm, m01]);
                        new_elements.extend_from_slice(&[v2, m23, mm, m12]);
                        new_elements.extend_from_slice(&[v3, m30, mm, m23]);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                        new_refs.push(*reference);
                    }
                }
                _ => {
                    // TODO
                    todo!();
                }
            }
            new_topology.push((*element_type, new_elements, new_refs));
        }

        Mesh {
            dimension: self.dimension,
            coordinates: new_coordinates,
            node_refs: new_node_refs,
            topology: new_topology,
        }
    }
}