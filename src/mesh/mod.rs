pub mod io;

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

#[derive(Default, Debug)]
pub struct Mesh<const D: usize> {
    nodes: Vec<[f64; D]>,
    elements: Vec<(ElementType, Vec<usize>)>,
}

impl<const D: usize> Mesh<D> {
    pub fn nodes(&self) -> impl Iterator<Item = [f64; D]> + ExactSizeIterator + '_ {
        self.nodes.iter().cloned()
    }

    pub fn node(&self, node_idx: usize) -> [f64; D] {
        self.nodes[node_idx]
    }

    pub fn triangles(&self) -> Option<impl Iterator<Item = [usize; 3]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Triangle)?;
        debug_assert_eq!(node_ids.len() % 3, 0);
        Some(node_ids.array_chunks().cloned())
    }

    pub fn num_quads(&self) -> usize {
        let quadrangles = self
            .elements_of_type(ElementType::Quadrangle)
            .map_or(0, |node_ids| {
                debug_assert_eq!(node_ids.len() % 4, 0);
                node_ids.len() / 4
            });
        let quadrilaterals =
            self.elements_of_type(ElementType::Quadrilateral)
                .map_or(0, |node_ids| {
                    debug_assert_eq!(node_ids.len() % 4, 0);
                    node_ids.len() / 4
                });
        quadrangles + quadrilaterals
    }

    pub fn quads(&self) -> Option<impl Iterator<Item = [usize; 4]> + '_> {
        let quadrangles = self
            .elements_of_type(ElementType::Quadrangle)?
            .array_chunks()
            .cloned();
        let quadrilaterals = self
            .elements_of_type(ElementType::Quadrilateral)?
            .array_chunks()
            .cloned();
        Some(quadrangles.chain(quadrilaterals))
    }

    pub fn quadrangles(&self) -> Option<impl Iterator<Item = [usize; 4]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Quadrangle)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned())
    }

    pub fn quadrilaterals(
        &self,
    ) -> Option<impl Iterator<Item = [usize; 4]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Quadrilateral)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned())
    }

    pub fn tetrahedra(&self) -> Option<impl Iterator<Item = [usize; 4]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Tetrahedron)?;
        debug_assert_eq!(node_ids.len() % 4, 0);
        Some(node_ids.array_chunks().cloned())
    }

    pub fn hexahedra(&self) -> Option<impl Iterator<Item = [usize; 8]> + ExactSizeIterator + '_> {
        let node_ids = self.elements_of_type(ElementType::Hexahedron)?;
        debug_assert_eq!(node_ids.len() % 8, 0);
        Some(node_ids.array_chunks().cloned())
    }

    fn elements_of_type(&self, el_type: ElementType) -> Option<&Vec<usize>> {
        self.elements
            .iter()
            .find(|(et, _node_ids)| et == &el_type)
            .map(|(_et, node_ids)| node_ids)
    }
}
