use super::Mesh;
use std::fs;
use std::io;
use std::path::Path;

mod medit;

impl<const D: usize> Mesh<D> {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Mesh<D>> {
        let file = fs::File::open(path)?;
        let r = io::BufReader::new(file);
        Mesh::from_reader(r)
    }

    pub fn from_reader(r: impl io::BufRead) -> io::Result<Mesh<D>> {
        // TODO
    }
}
