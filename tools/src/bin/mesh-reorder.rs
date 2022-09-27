use anyhow::Context as _;
use anyhow::Result;
use mesh_io::Mesh;
use std::env;

const USAGE: &str = "Usage: mesh-reorder [options] [in-mesh [out-mesh]] <in.mesh >out.mesh";

fn shuffle_couple<R, T>(mut rng: R, data: &[T], refs: &[isize]) -> (Vec<T>, Vec<isize>, Vec<usize>)
where
    R: rand::RngCore,
    T: Clone,
{
    use rand::seq::SliceRandom as _;

    let dim = data.len() / refs.len();
    assert_eq!(data.len() % refs.len(), 0);

    let mut permutation: Vec<_> = (0..refs.len()).collect();
    permutation.shuffle(&mut rng);

    let data = permutation
        .iter()
        .flat_map(|i| &data[dim * i..dim * (i + 1)])
        .cloned()
        .collect();
    let refs = permutation.iter().map(|i| refs[*i]).collect();

    (data, refs, permutation)
}

fn shuffle<R>(mut rng: R, mesh: Mesh) -> Mesh
where
    R: rand::RngCore,
{
    let dimension = mesh.dimension();

    let (coordinates, node_refs, node_permutation) =
        shuffle_couple(&mut rng, mesh.coordinates(), mesh.node_refs());

    let node_permutation = {
        let mut p = vec![0; node_permutation.len()];
        for (i, e) in node_permutation.into_iter().enumerate() {
            p[e] = i;
        }
        p
    };

    let topology = mesh
        .topology()
        .iter()
        .map(|(el_type, el_nodes, el_refs)| {
            let (mut el_nodes, el_refs, _) = shuffle_couple(&mut rng, el_nodes, el_refs);
            for node in &mut el_nodes {
                *node = node_permutation[*node];
            }
            (*el_type, el_nodes, el_refs)
        })
        .collect();

    Mesh::from_raw_parts(dimension, coordinates, node_refs, topology)
}

fn main() -> Result<()> {
    let mut options = getopts::Options::new();
    options.optflag("h", "help", "print this help menu");
    options.optopt("f", "format", "output format", "EXT");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage(USAGE));
        return Ok(());
    }
    if matches.free.len() > 2 {
        anyhow::bail!("too many arguments\n\n{}", options.usage(USAGE));
    }

    let format = matches
        .opt_get("f")
        .context("invalid value for option 'format'")?;

    eprintln!("Reading mesh...");
    let mut mesh = coupe_tools::read_mesh(matches.free.get(0))?;

    eprintln!("Shuffling mesh...");
    mesh = shuffle(rand::thread_rng(), mesh);

    eprintln!("Writing mesh...");
    coupe_tools::write_mesh(&mesh, format, matches.free.get(1))?;

    Ok(())
}
