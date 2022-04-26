use anyhow::Result;
use mesh_io::medit::Mesh;
use std::env;
use std::io;

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
    options.optopt("s", "seed", "RNG seed (default: 0)", "STRING");

    let matches = options.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        eprintln!("{}", options.usage("Usage: mesh-refine [options]"));
        return Ok(());
    }

    eprintln!("Reading mesh...");
    let stdin = io::stdin();
    let stdin = stdin.lock();
    let stdin = io::BufReader::new(stdin);
    let mut mesh = Mesh::from_reader(stdin).unwrap();

    eprintln!("Shuffling mesh...");
    mesh = shuffle(rand::thread_rng(), mesh);

    eprintln!("Writing mesh...");
    println!("{}", mesh);

    Ok(())
}
