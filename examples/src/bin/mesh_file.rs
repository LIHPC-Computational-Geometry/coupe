use clap::load_yaml;
use clap::{App, ArgMatches};
use failure::{bail, Error};
use rayon::prelude::*;

use coupe::geometry::Point2D;
use coupe::{Compose, Partitioner, TopologicPartitioner};
use mesh_io::medit::MeditMesh;
use mesh_io::{mesh::Mesh, mesh::D3, xyz::XYZMesh};

fn main() -> Result<(), Error> {
    let yaml = load_yaml!("../../mesh_file.yml");
    let matches = App::from_yaml(yaml).get_matches();
    let file_name = matches.value_of("INPUT").unwrap();

    match matches.value_of("format").unwrap_or_default() {
        "xyz" => {
            let mesh = XYZMesh::from_file(file_name)?;
            match matches.subcommand() {
                ("rcb", Some(submatches)) => rcb(&mesh, submatches),
                ("rib", Some(submatches)) => rib(&mesh, submatches),
                ("multi_jagged", Some(submatches)) => multi_jagged(&mesh, submatches),
                ("simplified_k_means", Some(submatches)) => simplified_k_means(&mesh, submatches),
                ("balanced_k_means", Some(submatches)) => balanced_k_means(&mesh, submatches),
                ("kernighan_lin", Some(_submatches)) => {
                    bail! { "kernighan_lin is not supported with the XYZ mesh format" }
                }
                ("fiduccia_mattheyses", Some(_submatches)) => {
                    bail! { "fiduccia_mattheyses is not supported with the XYZ mesh format" }
                }
                _ => bail! {"no subcommand specified"},
            }
        }
        "medit" => {
            let mesh = MeditMesh::from_file(file_name)?;
            match matches.subcommand() {
                ("kernighan_lin", Some(submatches)) => kernighan_lin(&mesh, submatches),
                ("fiduccia_mattheyses", Some(submatches)) => fiduccia_mattheyses(&mesh, submatches),
                _ => {
                    bail! { "unsupported mesh format for this algorithm or wrong command specified" }
                }
            }
        }
        _ => bail! { "Unknown file format" },
    }

    Ok(())
}

fn rcb<'a>(mesh: &impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let num_iter: usize = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let num_points = points.len();

    let weights = (1..num_points)
        .into_par_iter()
        .map(|_| 1.)
        .collect::<Vec<_>>();

    let rcb = coupe::Rcb::new(num_iter);

    println!("info: entering RCB algorithm");
    let now = std::time::Instant::now();
    let partition = rcb.partition(points.as_slice(), &weights).into_ids();
    let end = now.elapsed();
    println!("info: left RCB algorithm. {:?} elapsed.", end);

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn rib<'a>(mesh: &impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let num_iter: usize = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let num_points = points.len();

    let weights = (1..num_points)
        .into_par_iter()
        .map(|_| 1.)
        .collect::<Vec<_>>();

    let rib = coupe::Rib::new(num_iter);

    println!("info: entering RIB algorithm");
    let partition = rib.partition(points.as_slice(), &weights).into_ids();
    println!("info: left RIB algorithm");

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn multi_jagged<'a>(mesh: &impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let num_partitions: usize = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_partitions");

    let max_iter: usize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let num_points = points.len();

    let weights = (0..num_points)
        .into_par_iter()
        .map(|_| 1.)
        .collect::<Vec<_>>();

    let mj = coupe::MultiJagged::new(num_partitions, max_iter);

    println!("info: entering Multi-Jagged algorithm");
    let now = std::time::Instant::now();
    let partition = mj.partition(points.as_slice(), &weights).into_ids();
    let end = now.elapsed();
    println!("info: left Multi-Jagged algorithm. elapsed = {:?}", end);

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn simplified_k_means<'a>(mesh: &impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let weights = points.par_iter().map(|_| 1.).collect::<Vec<_>>();

    let max_iter: isize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

    let num_partitions: usize = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_partitions");

    let imbalance_tol: f64 = matches
        .value_of("imbalance_tol")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for imbalance_tol");

    println!("info: entering simplified_k_means algorithm");
    let partition = coupe::algorithms::k_means::simplified_k_means(
        &points,
        &weights,
        num_partitions,
        imbalance_tol,
        max_iter,
        true,
    );
    println!("info: left simplified_k_means algorithm");

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}

fn balanced_k_means<'a>(mesh: &impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let weights = points.par_iter().map(|_| 1.).collect::<Vec<_>>();

    let max_iter: usize = matches
        .value_of("max_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_iter");

    let max_balance_iter: usize = matches
        .value_of("max_balance_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for max_balance_iter");

    let num_partitions: usize = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_partitions");

    let imbalance_tol: f64 = matches
        .value_of("imbalance_tol")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for imbalance_tol");

    let delta_max: f64 = matches
        .value_of("delta_max")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for delta_max");

    let erode = matches.is_present("erode");

    let k_means = coupe::KMeans::new(
        num_partitions,
        imbalance_tol,
        delta_max,
        max_iter,
        max_balance_iter,
        erode,
        true,
        false,
    );
    let k_means = coupe::MultiJagged::new(num_partitions, 2).compose(k_means);

    println!("info: entering balanced_k_means algorithm");
    let partition = k_means.partition(points.as_slice(), &weights).into_ids();
    println!("info: left balanced_k_means algorithm");

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}

fn kernighan_lin<'a>(mesh: &MeditMesh, matches: &ArgMatches<'a>) {
    let conn = examples::generate_connectivity_matrix_medit(&mesh);
    let adjacency = coupe::topology::adjacency_matrix(conn.view(), 2);

    let coordinates = mesh.coordinates();

    assert_eq!(mesh.dimension(), 3);

    // Mesh is in 3D, discard the z-coordinate
    let coords: Vec<f64> = coordinates
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 != 2)
        .map(|(_, val)| *val)
        .collect::<Vec<_>>();
    let points =
        unsafe { std::slice::from_raw_parts(coords.as_ptr() as *const Point2D, coords.len() / 2) };

    // We are partitioning mesh elements, and we position them
    // via their centers. We need to construct the array of
    // elements centers from the nodes
    let views = mesh
        .topology()
        .iter()
        .map(|mat| mat.view())
        .collect::<Vec<_>>();
    // let stacked = sprs::vstack(&views);
    let points = views[1] // <------ WTH is this? what's inside views[0]?
        .outer_iterator()
        .map(|conn| {
            conn.iter().fold(Point2D::new(0., 0.), |acc, (j, _)| {
                Point2D::new(acc.x + points[j].x / 3., acc.y + points[j].y / 3.)
            })
        })
        .collect::<Vec<_>>();

    let num_points = points.len();

    let weights = (0..num_points)
        .into_par_iter()
        .map(|_| 1.)
        .collect::<Vec<_>>();

    let num_iter = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let num_partitions = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_partitions");

    // let mut k_means = coupe::KMeans::default();
    // k_means.num_partitions = 2;
    // k_means.imbalance_tol = 5.;
    // let algo = coupe::HilbertCurve::new(2, 4).compose(k_means);
    let algo =
        coupe::HilbertCurve::new(num_partitions, 4).compose(coupe::KernighanLin::new(num_iter, 2.));

    let partition = algo.partition(points.as_slice(), weights.as_slice(), adjacency.view());

    let ids = partition.into_ids();

    if !matches.is_present("quiet") {
        let part = points
            .iter()
            .cloned()
            // .zip(partition.ids().iter().cloned())
            .zip(ids.iter().cloned())
            .collect::<Vec<_>>();
        examples::plot_partition(part);
    }
}

fn fiduccia_mattheyses<'a>(mesh: &MeditMesh, matches: &ArgMatches<'a>) {
    let conn = examples::generate_connectivity_matrix_medit(&mesh);
    let adjacency = coupe::topology::adjacency_matrix(conn.view(), 2);

    let coordinates = mesh.coordinates();

    assert_eq!(mesh.dimension(), 3);

    // Mesh is in 3D, discard the z-coordinate
    let coords: Vec<f64> = coordinates
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 != 2)
        .map(|(_, val)| *val)
        .collect::<Vec<_>>();
    let points =
        unsafe { std::slice::from_raw_parts(coords.as_ptr() as *const Point2D, coords.len() / 2) };

    // We are partitioning mesh elements, and we position them
    // via their centers. We need to construct the array of
    // elements centers from the nodes
    let views = mesh
        .topology()
        .iter()
        .map(|mat| mat.view())
        .collect::<Vec<_>>();
    // let stacked = sprs::vstack(&views);
    let points = views[1] // <------ WTH is this? what's inside views[0]?
        .outer_iterator()
        .map(|conn| {
            conn.iter().fold(Point2D::new(0., 0.), |acc, (j, _)| {
                Point2D::new(acc.x + points[j].x / 3., acc.y + points[j].y / 3.)
            })
        })
        .collect::<Vec<_>>();

    let num_points = points.len();

    let weights = (0..num_points)
        .into_par_iter()
        .map(|_| 1.)
        .collect::<Vec<_>>();

    let num_iter = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let num_partitions = matches
        .value_of("num_partitions")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_partitions");

    // let mut k_means = coupe::KMeans::default();
    // k_means.num_partitions = 2;
    // k_means.imbalance_tol = 5.;
    // let algo = coupe::HilbertCurve::new(2, 4).compose(k_means);
    let algo = coupe::HilbertCurve::new(num_partitions, 4)
        .compose(coupe::FiducciaMattheyses::new(num_iter, 2.));

    let partition = algo.partition(points.as_slice(), weights.as_slice(), adjacency.view());

    let ids = partition.into_ids();

    if !matches.is_present("quiet") {
        let part = points
            .iter()
            .cloned()
            // .zip(partition.ids().iter().cloned())
            .zip(ids.iter().cloned())
            .collect::<Vec<_>>();
        examples::plot_partition(part);
    }
}
