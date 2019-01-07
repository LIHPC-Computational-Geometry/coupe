use clap::load_yaml;
use clap::{App, ArgMatches};
use failure::{bail, Error};
use rayon::prelude::*;

use coupe::geometry::Point2D;
use coupe::{Compose, Partitioner};
use mesh_io::{mesh::Mesh, mesh::D3, xyz::XYZMesh};

fn main() -> Result<(), Error> {
    let yaml = load_yaml!("../../xyz_mesh.yml");
    let matches = App::from_yaml(yaml).get_matches();
    let file_name = matches.value_of("INPUT").unwrap();

    let mesh = XYZMesh::from_file(file_name)?;

    match matches.subcommand() {
        ("rcb", Some(submatches)) => rcb(&mesh, submatches),
        ("rib", Some(submatches)) => rib(&mesh, submatches),
        ("multi_jagged", Some(submatches)) => multi_jagged(&mesh, submatches),
        ("simplified_k_means", Some(submatches)) => simplified_k_means(&mesh, submatches),
        ("balanced_k_means", Some(submatches)) => balanced_k_means(&mesh, submatches),
        _ => bail! {"no subcommand specified"},
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

    let rcb = coupe::Rcb { num_iter };

    println!("info: entering RCB algorithm");
    let now = std::time::Instant::now();
    let partition = rcb.partition(&points, &weights).into_ids();
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

    let rib = coupe::Rib { num_iter };

    println!("info: entering RIB algorithm");
    let partition = rib.partition(&points, &weights).into_ids();
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

    let mj = coupe::MultiJagged {
        num_partitions,
        max_iter,
    };

    println!("info: entering Multi-Jagged algorithm");
    let now = std::time::Instant::now();
    let partition = mj.partition(&points, &weights).into_ids();
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

    let k_means = coupe::MultiJagged {
        num_partitions,
        max_iter: 2,
    } // fast multijagged for initial partition
    .compose::<coupe::dimension::U2>(coupe::KMeans {
        num_partitions,
        imbalance_tol,
        max_iter,
        max_balance_iter,
        delta_threshold: delta_max,
        erode,
        ..Default::default()
    });

    println!("info: entering balanced_k_means algorithm");
    let partition = k_means.partition(&points, &weights).into_ids();
    println!("info: left balanced_k_means algorithm");

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}
