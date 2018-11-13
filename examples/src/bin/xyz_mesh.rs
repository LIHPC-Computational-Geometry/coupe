extern crate clap;
extern crate coupe;
extern crate failure;
extern crate mesh_io;
extern crate rayon;

use clap::load_yaml;
use clap::{App, ArgMatches};
use failure::{bail, Error};
use rayon::prelude::*;

use coupe::algorithms;
use coupe::algorithms::k_means::BalancedKmeansSettings;
use coupe::algorithms::multi_jagged::*;
use coupe::geometry::Point2D;
use mesh_io::{mesh::Mesh, mesh::D3, xyz::XYZMesh};

fn main() -> Result<(), Error> {
    let yaml = load_yaml!("../../xyz_mesh.yml");
    let matches = App::from_yaml(yaml).get_matches();
    let file_name = matches.value_of("INPUT").unwrap();

    let mesh = XYZMesh::from_file(file_name)?;

    match matches.subcommand() {
        ("rcb", Some(submatches)) => rcb(mesh, submatches),
        ("rib", Some(submatches)) => rib(mesh, submatches),
        ("multi_jagged", Some(submatches)) => multi_jagged(mesh, submatches),
        ("simplified_k_means", Some(submatches)) => simplified_k_means(mesh, submatches),
        ("balanced_k_means", Some(submatches)) => balanced_k_means(mesh, submatches),
        _ => bail! {"no subcommand specified"},
    }

    Ok(())
}

fn rcb<'a>(mesh: impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
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

    let points_clone = points.clone();
    println!("info: entering RCB algorithm");
    let now = std::time::Instant::now();
    let partition = algorithms::geometric::rcb(&weights, &points_clone, num_iter);
    let end = now.elapsed();
    println!("info: left RCB algorithm. {:?} elapsed.", end);

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn rib<'a>(mesh: impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
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

    println!("info: entering RIB algorithm");
    let partition = algorithms::geometric::rib(&weights, &points.clone(), num_iter);
    println!("info: left RIB algorithm");

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn multi_jagged<'a>(mesh: impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
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

    println!("info: entering Multi-Jagged algorithm");
    let now = std::time::Instant::now();
    let partition = multi_jagged_2d_with_scheme(&points, &weights, &[1, 1, 1]);
    let end = now.elapsed();
    println!("info: left Multi-Jagged algorithm. elapsed = {:?}", end);

    if !matches.is_present("quiet") {
        let part = points.into_par_iter().zip(partition).collect::<Vec<_>>();

        examples::plot_partition(part)
    }
}

fn simplified_k_means<'a>(mesh: impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
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
    let (partition, _weights) = algorithms::k_means::simplified_k_means(
        points,
        weights,
        num_partitions,
        imbalance_tol,
        max_iter,
        true,
    );
    println!("info: left simplified_k_means algorithm");

    if !matches.is_present("quiet") {
        examples::plot_partition(partition)
    }
}

fn balanced_k_means<'a>(mesh: impl Mesh<Dim = D3>, matches: &ArgMatches<'a>) {
    let points = mesh
        .vertices()
        .into_par_iter()
        .map(|p| Point2D::new(p.x, p.y))
        .collect::<Vec<_>>();

    let _weights = points.par_iter().map(|_| 1.).collect::<Vec<_>>();

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
    let hilbert = matches.is_present("hilbert");

    let settings = BalancedKmeansSettings {
        num_partitions,
        imbalance_tol: imbalance_tol,
        max_iter,
        max_balance_iter,
        delta_threshold: delta_max,
        erode: erode,
        hilbert: hilbert,
        ..Default::default()
    };

    println!("info: entering balanced_k_means algorithm");
    let partition = algorithms::k_means::balanced_k_means(points, settings);
    println!("info: left balanced_k_means algorithm");

    if !matches.is_present("quiet") {
        examples::plot_partition(partition)
    }
}
