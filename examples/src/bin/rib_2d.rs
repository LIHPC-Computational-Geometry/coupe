use clap::load_yaml;
use clap::App;

use coupe::geometry::Point2D;
use coupe::Partitioner;
use coupe::Rib;

fn main() {
    let yaml = load_yaml!("../../rcb.yml");
    let matches = App::from_yaml(yaml).get_matches();

    let num_iter: usize = matches
        .value_of("num_iter")
        .unwrap_or_default()
        .parse()
        .expect("wrong value for num_iter");

    let num_points: usize = matches
        .value_of("num_points")
        .unwrap_or_default()
        .parse()
        .expect("Wrong value for num_points");

    let rib = Rib { num_iter };

    let weights = vec![1.; num_points];

    let points = examples::generator::rectangle_uniform(num_points, Point2D::new(0., 0.), 2., 8.);

    let partition = rib.partition(&points, &weights).into_ids();

    if !matches.is_present("quiet") {
        let part = points.into_iter().zip(partition).collect::<Vec<_>>();
        examples::plot_partition(part)
    }
}
