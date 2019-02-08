use coupe::geometry::Point2D;
use coupe::Compose;
use coupe::Partitioner;

const NUM_POINTS: usize = 5000;

fn main() {
    // algorithm composition:
    //    - initial partitioner: Multi-Jagged
    //    - then improve with k-means
    let mut k_means = coupe::KMeans::default();
    k_means.num_partitions = 7;
    k_means.delta_threshold = 0.;
    let algo = coupe::MultiJagged::new(7, 3).compose(k_means);

    let weights = vec![1.; NUM_POINTS];
    let points = examples::generator::rectangle_uniform(NUM_POINTS, Point2D::new(0., 0.), 4., 2.);

    let partition = algo.partition(points.as_slice(), &weights).into_ids();

    let part = points.into_iter().zip(partition).collect::<Vec<_>>();

    examples::plot_partition(part);
}
