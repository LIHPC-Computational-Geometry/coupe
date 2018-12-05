extern crate coupe;

use coupe::geometry::Point2D;
use coupe::InitialPartition;

const NUM_POINTS: usize = 5000;

fn main() {
    // algorithm composition:
    //    - initial partitioner: ZCurve
    //    - then improve with k-means
    let z_curve = coupe::ZCurve {
        num_partitions: 10,
        order: 4,
    }; // 8 partitions
    let mut k_means: coupe::KMeans = Default::default();
    k_means.num_partitions = 10;
    k_means.delta_threshold = 0.000000001;
    k_means.max_balance_iter = 20;
    k_means.max_iter = 500;

    let algo = coupe::compose_two_initial_improver(z_curve, k_means);

    let weights = vec![1.; NUM_POINTS];
    let points = examples::generator::rectangle_uniform(NUM_POINTS, Point2D::new(0., 0.), 4., 2.);

    let partition = algo.partition(&points, &weights);

    let part = points.into_iter().zip(partition).collect::<Vec<_>>();

    examples::plot_partition(part);
}
