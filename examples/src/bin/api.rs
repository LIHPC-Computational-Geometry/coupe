use coupe::dimension::U2;
use coupe::geometry::Point2D;
use coupe::Compose;
use coupe::Partitioner;

const NUM_POINTS: usize = 5000;

fn main() {
    // algorithm composition:
    //    - initial partitioner: Multi-Jagged
    //    - then improve with k-means
    let algo = coupe::MultiJagged {
        num_partitions: 7,
        max_iter: 3,
    }
    .compose::<U2>(coupe::KMeans {
        num_partitions: 7,
        delta_threshold: 0.,
        ..Default::default()
    });

    let weights = vec![1.; NUM_POINTS];
    let points = examples::generator::rectangle_uniform(NUM_POINTS, Point2D::new(0., 0.), 4., 2.);

    let partition = algo.partition(&points, &weights);

    let part = points.into_iter().zip(partition).collect::<Vec<_>>();

    examples::plot_partition(part);
}
