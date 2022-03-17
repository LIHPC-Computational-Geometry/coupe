use coupe::geometry::Point2D;
use rand::{self, Rng};

pub fn uniform_f64(min: f64, max: f64, num_vals: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..num_vals).map(|_| rng.gen_range(min..max)).collect()
}

pub fn uniform_rectangle(p_min: Point2D, p_max: Point2D, num_points: usize) -> Vec<Point2D> {
    let mut rng = rand::thread_rng();
    (0..num_points)
        .map(|_| {
            Point2D::from([
                rng.gen_range(p_min.x..p_max.x),
                rng.gen_range(p_min.y..p_max.y),
            ])
        })
        .collect()
}
