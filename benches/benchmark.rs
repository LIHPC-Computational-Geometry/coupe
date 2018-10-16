#[macro_use]
extern crate criterion;
extern crate coupe;
extern crate itertools;
extern crate rand;

mod generator;

use coupe::algorithms::geometric::{axis_sort, rcb};
use coupe::geometry::Point2D;
use criterion::Criterion;

const SAMPLE_SIZE: usize = 5000;
const NUM_ITER: usize = 1;
// const NUM_PARTITION: usize = 2usize.pow(NUM_ITER as u32);

fn bench_axis_sort_random(c: &mut Criterion) {
    c.bench_function("axis_sort_random", move |b| {
        let sample_points =
            generator::uniform_rectangle(Point2D::new(0., 0.), Point2D::new(30., 10.), SAMPLE_SIZE);
        let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
        let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
        b.iter(|| axis_sort(ids.clone(), weights.clone(), sample_points.clone(), true))
    });
}

fn bench_axis_sort_sorted(c: &mut Criterion) {
    c.bench_function("axis_sort_sorted", move |b| {
        let sample_points = generator::already_x_sorted_rectangle(
            Point2D::new(0., 0.),
            Point2D::new(30., 10.),
            SAMPLE_SIZE,
        );
        let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
        let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
        b.iter(|| axis_sort(ids.clone(), weights.clone(), sample_points.clone(), true))
    });
}

fn bench_rcb_random(c: &mut Criterion) {
    c.bench_function("rcb_random", move |b| {
        let sample_points =
            generator::uniform_rectangle(Point2D::new(0., 0.), Point2D::new(30., 10.), SAMPLE_SIZE);
        let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
        let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
        b.iter(|| {
            rcb(
                ids.clone(),
                weights.clone(),
                sample_points.clone(),
                NUM_ITER,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_axis_sort_random,
    bench_axis_sort_sorted,
    bench_rcb_random
);
criterion_main!(benches);
