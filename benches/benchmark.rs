#[macro_use]
extern crate criterion;
extern crate coupe;
extern crate itertools;
extern crate rand;

mod generator;

use coupe::algorithms::geometric::{axis_sort, rcb};
use coupe::geometry::Point2D;
use criterion::{Benchmark, Criterion, Throughput};

const SAMPLE_SIZE: usize = 5000;
const NUM_ITER: usize = 1;

fn bench_axis_sort_random(c: &mut Criterion) {
    c.bench(
        "axis_sort_random",
        Benchmark::new("axis_sort_random", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::new(0., 0.),
                Point2D::new(30., 10.),
                SAMPLE_SIZE,
            );
            let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
            let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
            b.iter(|| axis_sort(ids.clone(), weights.clone(), sample_points.clone(), true))
        }).throughput(Throughput::Elements(SAMPLE_SIZE as u32)),
    );
}

fn bench_raw_pdqsort_random(c: &mut Criterion) {
    c.bench(
        "raw_pdqsort_random",
        Benchmark::new("raw_pdqsort_random", move |b| {
            let sample_points = generator::uniform_f64(0., 30., SAMPLE_SIZE);
            b.iter(|| {
                let mut sample = sample_points.clone();
                sample.as_mut_slice().sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal)
                });
            })
        }).throughput(Throughput::Elements(SAMPLE_SIZE as u32)),
    );
}

fn bench_raw_pdqsort_sorted(c: &mut Criterion) {
    c.bench(
        "raw_pdqsort_sorted",
        Benchmark::new("raw_pdqsort_sorted", move |b| {
            let mut sample_points = generator::uniform_f64(0., 30., SAMPLE_SIZE);
            sample_points
                .as_mut_slice()
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal));
            b.iter(|| {
                let mut sample = sample_points.clone();
                sample.as_mut_slice().sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal)
                });
            })
        }).throughput(Throughput::Elements(SAMPLE_SIZE as u32)),
    );
}

fn bench_axis_sort_sorted(c: &mut Criterion) {
    c.bench(
        "axis_sort_sorted",
        Benchmark::new("axis_sort_sorted", move |b| {
            let sample_points = generator::already_x_sorted_rectangle(
                Point2D::new(0., 0.),
                Point2D::new(30., 10.),
                SAMPLE_SIZE,
            );
            let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
            let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
            b.iter(|| axis_sort(ids.clone(), weights.clone(), sample_points.clone(), true))
        }).throughput(Throughput::Elements(SAMPLE_SIZE as u32)),
    );
}

fn bench_rcb_random(c: &mut Criterion) {
    c.bench(
        "rcb_random",
        Benchmark::new("rcb_random", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::new(0., 0.),
                Point2D::new(30., 10.),
                SAMPLE_SIZE,
            );
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
        }).throughput(Throughput::Elements(SAMPLE_SIZE as u32)),
    );
}

criterion_group!(
    benches,
    bench_axis_sort_random,
    bench_axis_sort_sorted,
    bench_raw_pdqsort_random,
    bench_raw_pdqsort_sorted,
    bench_rcb_random
);
criterion_main!(benches);
