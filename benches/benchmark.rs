mod generator;

use coupe::algorithms::k_means::simplified_k_means;
use coupe::algorithms::recursive_bisection::{axis_sort, rcb};
use coupe::geometry::Point2D;
use criterion::{criterion_group, criterion_main};
use criterion::{Criterion, Throughput};
use rayon::prelude::*;

const SAMPLE_SIZE: usize = 5000;
const NUM_ITER: usize = 2;

fn bench_axis_sort_random(c: &mut Criterion) {
    c.benchmark_group("axis_sort_random")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("axis_sort_random", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::from([0., 0.]),
                Point2D::from([30., 10.]),
                SAMPLE_SIZE,
            );
            let mut permutation: Vec<_> = (0..SAMPLE_SIZE).collect();

            b.iter(|| axis_sort(&sample_points, &mut permutation, 0))
        });
}

fn bench_raw_pdqsort_random(c: &mut Criterion) {
    c.benchmark_group("raw_pdqsort_random")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("raw_pdqsort_random", move |b| {
            let sample_points = generator::uniform_f64(0., 30., SAMPLE_SIZE);
            b.iter(|| {
                let mut sample = sample_points.clone();
                sample.as_mut_slice().sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal)
                });
            })
        });
}

fn bench_parallel_raw_pdqsort_random(c: &mut Criterion) {
    c.benchmark_group("parallel_raw_pdqsort_random")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("parallel_raw_pdqsort_random", move |b| {
            let sample_points = generator::uniform_f64(0., 30., SAMPLE_SIZE);
            b.iter(|| {
                let mut sample = sample_points.clone();
                sample.as_mut_slice().par_sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal)
                });
            })
        });
}

fn bench_raw_pdqsort_sorted(c: &mut Criterion) {
    c.benchmark_group("raw_pdqsort_sorted")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("raw_pdqsort_sorted", move |b| {
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
        });
}

fn bench_parallel_raw_pdqsort_sorted(c: &mut Criterion) {
    c.benchmark_group("parallel_raw_pdqsort_sorted")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("parallel_raw_pdqsort_sorted", move |b| {
            let mut sample_points = generator::uniform_f64(0., 30., SAMPLE_SIZE);
            sample_points
                .as_mut_slice()
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal));
            b.iter(|| {
                let mut sample = sample_points.clone();
                sample.as_mut_slice().par_sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal)
                });
            })
        });
}

fn bench_axis_sort_sorted(c: &mut Criterion) {
    c.benchmark_group("axis_sort_sorted")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("axis_sort_sorted", move |b| {
            let sample_points = generator::already_x_sorted_rectangle(
                Point2D::from([0., 0.]),
                Point2D::from([30., 10.]),
                SAMPLE_SIZE,
            );
            let mut permutation: Vec<_> = (0..SAMPLE_SIZE).collect();
            b.iter(|| axis_sort(&sample_points, &mut permutation, 0))
        });
}

fn bench_rcb_random(c: &mut Criterion) {
    c.benchmark_group("rcb_random")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("rcb_random", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::from([0., 0.]),
                Point2D::from([30., 10.]),
                SAMPLE_SIZE,
            );
            let weights = vec![1.0; sample_points.len()];
            let mut ids = vec![0; sample_points.len()];
            b.iter(|| rcb(&mut ids, &sample_points, &weights, NUM_ITER))
        });
}

fn bench_simplified_k_means(c: &mut Criterion) {
    c.benchmark_group("simplified_k_means")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("simplified_k_means", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::new(0., 0.),
                Point2D::new(30., 10.),
                SAMPLE_SIZE,
            );
            let ids: Vec<_> = (0..SAMPLE_SIZE).collect();
            let weights: Vec<_> = ids.iter().map(|_| 1.).collect();
            b.iter(|| {
                simplified_k_means(
                    &sample_points,
                    &weights,
                    2usize.pow(NUM_ITER as u32),
                    5.,
                    1000,
                    true,
                )
            })
        });
}

criterion_group!(
    benches,
    bench_axis_sort_random,
    bench_axis_sort_sorted,
    bench_raw_pdqsort_random,
    bench_parallel_raw_pdqsort_random,
    bench_raw_pdqsort_sorted,
    bench_parallel_raw_pdqsort_sorted,
    bench_rcb_random,
    bench_simplified_k_means
);
criterion_main!(benches);
