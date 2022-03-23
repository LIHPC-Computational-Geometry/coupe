mod generator;

use coupe::Partition as _;
use coupe::Point2D;
use criterion::{criterion_group, criterion_main};
use criterion::{Criterion, Throughput};
use rayon::prelude::*;

const SAMPLE_SIZE: usize = 5000;
const NUM_ITER: usize = 2;

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

fn bench_rcb_random(c: &mut Criterion) {
    c.benchmark_group("rcb_random")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("rcb_random", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::from([0., 0.]),
                Point2D::from([30., 10.]),
                SAMPLE_SIZE,
            );
            let weights = vec![1; SAMPLE_SIZE];
            let mut ids = vec![0; SAMPLE_SIZE];
            b.iter(|| {
                use rayon::iter::IntoParallelRefIterator as _;
                use rayon::iter::ParallelIterator as _;

                let p = sample_points.par_iter().cloned();
                let w = weights.par_iter().cloned();
                coupe::Rcb {
                    iter_count: NUM_ITER,
                }
                .partition(&mut ids, (p, w))
                .unwrap()
            })
        });
}

fn bench_k_means(c: &mut Criterion) {
    c.benchmark_group("k_means")
        .throughput(Throughput::Elements(SAMPLE_SIZE as u64))
        .bench_function("k_means", move |b| {
            let sample_points = generator::uniform_rectangle(
                Point2D::new(0., 0.),
                Point2D::new(30., 10.),
                SAMPLE_SIZE,
            );
            let weights = vec![1.0; SAMPLE_SIZE];
            b.iter(|| {
                coupe::KMeans {
                    part_count: 2_usize.pow(NUM_ITER as u32),
                    imbalance_tol: 5.0,
                    delta_threshold: 0.0,
                    hilbert: true,
                    ..Default::default()
                }
                .partition(&mut [0; SAMPLE_SIZE], (&sample_points, &weights))
                .unwrap()
            })
        });
}

criterion_group!(
    benches,
    bench_raw_pdqsort_random,
    bench_parallel_raw_pdqsort_random,
    bench_raw_pdqsort_sorted,
    bench_parallel_raw_pdqsort_sorted,
    bench_rcb_random,
    bench_k_means
);
criterion_main!(benches);
