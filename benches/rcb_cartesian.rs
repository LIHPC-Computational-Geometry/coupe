use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use std::num::NonZeroUsize;

pub fn bench(c: &mut Criterion) {
    let width = NonZeroUsize::new(10000).unwrap();
    let height = NonZeroUsize::new(10000).unwrap();

    let count = usize::from(width) * usize::from(height);

    let grid = coupe::Grid::new_2d(width, height);
    let weights: Vec<f64> = (0..count).map(|i| i as f64).collect();
    let mut partition = vec![0; count];

    let core_count = num_cpus::get();
    let mut group = c.benchmark_group("rcb_cartesian");

    for thread_count in [1, 2, 4, 8, 16, 24, 32, 40] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .spawn_handler(|thread| {
                let mut b = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    b = b.name(name.to_owned());
                }
                if let Some(stack_size) = thread.stack_size() {
                    b = b.stack_size(stack_size);
                }
                b.spawn(move || {
                    let core_idx = thread.index() % core_count;
                    core_affinity::set_for_current(core_affinity::CoreId { id: core_idx });
                    thread.run();
                })?;
                Ok(())
            })
            .build()
            .unwrap();
        group.bench_function(&thread_count.to_string(), |b| {
            pool.install(|| {
                b.iter(|| grid.rcb(black_box(&mut partition), black_box(&weights), 12))
            });
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
