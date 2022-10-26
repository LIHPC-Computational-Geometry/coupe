fn main() {
    let x = std::env::args().nth(1).unwrap().parse().unwrap();
    let y = std::env::args().nth(2).unwrap().parse().unwrap();
    let iter = std::env::args()
        .nth(3)
        .unwrap_or_else(|| String::from("12"))
        .parse()
        .unwrap();
    eprintln!("grid size: ({x},{y}); rcb iters: {iter}");
    let grid = coupe::Grid::new_2d(x, y);
    let n = usize::from(x) * usize::from(y);
    let weights: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut partition = vec![0; n];

    let domain = ittapi::Domain::new("MyIncredibleDomain");
    let before = std::time::Instant::now();
    let task = ittapi::Task::begin(&domain, "MyIncredibleTask");
    grid.rcb(&mut partition, &weights, iter);
    std::mem::drop(task);
    eprintln!("time: {:?}", before.elapsed());

    let i = usize::from(x);
    eprint!("partition[{}] = {}\r", i, partition[i]);
}
