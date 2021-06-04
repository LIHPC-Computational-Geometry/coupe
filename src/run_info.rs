/// Information on an algorithm run.
///
/// Filled in by algorithms when run on a given input.  Gives information about how the run went.
#[derive(Default, Copy, Clone)]
pub struct RunInfo {
    /// Number of iterations the algorithm underwent to provide a partition.
    pub algo_iterations: Option<u32>,
}

impl RunInfo {
    pub fn skip() -> RunInfo {
        RunInfo {
            algo_iterations: Some(0),
        }
    }
}
