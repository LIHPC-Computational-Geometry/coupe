struct Defer<F>(Option<F>)
where
    F: FnOnce();

impl<F> Drop for Defer<F>
where
    F: FnOnce(),
{
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

/// Call `f` on drop.
pub fn defer(f: impl FnOnce()) -> impl Drop {
    Defer(Some(f))
}
