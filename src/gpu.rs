use once_cell::sync::Lazy;

pub enum InitError {
    NoAdapter,
    CannotRequestDevice,
}

struct Context {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn init_context() -> Result<Context, InitError> {
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or(InitError::NoAdapter)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|_| InitError::CannotRequestDevice)?;
    Ok(Context {
        instance,
        adapter,
        device,
        queue,
    })
}

static CONTEXT: Lazy<Result<Context, InitError>> =
    Lazy::new(|| futures_lite::future::block_on(init_context()));

pub fn is_available() -> bool {
    CONTEXT.is_ok()
}
