use std::mem;

use wgpu::util::DeviceExt as _;

fn slice_as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * mem::size_of::<T>()) }
}

fn bytes_as_slice<T>(v: &[u8]) -> &[T] {
    assert_eq!(v.len() % mem::size_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const T, v.len() / mem::size_of::<T>()) }
}

async fn steps_many(numbers: &[u32]) -> Vec<u32> {
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let number_buf = slice_as_bytes(numbers);
    let number_buf_size = number_buf.len() as wgpu::BufferAddress;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("my staging buffer"),
        size: number_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("my storage buffer"),
        contents: number_buf,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let cs_module =
        device.create_shader_module(&wgpu::include_wgsl!("algorithms/recursive_bisection.wgsl"));
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("my compute pipeline"),
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("my bindgroup"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("my command encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("my compute pass"),
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.insert_debug_marker("compute this mn yeah");
        compute_pass.dispatch(numbers.len() as u32, 1, 1);
    }
    command_encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, number_buf_size);
    queue.submit(Some(command_encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if buffer_future.await.is_err() {
        panic!("oh noes...");
    }

    let data = buffer_slice.get_mapped_range();
    let result = bytes_as_slice(&data).to_vec();
    mem::drop(data);
    staging_buffer.unmap();

    result
}

fn main() {
    let numbers: Vec<u32> = (1..128).collect();
    let steps = futures_lite::future::block_on(steps_many(&numbers));

    for step in steps {
        if step == u32::MAX {
            println!("overflow");
        } else {
            println!("{step}");
        }
    }
}
