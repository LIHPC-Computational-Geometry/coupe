struct Numbers {
    data: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> numbers: Numbers;

[[override]] let blockSize: u32;
var<workgroup> workgroup_data: array<u32, blockSize>;

[[stage(compute), workgroup_size(16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id: vec3<u32>,
) {
    var n: u32 = arrayLength(&numbers.data);

    if (global_id.x < n) {
        workgroup_data[local_id.x] = numbers.data[global_id.x];
    } else {
        workgroup_data[local_id.x] = 0u;
    }

    workgroupBarrier();

    for (var stride: u32 = blockSize / 2u; stride > 0u; stride = stride >> 1u) {
        var flag: u32 = u32(local_id.x < stride);
        workgroup_data[local_id.x] = workgroup_data[local_id.x] + flag * workgroup_data[local_id.x + flag * stride];
    }

    if (local_id.x == 0u) {
        numbers.data[workgroup_id.x] = workgroup_data[0];
    }
}
