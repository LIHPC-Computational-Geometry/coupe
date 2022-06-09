struct PrimeIndices {
    data: [[stride(4)]] array<u32>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage, read_write> v_indices: PrimeIndices;

let blockSize = 16u;
var<workgroup> sdata: array<u32, blockSize>;

[[stage(compute), workgroup_size(16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id: vec3<u32>,
) {
    var n: u32 = arrayLength(&v_indices.data);

    if (global_id.x < n) {
        sdata[local_id.x] = v_indices.data[global_id.x];
    } else {
        sdata[local_id.x] = 0u;
    }

    workgroupBarrier();

    for (var stride: u32 = blockSize / 2u; stride > 0u; stride = stride >> 1u) {
        if (local_id.x < stride) {
            sdata[local_id.x] = sdata[local_id.x] + sdata[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if (local_id.x == 0u) {
        v_indices.data[workgroup_id.x] = sdata[0];
    }
}
