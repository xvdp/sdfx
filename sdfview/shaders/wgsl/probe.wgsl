// probe.wgsl — field evaluation at query points (WGSL counterpart of probe.frag): a compute shader, one invocation
// per query; out[i] = (φ(q_i) - isovalue, ∇φ(q_i)). Queries and results are storage buffers (group 2).
#include "common.wgsl"
#include "field.wgsl"

@group(2) @binding(0) var<storage, read> queries: array<vec4f>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = i32(id.x);
    if (i >= U.u_count) { return; }
    var g: vec3f;
    let d = fieldDistanceGradient(queries[i].xyz, &g);
    probe_out[i] = vec4f(d, g);
}
