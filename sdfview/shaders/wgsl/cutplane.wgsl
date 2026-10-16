// cutplane.wgsl — field values on a plane through the scene box (WGSL counterpart of cutplane.vert + cutplane.frag):
// a rectangle origin ± u ± v (triangle strip of 4 vertices, no vertex buffer), colormap, isobands, zero contour.
#include "common.wgsl"
#include "camera.wgsl"
#include "field.wgsl"
#include "colormap.wgsl"

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let c = vec2f(f32(i & 1u), f32((i >> 1u) & 1u)) * 2.0 - 1.0;
    let world = U.u_plane_origin + c.x * U.u_plane_u + c.y * U.u_plane_v;
    return VertexOut(glToWebGPUClip(U.u_view_proj * vec4f(world, 1.0)), world);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let phi = fieldDistance(in.world);
    let fw = fwidth(phi);  // before the discard: derivatives need uniform control flow
    let slack = 1e-4 * (U.u_box_max - U.u_box_min);
    if (any(in.world < U.u_box_min - slack) || any(in.world > U.u_box_max + slack)) { discard; }
    return vec4f(sdfColor(phi, fw, U.u_colormap_range, U.u_isoband_frequency), U.u_opacity);
}
