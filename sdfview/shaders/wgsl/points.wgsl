// points.wgsl — point sprites (WGSL counterpart of points.vert + points.frag). WebGPU rasterizes point primitives
// 1 pixel wide, so each point is an instanced quad (6 vertices, two triangles) of the size GL gives gl_PointSize:
// diameter = 2 r · P[1][1] · (h / 2) / w_clip pixels, clamped to [1, 64]; `corner` replaces 2 gl_PointCoord - 1.
#include "common.wgsl"
#include "camera.wgsl"

@group(1) @binding(0) var<storage, read> u_positions: array<vec4f>;  // (x, y, z, outlier)
@group(1) @binding(1) var<storage, read> u_normals: array<vec4f>;    // (nx, ny, nz, 0); a placeholder without normals

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) corner: vec2f,
    @location(1) normal: vec3f,
    @location(2) outlier: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOut {
    var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
    let corner = corners[v];
    let p = u_positions[i];
    var clip = U.u_view_proj * vec4f(p.xyz, 1.0);
    let size = clamp(U.u_point_radius * U.u_proj11 * U.u_viewport.w / clip.w, 1.0, 64.0);  // pixels
    clip = vec4f(clip.xy + corner * size / U.u_viewport.zw * clip.w, clip.zw);  // ± size/2 px = ± size/viewport NDC
    clip = glToWebGPUClip(clip);
    var normal = vec3f(0.0);
    if (U.u_has_normals != 0) { normal = u_normals[i].xyz; }
    return VertexOut(clip, corner, normal, p.w);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let r2 = dot(in.corner, in.corner);
    if (r2 > 1.0) { discard; }
    var col = select(U.u_point_color, vec3f(0.85, 0.1, 0.1), in.outlier > 0.5);
    var shade = 1.0 - 0.35 * r2;  // sphere-like falloff
    if (U.u_has_normals != 0) {
        let nd = dot(normalize(in.normal), -U.u_camera_dir);
        col = select(mix(col, vec3f(0.55), 0.6), col, nd >= 0.0);  // facing away: grayed
        shade *= 0.45 + 0.55 * abs(nd);
    }
    return vec4f(col * shade, 1.0);
}
