// common.wgsl — constants and the uniform block shared by all modules (WGSL counterpart of common.glsl).
//
// Arrays: storage buffers holding the scene's rows as stored (sdfview/scene.py), one vec4 per 4 values; element i of
// an array with t vec4 per element is at entries t·i .. t·i + t - 1. Linear indexing replaces GLSL's texel addressing
// (fetch_texel: index -> (i mod W, i div W) in a 2D texture), which WebGL needs and WebGPU does not.
// Uniforms: WGSL has no loose uniforms; the GLSL uniforms of every module are members of one struct (same names),
// group 0 binding 0. Layout (std140-like rules of WGSL uniform buffers; offsets in bytes) mirrored in
// sdfview/web/webgpu/renderer.js UNIFORMS.

const INFINITY: f32 = 1e30;

struct Uniforms {
    u_view_proj: mat4x4f,       //   0  P V (column major)
    u_inv_view_proj: mat4x4f,   //  64  (P V)⁻¹
    u_viewport: vec4f,          // 128  x, y, width, height (pixels)
    u_box_min: vec3f,           // 144
    u_max_steps: i32,           // 156
    u_box_max: vec3f,           // 160
    u_epsilon: f32,             // 172
    u_color_front: vec3f,       // 176
    u_step_scale: f32,          // 188
    u_color_back: vec3f,        // 192
    u_opacity: f32,             // 204
    grid_min: vec3f,            // 208
    field_isovalue: f32,        // 220
    grid_max: vec3f,            // 224
    pat_k: i32,                 // 236
    pat_lambda_scale: f32,      // 240
    pat_max_exp_arg: f32,       // 244
    u_count: i32,               // 248  probe: number of queries
    u_color_parts: i32,         // 252
    u_clip_plane: vec4f,        // 256  surface: keep n·x <= c, (n, c)
    u_plane_origin: vec3f,      // 272  cut plane: rectangle origin ± u ± v
    u_clip: i32,                // 284  surface: clip at u_clip_plane (0 / 1)
    u_plane_u: vec3f,           // 288
    u_colormap_range: f32,      // 300
    u_plane_v: vec3f,           // 304
    u_isoband_frequency: f32,   // 316
    u_camera_dir: vec3f,        // 320  points: view direction
    u_point_radius: f32,        // 332
    u_point_color: vec3f,       // 336
    u_proj11: f32,              // 348  P[1][1] = cot(fov_y / 2)
    u_has_normals: i32,         // 352
}                               // 368 bytes (struct size: a multiple of its 16-byte alignment)

@group(0) @binding(0) var<uniform> U: Uniforms;
