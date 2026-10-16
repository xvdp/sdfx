// camera.wgsl — primary rays and depth for full-screen passes (WGSL counterpart of camera.glsl).
//   ray through a pixel: unproject the NDC points (x, y, -1) and (x, y, 1) with (P V)⁻¹ (OpenGL-style matrices from
//   web/core/camera.js, the same as the GLSL hosts); depth of a world point: ½ z_ndc + ½ in [0, 1].
// Deviation: WebGPU's @builtin(position) has its origin at the top left (y down), GL's gl_FragCoord at the bottom left;
// cameraRay flips y so both give the same image.
#include "common.wgsl"

struct Ray {
    ro: vec3f,
    rd: vec3f,
}

fn cameraRay(frag_xy: vec2f) -> Ray {
    let v = U.u_viewport;
    let ndc = vec2f(2.0 * (frag_xy.x - v.x) / v.z - 1.0, 1.0 - 2.0 * (frag_xy.y - v.y) / v.w);
    let near = U.u_inv_view_proj * vec4f(ndc, -1.0, 1.0);
    let far = U.u_inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let ro = near.xyz / near.w;
    return Ray(ro, normalize(far.xyz / far.w - ro));
}

fn fragDepth(p: vec3f) -> f32 {
    let clip = U.u_view_proj * vec4f(p, 1.0);
    return clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
}

// slab test: (t_near, t_far); the ray (line) meets the box when t_near <= t_far and t_far >= 0
fn intersectAABB(ro: vec3f, rd: vec3f, aabb_min: vec3f, aabb_max: vec3f) -> vec2f {
    let inv_dir = 1.0 / rd;
    let t0 = (aabb_min - ro) * inv_dir;
    let t1 = (aabb_max - ro) * inv_dir;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    return vec2f(max(max(t_min.x, t_min.y), t_min.z), min(min(t_max.x, t_max.y), t_max.z));
}
