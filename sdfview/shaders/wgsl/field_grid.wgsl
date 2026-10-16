// field_grid.wgsl — signed distance sampled on a regular grid (WGSL counterpart of field_grid.glsl).
// values[z][y][x] = φ(grid_min + (x, y, z) / (res - 1) · (grid_max - grid_min)) in an r32float 3D texture; trilinear
// interpolation in the shader from 8 textureLoad's (exact weights; r32float is not filterable in WebGPU without the
// optional float32-filterable feature, and the GLSL's hardware filter may round its weights to 8 bits).
// Outside the box: φ(clamp(p)) + |p - clamp(p)|. Gradient: central differences of the interpolant, h = one cell.
#include "common.wgsl"

@group(1) @binding(0) var grid_values: texture_3d<f32>;

fn gridSample(p: vec3f) -> f32 {
    let res = vec3i(textureDimensions(grid_values));
    let x = (p - U.grid_min) / (U.grid_max - U.grid_min) * vec3f(res - 1);  // in vertex units
    let i0 = clamp(vec3i(floor(x)), vec3i(0), res - 2);
    let t = x - vec3f(i0);
    let c000 = textureLoad(grid_values, i0, 0).r;
    let c100 = textureLoad(grid_values, i0 + vec3i(1, 0, 0), 0).r;
    let c010 = textureLoad(grid_values, i0 + vec3i(0, 1, 0), 0).r;
    let c110 = textureLoad(grid_values, i0 + vec3i(1, 1, 0), 0).r;
    let c001 = textureLoad(grid_values, i0 + vec3i(0, 0, 1), 0).r;
    let c101 = textureLoad(grid_values, i0 + vec3i(1, 0, 1), 0).r;
    let c011 = textureLoad(grid_values, i0 + vec3i(0, 1, 1), 0).r;
    let c111 = textureLoad(grid_values, i0 + vec3i(1, 1, 1), 0).r;
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

fn gridDistance(p: vec3f) -> f32 {
    let pc = clamp(p, U.grid_min, U.grid_max);
    return gridSample(pc) + length(p - pc);
}

fn gridDistanceGradient(p: vec3f, gradient: ptr<function, vec3f>) -> f32 {
    let h = (U.grid_max - U.grid_min) / (vec3f(textureDimensions(grid_values)) - 1.0);
    *gradient = vec3f(gridDistance(p + vec3f(h.x, 0.0, 0.0)) - gridDistance(p - vec3f(h.x, 0.0, 0.0)),
                      gridDistance(p + vec3f(0.0, h.y, 0.0)) - gridDistance(p - vec3f(0.0, h.y, 0.0)),
                      gridDistance(p + vec3f(0.0, 0.0, h.z)) - gridDistance(p - vec3f(0.0, 0.0, h.z))) / (2.0 * h);
    return gridDistance(p);
}
