// field_grid.glsl — signed distance sampled on a regular grid (scene field type "grid"), trilinear interpolation.
//
// values[z][y][x] = φ(grid_min + (x, y, z) / (res - 1) · (grid_max - grid_min)): samples sit on the grid vertices, so
// the texture coordinate of p is uvw = ((p - min)/(max - min) · (res - 1) + ½) / res (texel centres = vertices).
// Outside the box: φ(clamp(p)) + |p - clamp(p)| (rays are clipped to the box; only probes go outside).
// Gradient: central differences of the interpolant with h = one cell.
#include "common.glsl"

uniform sampler3D grid_values;  // R32F (linear filtering)
uniform vec3 grid_min;
uniform vec3 grid_max;

float gridSample(vec3 p) {
    vec3 res = vec3(textureSize(grid_values, 0));
    vec3 uvw = (p - grid_min) / (grid_max - grid_min);
    return texture(grid_values, (uvw * (res - 1.0) + 0.5) / res).r;
}

float gridDistance(in vec3 p) {
    vec3 pc = clamp(p, grid_min, grid_max);
    return gridSample(pc) + length(p - pc);
}

float gridDistanceGradient(in vec3 p, out vec3 gradient) {
    vec3 h = (grid_max - grid_min) / (vec3(textureSize(grid_values, 0)) - 1.0);
    gradient = vec3(gridDistance(p + vec3(h.x, 0.0, 0.0)) - gridDistance(p - vec3(h.x, 0.0, 0.0)),
                    gridDistance(p + vec3(0.0, h.y, 0.0)) - gridDistance(p - vec3(0.0, h.y, 0.0)),
                    gridDistance(p + vec3(0.0, 0.0, h.z)) - gridDistance(p - vec3(0.0, 0.0, h.z))) / (2.0 * h);
    return gridDistance(p);
}
