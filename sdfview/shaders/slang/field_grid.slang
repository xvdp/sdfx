// field_grid.slang — signed distance sampled on a regular grid (mirror of field_grid.glsl / field_grid.wgsl).
// values[z, y, x] = φ(grid_min + (x, y, z) / (res - 1) · (grid_max - grid_min)), a (d, h, w) tensor; trilinear
// interpolation from 8 reads (exact weights, as the WGSL; the GLSL's hardware filter may round them to 8 bits).
// Outside the box: φ(clamp(p)) + |p - clamp(p)|. Gradient: central differences of the interpolant, h = one cell.
module field_grid;

import slangpy;
import field;

public struct GridField : IField
{
    public Tensor<float, 3> values;  // [z, y, x]  (GLSL: sampler3D grid_values)
    public int3 res;                 // (w, h, d)  (GLSL: textureSize(grid_values, 0))
    public float3 grid_min;
    public float3 grid_max;

    public float distance(float3 p) { return gridDistance(this, p); }
    public float distanceGradient(float3 p, out float3 gradient) { return gridDistanceGradient(this, p, gradient); }
}

float gridValue(GridField f, int3 i)
{
    return f.values[i.z, i.y, i.x];
}

public float gridSample(GridField f, float3 p)
{
    float3 x = (p - f.grid_min) / (f.grid_max - f.grid_min) * float3(f.res - 1);  // in vertex units
    int3 i0 = clamp(int3(floor(x)), int3(0), f.res - 2);
    float3 t = x - float3(i0);
    float c00 = lerp(gridValue(f, i0), gridValue(f, i0 + int3(1, 0, 0)), t.x);
    float c10 = lerp(gridValue(f, i0 + int3(0, 1, 0)), gridValue(f, i0 + int3(1, 1, 0)), t.x);
    float c01 = lerp(gridValue(f, i0 + int3(0, 0, 1)), gridValue(f, i0 + int3(1, 0, 1)), t.x);
    float c11 = lerp(gridValue(f, i0 + int3(0, 1, 1)), gridValue(f, i0 + int3(1, 1, 1)), t.x);
    return lerp(lerp(c00, c10, t.y), lerp(c01, c11, t.y), t.z);
}

public float gridDistance(GridField f, float3 p)
{
    float3 pc = clamp(p, f.grid_min, f.grid_max);
    return gridSample(f, pc) + length(p - pc);
}

public float gridDistanceGradient(GridField f, float3 p, out float3 gradient)
{
    float3 h = (f.grid_max - f.grid_min) / (float3(f.res) - 1.0);
    gradient = float3(gridDistance(f, p + float3(h.x, 0.0, 0.0)) - gridDistance(f, p - float3(h.x, 0.0, 0.0)),
                      gridDistance(f, p + float3(0.0, h.y, 0.0)) - gridDistance(f, p - float3(0.0, h.y, 0.0)),
                      gridDistance(f, p + float3(0.0, 0.0, h.z)) - gridDistance(f, p - float3(0.0, 0.0, h.z))) / (2.0 * h);
    return gridDistance(f, p);
}
