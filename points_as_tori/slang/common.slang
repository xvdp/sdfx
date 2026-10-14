// common.slang — constants and texel addressing shared by all modules (mirror of sdfview/shaders/glsl/common.glsl).
//
// Arrays are slangpy tensors with one element row per texel (sdfview/scene.py layout): element i of an array with t
// texels per element occupies rows t·i .. t·i + t - 1 of a (rows, 4) tensor. The GLSL reads the same rows from 2D
// textures at (j mod W, j div W); here row j is read directly.
module common;

import slangpy;

public static const float INFINITY = 1e30;
public static const float PI = 3.14159265358979;

public float4 fetch_texel(Tensor<float, 2> tex, int idx)
{
    return float4(tex[idx, 0], tex[idx, 1], tex[idx, 2], tex[idx, 3]);
}

public int4 fetch_texel(Tensor<int, 2> tex, int idx)
{
    return int4(tex[idx, 0], tex[idx, 1], tex[idx, 2], tex[idx, 3]);
}
