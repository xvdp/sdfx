// common.glsl — constants and texel addressing shared by all modules.
//
// Arrays live in 2D textures, one element row per texel (sdfview/scene.py): element i of an array with t texels per
// element occupies texels t·i .. t·i + t - 1, and texel j sits at (j mod W, j div W), W = texture width.
// Same scheme as fetch_texel in ext/points-as-tori/demo/shaders_3d.py, with W read from the texture itself.

const float INFINITY = 1e30;
const float PI = 3.14159265358979;

ivec2 index_to_texel(int idx, int width) {
    return ivec2(idx % width, idx / width);
}

vec4 fetch_texel(sampler2D tex, int idx) {
    return texelFetch(tex, index_to_texel(idx, textureSize(tex, 0).x), 0);
}

ivec4 fetch_texel(isampler2D tex, int idx) {
    return texelFetch(tex, index_to_texel(idx, textureSize(tex, 0).x), 0);
}
