// probe_knn.frag — k nearest points of each query through the BVH (parity tests): the neighbors are sorted by
// (distance, index) and entries 4·slot .. 4·slot + 3 are written (-1 past the end); run ⌈k/4⌉ times.
#include "common.glsl"
#include "bvh.glsl"

uniform sampler2D u_queries;
uniform int u_count;
uniform int u_k;
uniform int u_slot;
uniform sampler2D pat_points;
uniform sampler2D pat_bvh_bounds;
uniform isampler2D pat_bvh_nodes;
uniform isampler2D pat_bvh_prims;

layout(location = 0) out ivec4 frag_out;

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    int idx = xy.y * textureSize(u_queries, 0).x + xy.x;
    frag_out = ivec4(-1);
    if (idx >= u_count) return;
    int indices[BVH_K_MAX];
    float dists[BVH_K_MAX];
    int count = bvhKNearestNeighbors(pat_bvh_bounds, pat_bvh_nodes, pat_bvh_prims, pat_points,
                                     fetch_texel(u_queries, idx).xyz, u_k, indices, dists);
    // insertion sort by (distance, index)
    for (int i = 1; i < count; i++) {
        int ki = indices[i];
        float kd = dists[i];
        int j = i - 1;
        while (j >= 0 && (dists[j] > kd || (dists[j] == kd && indices[j] > ki))) {
            indices[j + 1] = indices[j];
            dists[j + 1] = dists[j];
            j--;
        }
        indices[j + 1] = ki;
        dists[j + 1] = kd;
    }
    ivec4 out_idx = ivec4(-1);
    for (int c = 0; c < 4; c++) {
        int i = 4 * u_slot + c;
        if (i < count) out_idx[c] = indices[i];
    }
    frag_out = out_idx;
}
