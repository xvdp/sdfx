// field_pat.glsl — Points-as-Tori signed distance: a blend of per-point torus SDFs over the k nearest points.
//
// Same evaluation as TorusDistanceField in k-NN mode (points_as_tori/torus_distance.py) and accumulateTorusPoint /
// blendedTorusDistanceGradient (ext/points-as-tori/demo/shaders_3d.py). Per query q, neighbors i ∈ kNN(q):
//   torus SDF   g_i(q) = s_i (sqrt(d1² + d2²) - |r_i|),  d1 = |(q - c_i) × a_i| - R_i,  d2 = (q - c_i)·a_i,  s_i = sign(r_i)
//   weights     w_i = exp(-λ (|q - p_i| - shift)),  shift = ½ max_i |q - p_i|,  λ = max_exp_arg · λ_scale / shift
//   blend       φ(q) = Σ w_i g_i / W,  W = Σ w_i
//   gradient    ∇φ = A + φ B,  A = Σ w_i (∇g_i - λ r̂_i g_i) / W,  B = λ Σ w_i r̂_i / W,  r̂_i = (q - p_i)/|q - p_i|
//               (shift and λ held fixed, as in the C++ / torch evaluators)
//   torus grad  ∇g_i = s_i (ŵ_x G_1 + ŵ_y a_i),  (ŵ_x, ŵ_y) = (d1, d2)/|(d1, d2)|,  G_1 = ∂|(q - c) × a|/∂q
// Points flagged as outliers (points.w > 0.5) take part in the neighbor search and the shift, not in the sums
// (torch: `include = not_outlier[idx]`). Degenerate pairs are skipped as in the original shader (query on a point,
// on a torus axis, or on a torus core circle), where the torch evaluator would return NaN.
#include "common.glsl"
#include "bvh.glsl"

uniform sampler2D pat_points;       // (x, y, z, outlier)
uniform sampler2D pat_tori;         // 2 texels per torus: (center.xyz, R), (axis.xyz, r)
uniform sampler2D pat_bvh_bounds;   // 2 texels per node: (min.xyz, 0), (max.xyz, 0)
uniform isampler2D pat_bvh_nodes;   // (left, right, prim_start, prim_count)
uniform isampler2D pat_bvh_prims;   // leaf slot -> point index
uniform int pat_k;                  // neighbors (<= BVH_K_MAX)
uniform float pat_lambda_scale;     // 1.0
uniform float pat_max_exp_arg;      // 64.0

void get_torus_params(in int i, out vec3 center, out vec3 axis, out float R, out float r) {
    vec4 t0 = fetch_texel(pat_tori, 2 * i);
    vec4 t1 = fetch_texel(pat_tori, 2 * i + 1);
    center = t0.xyz;
    R = t0.w;
    axis = t1.xyz;
    r = t1.w;
}

struct TorusAccumulator {
    float g;                        // Σ w_i g_i
    vec3 grad_g_minus_lambda_n_g;   // Σ w_i (∇g_i - λ r̂_i g_i)
    vec3 n;                         // Σ w_i r̂_i
    float normalization;            // Σ w_i
};

// Accumulate the contribution of point `pointIdx`
void accumulateTorusPoint(vec3 q, int pointIdx, float lambda, float shift, inout TorusAccumulator acc) {
    vec4 p_i = fetch_texel(pat_points, pointIdx);
    if (p_i.w > 0.5) return;  // outlier
    vec3 diff = q - p_i.xyz;
    float dist = length(diff);
    if (dist < 1e-10) return;

    float weight = exp(-lambda * (dist - shift));
    if (isnan(weight) || weight < 1e-30) return;

    vec3 center, axis;
    float R, r;
    get_torus_params(pointIdx, center, axis, R, r);

    // torus signed distance g_i(q)
    vec3 rVec = q - center;
    float v = length(cross(rVec, axis));
    if (v < 1e-10) return;
    vec2 u = vec2(v - R, dot(rVec, axis));
    float u_len = length(u);
    if (u_len < 1e-10) return;
    float s = r >= 0.0 ? 1.0 : -1.0;
    float g_z = s * (u_len - abs(r));

    // ∇g_i(q): G is the 2x3 Jacobian of (v - R, (q - c)·a) with respect to q
    float G_11 = (axis.x * axis.y * (center.y - q.y) + axis.x * axis.z * (center.z - q.z)
                  - (axis.y * axis.y + axis.z * axis.z) * (center.x - q.x)) / v;
    float G_12 = (axis.x * axis.y * (center.x - q.x) + axis.y * axis.z * (center.z - q.z)
                  - (axis.x * axis.x + axis.z * axis.z) * (center.y - q.y)) / v;
    float G_13 = (axis.x * axis.z * (center.x - q.x) + axis.y * axis.z * (center.y - q.y)
                  - (axis.x * axis.x + axis.y * axis.y) * (center.z - q.z)) / v;
    vec2 w = u / u_len;
    vec3 grad_g = s * vec3(w.x * G_11 + w.y * axis.x, w.x * G_12 + w.y * axis.y, w.x * G_13 + w.y * axis.z);

    vec3 rHat = diff / dist;
    acc.grad_g_minus_lambda_n_g += weight * (grad_g - lambda * rHat * g_z);
    acc.g += weight * g_z;
    acc.n += weight * rHat;
    acc.normalization += weight;
}

// φ(q) and ∇φ(q) from the k nearest points
float patDistanceGradient(in vec3 q, out vec3 gradient) {
    int indices[BVH_K_MAX];
    float dists[BVH_K_MAX];
    int count = bvhKNearestNeighbors(pat_bvh_bounds, pat_bvh_nodes, pat_bvh_prims, pat_points, q, pat_k,
                                     indices, dists);
    gradient = vec3(0.0);
    if (count == 0) return INFINITY;

    float max_dist = 0.0;
    for (int i = 0; i < count; i++) {
        max_dist = max(max_dist, dists[i]);
    }
    float shift = 0.5 * max_dist;
    float lambda = pat_lambda_scale * pat_max_exp_arg / shift;

    TorusAccumulator acc;
    acc.g = 0.0;
    acc.grad_g_minus_lambda_n_g = vec3(0.0);
    acc.n = vec3(0.0);
    acc.normalization = 0.0;
    for (int i = 0; i < count; i++) {
        accumulateTorusPoint(q, indices[i], lambda, shift, acc);
    }

    vec3 A = acc.grad_g_minus_lambda_n_g / acc.normalization;
    float distance = acc.g / acc.normalization;
    vec3 B = lambda * acc.n / acc.normalization;
    gradient = A + distance * B;
    return distance;
}

float patDistance(in vec3 q) {
    vec3 gradient;  // the gradient terms cost little next to the neighbor search
    return patDistanceGradient(q, gradient);
}
