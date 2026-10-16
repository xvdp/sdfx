// field_pat.slang — Points-as-Tori signed distance: a blend of per-point torus SDFs over the k nearest points.
// Mirror of sdfview/shaders/glsl/field_pat.glsl, function by function; the formulas are documented there.
// Differences from the GLSL: the textures and uniforms are members of PatField (slangpy tensors, (rows, 4) as in
// points_as_tori/slang/common.slang), passed as the first argument; the neighbor search is bvh.slang's
// bvhKNearestNeighbors (points_as_tori/slang, the module the fused k-NN kernel uses), with BVH_K_MAX and BVH_STACK_SIZE
// link-time constants set by the host.
module field_pat;

import slangpy;
import common;
import bvh;
import field;

public struct PatField : IField
{
    public Tensor<float, 2> points;      // (x, y, z, outlier)
    public Tensor<float, 2> tori;        // 2 rows per torus: (center.xyz, R), (axis.xyz, r)
    public Tensor<float, 2> bvh_bounds;  // 2 rows per node: (min.xyz, 0), (max.xyz, 0)
    public Tensor<int, 2> bvh_nodes;     // (left, right, prim_start, prim_count)
    public Tensor<int, 2> bvh_prims;     // (N, 1): leaf slot -> point index
    public int k;                        // neighbors (<= BVH_K_MAX)       (GLSL: pat_k)
    public float lambda_scale;           // 1.0                            (GLSL: pat_lambda_scale)
    public float max_exp_arg;            // 64.0                           (GLSL: pat_max_exp_arg)

    public float distance(float3 p) { return patDistance(this, p); }
    public float distanceGradient(float3 p, out float3 gradient) { return patDistanceGradient(this, p, gradient); }
}

void get_torus_params(PatField f, int i, out float3 center, out float3 axis, out float R, out float r)
{
    float4 t0 = fetch_texel(f.tori, 2 * i);
    float4 t1 = fetch_texel(f.tori, 2 * i + 1);
    center = t0.xyz;
    R = t0.w;
    axis = t1.xyz;
    r = t1.w;
}

struct TorusAccumulator
{
    float g;                          // Σ w_i g_i
    float3 grad_g_minus_lambda_n_g;   // Σ w_i (∇g_i - λ r̂_i g_i)
    float3 n;                         // Σ w_i r̂_i
    float normalization;              // Σ w_i
}

// Accumulate the contribution of point `pointIdx`
void accumulateTorusPoint(PatField f, float3 q, int pointIdx, float lambda, float shift, inout TorusAccumulator acc)
{
    float4 p_i = fetch_texel(f.points, pointIdx);
    if (p_i.w > 0.5) return;  // outlier
    float3 diff = q - p_i.xyz;
    float dist = length(diff);
    if (dist < 1e-10) return;

    float weight = exp(-lambda * (dist - shift));
    if (isnan(weight) || weight < 1e-30) return;

    float3 center, axis;
    float R, r;
    get_torus_params(f, pointIdx, center, axis, R, r);

    // torus signed distance g_i(q)
    float3 rVec = q - center;
    float v = length(cross(rVec, axis));
    if (v < 1e-10) return;
    float2 u = float2(v - R, dot(rVec, axis));
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
    float2 w = u / u_len;
    float3 grad_g = s * float3(w.x * G_11 + w.y * axis.x, w.x * G_12 + w.y * axis.y, w.x * G_13 + w.y * axis.z);

    float3 rHat = diff / dist;
    acc.grad_g_minus_lambda_n_g += weight * (grad_g - lambda * rHat * g_z);
    acc.g += weight * g_z;
    acc.n += weight * rHat;
    acc.normalization += weight;
}

// φ(q) and ∇φ(q) from the k nearest points
public float patDistanceGradient(PatField f, float3 q, out float3 gradient)
{
    int indices[BVH_K_MAX];
    float dists[BVH_K_MAX];
    int count = bvhKNearestNeighbors(f.bvh_bounds, f.bvh_nodes, f.bvh_prims, f.points, q, f.k, indices, dists);
    gradient = float3(0.0);
    if (count == 0) return INFINITY;

    float max_dist = 0.0;
    for (int i = 0; i < count; i++) {
        max_dist = max(max_dist, dists[i]);
    }
    float shift = 0.5 * max_dist;
    float lambda = f.lambda_scale * f.max_exp_arg / shift;

    TorusAccumulator acc;
    acc.g = 0.0;
    acc.grad_g_minus_lambda_n_g = float3(0.0);
    acc.n = float3(0.0);
    acc.normalization = 0.0;
    for (int i = 0; i < count; i++) {
        accumulateTorusPoint(f, q, indices[i], lambda, shift, acc);
    }

    float3 A = acc.grad_g_minus_lambda_n_g / acc.normalization;
    float distance = acc.g / acc.normalization;
    float3 B = lambda * acc.n / acc.normalization;
    gradient = A + distance * B;
    return distance;
}

public float patDistance(PatField f, float3 q)
{
    float3 gradient;  // the gradient terms cost little next to the neighbor search
    return patDistanceGradient(f, q, gradient);
}
