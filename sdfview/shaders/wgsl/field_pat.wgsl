// field_pat.wgsl — Points-as-Tori signed distance (WGSL counterpart of field_pat.glsl; the formulas are listed there):
// per query q, a blend of the torus SDFs g_i of its k nearest points with weights w_i = exp(-λ (|q - p_i| - shift)),
// φ = Σ w_i g_i / Σ w_i, ∇φ = A + φ B. Outlier points (points.w > 0.5) take part in the search, not in the sums.
#include "common.wgsl"
#include "bvh.wgsl"

@group(1) @binding(0) var<storage, read> pat_points: array<vec4f>;    // (x, y, z, outlier)
@group(1) @binding(1) var<storage, read> pat_tori: array<vec4f>;      // 2 per torus: (center.xyz, R), (axis.xyz, r)
@group(1) @binding(2) var<storage, read> pat_bvh_bounds: array<vec4f>; // 2 per node: (min.xyz, 0), (max.xyz, 0)
@group(1) @binding(3) var<storage, read> pat_bvh_nodes: array<vec4i>; // (left, right, prim_start, prim_count)
@group(1) @binding(4) var<storage, read> pat_bvh_prims: array<i32>;   // leaf slot -> point index

struct TorusAccumulator {
    g: f32,                          // Σ w_i g_i
    grad_g_minus_lambda_n_g: vec3f,  // Σ w_i (∇g_i - λ r̂_i g_i)
    n: vec3f,                        // Σ w_i r̂_i
    normalization: f32,              // Σ w_i
}

// Accumulate the contribution of point `pointIdx`
fn accumulateTorusPoint(q: vec3f, pointIdx: i32, lambda: f32, shift: f32, acc: ptr<function, TorusAccumulator>) {
    let p_i = pat_points[pointIdx];
    if (p_i.w > 0.5) { return; }  // outlier
    let diff = q - p_i.xyz;
    let dist = length(diff);
    if (dist < 1e-10) { return; }

    let weight = exp(-lambda * (dist - shift));
    if (weight != weight || weight < 1e-30) { return; }  // NaN (WGSL has no isnan) or negligible

    let t0 = pat_tori[2 * pointIdx];
    let t1 = pat_tori[2 * pointIdx + 1];
    let center = t0.xyz;
    let R = t0.w;
    let axis = t1.xyz;
    let r = t1.w;

    // torus signed distance g_i(q)
    let rVec = q - center;
    let v = length(cross(rVec, axis));
    if (v < 1e-10) { return; }
    let u = vec2f(v - R, dot(rVec, axis));
    let u_len = length(u);
    if (u_len < 1e-10) { return; }
    let s = select(-1.0, 1.0, r >= 0.0);
    let g_z = s * (u_len - abs(r));

    // ∇g_i(q): G is the 2x3 Jacobian of (v - R, (q - c)·a) with respect to q
    let G_11 = (axis.x * axis.y * (center.y - q.y) + axis.x * axis.z * (center.z - q.z)
                - (axis.y * axis.y + axis.z * axis.z) * (center.x - q.x)) / v;
    let G_12 = (axis.x * axis.y * (center.x - q.x) + axis.y * axis.z * (center.z - q.z)
                - (axis.x * axis.x + axis.z * axis.z) * (center.y - q.y)) / v;
    let G_13 = (axis.x * axis.z * (center.x - q.x) + axis.y * axis.z * (center.y - q.y)
                - (axis.x * axis.x + axis.y * axis.y) * (center.z - q.z)) / v;
    let w = u / u_len;
    let grad_g = s * vec3f(w.x * G_11 + w.y * axis.x, w.x * G_12 + w.y * axis.y, w.x * G_13 + w.y * axis.z);

    let rHat = diff / dist;
    (*acc).grad_g_minus_lambda_n_g += weight * (grad_g - lambda * rHat * g_z);
    (*acc).g += weight * g_z;
    (*acc).n += weight * rHat;
    (*acc).normalization += weight;
}

// φ(q) and ∇φ(q) from the k nearest points
fn patDistanceGradient(q: vec3f, gradient: ptr<function, vec3f>) -> f32 {
    var indices: array<i32, BVH_K_MAX>;
    var dists: array<f32, BVH_K_MAX>;
    let count = bvhKNearestNeighbors(q, U.pat_k, &indices, &dists);
    *gradient = vec3f(0.0);
    if (count == 0) { return INFINITY; }

    var max_dist = 0.0;
    for (var i = 0; i < count; i++) {
        max_dist = max(max_dist, dists[i]);
    }
    let shift = 0.5 * max_dist;
    let lambda = U.pat_lambda_scale * U.pat_max_exp_arg / shift;

    var acc = TorusAccumulator(0.0, vec3f(0.0), vec3f(0.0), 0.0);
    for (var i = 0; i < count; i++) {
        accumulateTorusPoint(q, indices[i], lambda, shift, &acc);
    }

    let A = acc.grad_g_minus_lambda_n_g / acc.normalization;
    let distance = acc.g / acc.normalization;
    let B = lambda * acc.n / acc.normalization;
    *gradient = A + distance * B;
    return distance;
}

fn patDistance(q: vec3f) -> f32 {
    var gradient: vec3f;  // the gradient terms cost little next to the neighbor search
    return patDistanceGradient(q, &gradient);
}
