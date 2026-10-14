// bvh.slang — bounding volume hierarchy over points: node fetch and k-nearest-neighbor traversal.
//
// Mirror of sdfview/shaders/glsl/bvh.glsl, function by function (same names, arguments and control flow), itself a
// port of fetchBVHNode / pointAABBDistanceSq / bvhKNearestNeighbors (ext/points-as-tori/demo/shaders_3d.py).
// Node layout (sdfview/scene.py, built by points_as_tori/bvh.py): bounds rows 2i = (min.xyz, 0), 2i+1 = (max.xyz, 0);
// integer row i = (left, right, prim_start, prim_count), prim_count = 0 for internal nodes; root = node 0.
// Differences from the GLSL: textures are slangpy tensors (common.slang); bvh_prims is the (N, 1) array read as
// bvh_prims[i, 0] (GLSL: fetch_texel(bvh_prims, i).x); BVH_STACK_SIZE and BVH_K_MAX are link-time constants set
// per call from Python (GLSL: #define before #include).
module bvh;

import slangpy;
__exported import common;

public extern static const int BVH_STACK_SIZE = 64;
public extern static const int BVH_K_MAX = 64;

public void fetchBVHNode(Tensor<float, 2> bvh_bounds, Tensor<int, 2> bvh_nodes, int nodeIdx, out float3 aabb_min,
                         out float3 aabb_max, out int left_idx, out int right_idx, out int prim_start,
                         out int prim_count)
{
    aabb_min = fetch_texel(bvh_bounds, 2 * nodeIdx).xyz;
    aabb_max = fetch_texel(bvh_bounds, 2 * nodeIdx + 1).xyz;
    int4 node = fetch_texel(bvh_nodes, nodeIdx);
    left_idx = node.x;
    right_idx = node.y;
    prim_start = node.z;
    prim_count = node.w;
}

public void fetchBVHNodeAABB(Tensor<float, 2> bvh_bounds, int nodeIdx, out float3 aabb_min, out float3 aabb_max)
{
    aabb_min = fetch_texel(bvh_bounds, 2 * nodeIdx).xyz;
    aabb_max = fetch_texel(bvh_bounds, 2 * nodeIdx + 1).xyz;
}

// squared distance from a point to a box (0 inside)
public float pointAABBDistanceSq(float3 point, float3 aabb_min, float3 aabb_max)
{
    float3 closest = clamp(point, aabb_min, aabb_max);
    float3 diff = point - closest;
    return dot(diff, diff);
}

// k nearest points to `query` (k <= BVH_K_MAX). Returns the number found, min(k, n); results are not sorted.
// Depth-first traversal, nearer child first; a node is skipped once k points are held and its box is farther than
// the current k-th distance. The held set is an unsorted array: a closer point replaces the farthest one.
public int bvhKNearestNeighbors(Tensor<float, 2> bvh_bounds, Tensor<int, 2> bvh_nodes, Tensor<int, 2> bvh_prims,
                                Tensor<float, 2> points, float3 query, int k, out int result_indices[BVH_K_MAX],
                                out float result_dists[BVH_K_MAX])
{
    // every slot defined (the GLSL leaves unused ones undefined; Slang requires out parameters to be assigned)
    for (int j = 0; j < BVH_K_MAX; j++) {
        result_indices[j] = -1;
        result_dists[j] = INFINITY;
    }
    k = min(k, BVH_K_MAX);
    if (k <= 0) return 0;

    int count = 0;
    float kth_dist_sq = INFINITY;  // distance² to the k-th nearest held point (infinite until k are held)

    int stack[BVH_STACK_SIZE];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // root

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];

        float3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(bvh_bounds, bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, prim_start,
                     prim_count);

        // Prune: if the closest point of the box is farther than the k-th nearest, skip
        if (count >= k && pointAABBDistanceSq(query, aabb_min, aabb_max) > kth_dist_sq) {
            continue;
        }

        if (prim_count > 0) {
            // Leaf node: process points
            for (int i = 0; i < prim_count; i++) {
                int prim_idx = bvh_prims[prim_start + i, 0];
                float3 point = fetch_texel(points, prim_idx).xyz;
                float dist_sq = dot(query - point, query - point);

                if (count < k) {
                    // fewer than k held: append
                    result_indices[count] = prim_idx;
                    result_dists[count] = sqrt(dist_sq);
                    count++;
                    if (count == k) {
                        kth_dist_sq = 0.0;
                        for (int j = 0; j < k; j++) {
                            kth_dist_sq = max(kth_dist_sq, result_dists[j] * result_dists[j]);
                        }
                    }
                } else if (dist_sq < kth_dist_sq) {
                    // replace the farthest held point
                    int max_idx = 0;
                    float max_dist = result_dists[0];
                    for (int j = 1; j < k; j++) {
                        if (result_dists[j] > max_dist) {
                            max_dist = result_dists[j];
                            max_idx = j;
                        }
                    }
                    result_indices[max_idx] = prim_idx;
                    result_dists[max_idx] = sqrt(dist_sq);

                    kth_dist_sq = 0.0;
                    for (int j = 0; j < k; j++) {
                        kth_dist_sq = max(kth_dist_sq, result_dists[j] * result_dists[j]);
                    }
                }
            }
        } else {
            // Internal node: push the farther child first so that the nearer one is processed first
            float3 left_min, left_max, right_min, right_max;
            fetchBVHNodeAABB(bvh_bounds, left_idx, left_min, left_max);
            fetchBVHNodeAABB(bvh_bounds, right_idx, right_min, right_max);
            float left_dist_sq = pointAABBDistanceSq(query, left_min, left_max);
            float right_dist_sq = pointAABBDistanceSq(query, right_min, right_max);

            int near_idx = left_idx, far_idx = right_idx;
            float near_dist_sq = left_dist_sq, far_dist_sq = right_dist_sq;
            if (left_dist_sq >= right_dist_sq) {  // ties: right first, as in the original
                near_idx = right_idx; far_idx = left_idx;
                near_dist_sq = right_dist_sq; far_dist_sq = left_dist_sq;
            }
            if ((count < k || far_dist_sq <= kth_dist_sq) && stack_ptr < BVH_STACK_SIZE) stack[stack_ptr++] = far_idx;
            if ((count < k || near_dist_sq <= kth_dist_sq) && stack_ptr < BVH_STACK_SIZE) stack[stack_ptr++] = near_idx;
        }
    }
    return count;
}
