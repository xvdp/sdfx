// bvh.wgsl — bounding volume hierarchy over points: node fetch and k-nearest-neighbor traversal (WGSL counterpart of
// bvh.glsl, same functions and algorithm). Node layout (sdfview/scene.py): bounds entries 2i = (min.xyz, 0),
// 2i+1 = (max.xyz, 0); nodes[i] = (left, right, prim_start, prim_count), prim_count = 0 for internal nodes; root 0.
// Deviation from the GLSL: the arrays are the module's storage buffers (pat_bvh_bounds, pat_bvh_nodes, pat_bvh_prims,
// pat_points, declared in field_pat.wgsl) rather than parameters: WGSL passes storage buffers to functions only
// through pointers, which need the unrestricted_pointer_parameters language feature. The held set is returned
// through function-address-space pointers (GLSL: out arrays).
#include "common.wgsl"

struct BVHNode {
    aabb_min: vec3f,
    aabb_max: vec3f,
    left_idx: i32,
    right_idx: i32,
    prim_start: i32,
    prim_count: i32,
}

fn fetchBVHNode(nodeIdx: i32) -> BVHNode {
    let n = pat_bvh_nodes[nodeIdx];
    return BVHNode(pat_bvh_bounds[2 * nodeIdx].xyz, pat_bvh_bounds[2 * nodeIdx + 1].xyz, n.x, n.y, n.z, n.w);
}

// squared distance from a point to a box (0 inside)
fn pointAABBDistanceSq(p: vec3f, aabb_min: vec3f, aabb_max: vec3f) -> f32 {
    let diff = p - clamp(p, aabb_min, aabb_max);
    return dot(diff, diff);
}

fn nodeDistanceSq(query: vec3f, nodeIdx: i32) -> f32 {  // fetchBVHNodeAABB + pointAABBDistanceSq
    return pointAABBDistanceSq(query, pat_bvh_bounds[2 * nodeIdx].xyz, pat_bvh_bounds[2 * nodeIdx + 1].xyz);
}

// k nearest points to `query` (k <= BVH_K_MAX). Returns the number found, min(k, n); results are not sorted.
// Depth-first traversal, nearer child first; a node is skipped once k points are held and its box is farther than
// the current k-th distance. The held set is an unsorted array: a closer point replaces the farthest one.
fn bvhKNearestNeighbors(query: vec3f, k_in: i32, result_indices: ptr<function, array<i32, BVH_K_MAX>>,
                        result_dists: ptr<function, array<f32, BVH_K_MAX>>) -> i32 {
    let k = min(k_in, BVH_K_MAX);
    if (k <= 0) { return 0; }

    var count = 0;
    var kth_dist_sq = INFINITY;  // distance² to the k-th nearest held point (infinite until k are held)

    var stack: array<i32, BVH_STACK_SIZE>;
    var stack_ptr = 0;
    stack[stack_ptr] = 0;  // root
    stack_ptr++;

    while (stack_ptr > 0) {
        stack_ptr--;
        let node = fetchBVHNode(stack[stack_ptr]);

        // Prune: if the closest point of the box is farther than the k-th nearest, skip
        if (count >= k && pointAABBDistanceSq(query, node.aabb_min, node.aabb_max) > kth_dist_sq) {
            continue;
        }

        if (node.prim_count > 0) {
            // Leaf node: process points
            for (var i = 0; i < node.prim_count; i++) {
                let prim_idx = pat_bvh_prims[node.prim_start + i];
                let d = query - pat_points[prim_idx].xyz;
                let dist_sq = dot(d, d);

                if (count < k) {
                    // fewer than k held: append
                    (*result_indices)[count] = prim_idx;
                    (*result_dists)[count] = sqrt(dist_sq);
                    count++;
                    if (count == k) {
                        kth_dist_sq = 0.0;
                        for (var j = 0; j < k; j++) {
                            kth_dist_sq = max(kth_dist_sq, (*result_dists)[j] * (*result_dists)[j]);
                        }
                    }
                } else if (dist_sq < kth_dist_sq) {
                    // replace the farthest held point
                    var max_idx = 0;
                    var max_dist = (*result_dists)[0];
                    for (var j = 1; j < k; j++) {
                        if ((*result_dists)[j] > max_dist) {
                            max_dist = (*result_dists)[j];
                            max_idx = j;
                        }
                    }
                    (*result_indices)[max_idx] = prim_idx;
                    (*result_dists)[max_idx] = sqrt(dist_sq);

                    kth_dist_sq = 0.0;
                    for (var j = 0; j < k; j++) {
                        kth_dist_sq = max(kth_dist_sq, (*result_dists)[j] * (*result_dists)[j]);
                    }
                }
            }
        } else {
            // Internal node: push the farther child first so that the nearer one is processed first
            let left_dist_sq = nodeDistanceSq(query, node.left_idx);
            let right_dist_sq = nodeDistanceSq(query, node.right_idx);

            var near_idx = node.left_idx;
            var far_idx = node.right_idx;
            var near_dist_sq = left_dist_sq;
            var far_dist_sq = right_dist_sq;
            if (left_dist_sq >= right_dist_sq) {  // ties: right first, as in the original
                near_idx = node.right_idx; far_idx = node.left_idx;
                near_dist_sq = right_dist_sq; far_dist_sq = left_dist_sq;
            }
            if ((count < k || far_dist_sq <= kth_dist_sq) && stack_ptr < BVH_STACK_SIZE) {
                stack[stack_ptr] = far_idx;
                stack_ptr++;
            }
            if ((count < k || near_dist_sq <= kth_dist_sq) && stack_ptr < BVH_STACK_SIZE) {
                stack[stack_ptr] = near_idx;
                stack_ptr++;
            }
        }
    }
    return count;
}
