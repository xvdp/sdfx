// knn.slang — fused k-nearest-neighbor kernel: one thread per query runs the whole BVH traversal
// (bvhKNearestNeighbors, bvh.slang), sorts its k results and writes them out. Called from points_as_tori/knn_slang.py
// through slangpy: `query` is vectorized over a (Q, 3) tensor (one thread per row), `query_id` is the thread's row
// (spy.call_id()), every other argument is passed whole.
//
// Output row query_id: out_indices[query_id, j], out_dists[query_id, j] for j < min(k, N), nearest first; equal
// distances ordered by point index.
module knn;

import slangpy;
import bvh;

public void knn(int query_id, float3 query, Tensor<float, 2> bvh_bounds, Tensor<int, 2> bvh_nodes,
                Tensor<int, 2> bvh_prims, Tensor<float, 2> points, int k, RWTensor<int, 2> out_indices,
                RWTensor<float, 2> out_dists)
{
    int result_indices[BVH_K_MAX];
    float result_dists[BVH_K_MAX];
    int count = bvhKNearestNeighbors(bvh_bounds, bvh_nodes, bvh_prims, points, query, k, result_indices,
                                     result_dists);

    // insertion sort by (distance, index): count <= BVH_K_MAX, O(count²) compare-and-shift in registers / local
    // memory, no extra storage
    for (int i = 1; i < count; i++) {
        float d = result_dists[i];
        int id = result_indices[i];
        int j = i - 1;
        while (j >= 0 && (result_dists[j] > d || (result_dists[j] == d && result_indices[j] > id))) {
            result_dists[j + 1] = result_dists[j];
            result_indices[j + 1] = result_indices[j];
            j--;
        }
        result_dists[j + 1] = d;
        result_indices[j + 1] = id;
    }

    for (int j = 0; j < count; j++) {
        out_indices[query_id, j] = result_indices[j];
        out_dists[query_id, j] = result_dists[j];
    }
}
