// knn.slang — fused k-nearest-neighbor kernels: one thread per query runs the whole BVH traversal (bvh.slang),
// orders its k results and writes them out. Called from points_as_tori/knn_slang.py through slangpy: `query` is
// vectorized over a (Q, 3) tensor (one thread per row), `query_id` is the thread's row (spy.call_id()), every other
// argument is passed whole.
//
//   knn       held set as a max-heap (bvhKNearestNeighborsHeap, as the original C++ BVH): O(log k) per accepted
//             candidate; heapsort at the end; float
//   knn_f64   the same in double (bounds, points, queries and distances in double; optional, config.SLANG_FLOAT64)
//   knn_scan  held set as an unsorted array (bvhKNearestNeighbors, as sdfview/shaders/glsl/bvh.glsl and the original
//             demo shader): O(k) per accepted candidate; insertion sort at the end
//
// Output row query_id: out_indices[query_id, j], out_dists[query_id, j] for j < min(k, N), nearest first; equal
// distances ordered by point index.
module knn;

import slangpy;
import bvh;

// heap kernel in precision T: traversal, heapsort, ties by index, write row query_id (distances, not squared)
void knnHeap<T : __BuiltinFloatingPointType>(int query_id, vector<T, 3> query, Tensor<T, 2> bvh_bounds,
                                             Tensor<int, 2> bvh_nodes, Tensor<int, 2> bvh_prims, Tensor<T, 2> points,
                                             int k, RWTensor<int, 2> out_indices, RWTensor<T, 2> out_dists)
{
    int heap_i[BVH_K_MAX];
    T heap_d[BVH_K_MAX];  // squared distances
    int count = bvhKNearestNeighborsHeap<T>(bvh_bounds, bvh_nodes, bvh_prims, points, query, k, heap_i, heap_d);

    // heapsort: move the root (largest) to the end of the shrinking heap; ascending afterwards
    for (int end = count - 1; end > 0; end--) {
        T d = heap_d[end];
        int id = heap_i[end];
        heap_d[end] = heap_d[0];
        heap_i[end] = heap_i[0];
        int c = 0;
        while (true) {
            int child = 2 * c + 1;
            if (child >= end) break;
            if (child + 1 < end && heap_d[child + 1] > heap_d[child]) child++;
            if (heap_d[child] <= d) break;
            heap_d[c] = heap_d[child];
            heap_i[c] = heap_i[child];
            c = child;
        }
        heap_d[c] = d;
        heap_i[c] = id;
    }
    // equal distances by index: insertion sort, O(count + inversions) on the distance-sorted array
    for (int i = 1; i < count; i++) {
        T d = heap_d[i];
        int id = heap_i[i];
        int j = i - 1;
        while (j >= 0 && heap_d[j] == d && heap_i[j] > id) {
            heap_d[j + 1] = heap_d[j];
            heap_i[j + 1] = heap_i[j];
            j--;
        }
        heap_d[j + 1] = d;
        heap_i[j + 1] = id;
    }

    for (int j = 0; j < count; j++) {
        out_indices[query_id, j] = heap_i[j];
        out_dists[query_id, j] = sqrt(heap_d[j]);
    }
}

public void knn(int query_id, float3 query, Tensor<float, 2> bvh_bounds, Tensor<int, 2> bvh_nodes,
                Tensor<int, 2> bvh_prims, Tensor<float, 2> points, int k, RWTensor<int, 2> out_indices,
                RWTensor<float, 2> out_dists)
{
    knnHeap<float>(query_id, query, bvh_bounds, bvh_nodes, bvh_prims, points, k, out_indices, out_dists);
}

public void knn_f64(int query_id, double3 query, Tensor<double, 2> bvh_bounds, Tensor<int, 2> bvh_nodes,
                    Tensor<int, 2> bvh_prims, Tensor<double, 2> points, int k, RWTensor<int, 2> out_indices,
                    RWTensor<double, 2> out_dists)
{
    knnHeap<double>(query_id, query, bvh_bounds, bvh_nodes, bvh_prims, points, k, out_indices, out_dists);
}

public void knn_scan(int query_id, float3 query, Tensor<float, 2> bvh_bounds, Tensor<int, 2> bvh_nodes,
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
