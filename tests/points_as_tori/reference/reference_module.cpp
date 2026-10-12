// Reference build of the original PAT C++ evaluator (ext/points-as-tori/src/cpp),
// used only by parity tests. Compiles the original torus_distance.cpp and
// shape_2d.cpp unmodified; point_cloud_bounding_box is copied verbatim from
// shape_3d.cpp (which otherwise pulls in mesh/fcpw code not needed here).
#include <nanobind/nanobind.h>
#include "shape_3d.h"

namespace nb = nanobind;

std::pair<Eigen::Vector3d, Eigen::Vector3d>
point_cloud_bounding_box(const Eigen::Matrix<double, 3, Eigen::Dynamic>& positions) {
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector3d bbox_min = {inf, inf, inf};
    Eigen::Vector3d bbox_max = {-inf, -inf, -inf};
    int n_points = positions.cols();
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < 3; j++) {
            bbox_min[j] = std::min(bbox_min[j], positions(j, i));
            bbox_max[j] = std::max(bbox_max[j], positions(j, i));
        }
    }
    return {bbox_min, bbox_max};
}

void bind_torus_distance(nb::module_& m);
void bind_shape_2d(nb::module_& m);

NB_MODULE(pat_reference, m) {
    bind_torus_distance(m);
}
