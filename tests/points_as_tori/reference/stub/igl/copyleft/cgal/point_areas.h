// Test-only stub: shadows libigl's CGAL-backed point_areas so the reference
// module builds without CGAL. Point areas are not used on the inference path
// (signed distance / gradient / Laplacian / torus fitting).
#pragma once
#include <Eigen/Core>
#include <stdexcept>
namespace igl { namespace copyleft { namespace cgal {
template <typename DerivedP, typename DerivedI, typename DerivedN, typename DerivedA>
inline void point_areas(const Eigen::MatrixBase<DerivedP>&, const Eigen::MatrixBase<DerivedI>&,
                        const Eigen::MatrixBase<DerivedN>&, Eigen::PlainObjectBase<DerivedA>&) {
    throw std::runtime_error("point_areas: CGAL stub (reference build)");
}
}}}
