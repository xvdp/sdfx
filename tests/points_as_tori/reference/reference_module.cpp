// Reference build of the original PAT C++ code (ext/points-as-tori/src/cpp), used only by parity tests.
// Compiles the original sources unmodified and registers the same bindings as src/cpp/core.cpp, in a module named
// pat_reference. `has_cgal` reports whether igl::copyleft::cgal::point_areas is the original (CGAL) or the stub.
#include "core.h"

#ifndef PAT_REFERENCE_CGAL
#define PAT_REFERENCE_CGAL 0
#endif

NB_MODULE(pat_reference, m) {
    m.doc() = "Points as Tori C++ code, reference build for the torch port's tests";

    bind_shape_2d(m);
    bind_shape_3d(m);
    bind_torus_distance(m);
    bind_bvh(m);
    m.attr("has_cgal") = bool(PAT_REFERENCE_CGAL);
}
