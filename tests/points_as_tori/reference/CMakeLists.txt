# Reference build of the original PAT C++ code (ext/points-as-tori/src/cpp), for parity tests only.
# Same sources and bindings as the original module (src/cpp/core.cpp), named pat_reference.
#   cmake -S tests/points_as_tori/reference -B build/pat_reference \
#         -Dnanobind_DIR=$(python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python)
#   cmake --build build/pat_reference -j
# With CGAL installed (e.g. apt install libcgal-dev, conda install -c conda-forge cgal-cpp) the point areas are the
# original igl::copyleft::cgal::point_areas; without it a stub that throws is used and the point-area tests skip
# (pat_reference.has_cgal tells which).
cmake_minimum_required(VERSION 3.15)
project(pat_reference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(PAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../ext/points-as-tori)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)
find_package(CGAL QUIET)

nanobind_add_module(pat_reference
    reference_module.cpp
    ${PAT_ROOT}/src/cpp/torus_distance.cpp
    ${PAT_ROOT}/src/cpp/shape_2d.cpp
    ${PAT_ROOT}/src/cpp/shape_3d.cpp
    ${PAT_ROOT}/src/cpp/bvh.cpp
)
set(PAT_INCLUDES
    ${PAT_ROOT}/src/cpp
    ${PAT_ROOT}/deps/fcpw/deps/eigen
    ${PAT_ROOT}/deps/fcpw/include
    ${PAT_ROOT}/deps/nanoflann/include
    ${PAT_ROOT}/deps/libigl/include
)
if(CGAL_FOUND)
    message(STATUS "pat_reference: CGAL ${CGAL_VERSION} found, original point_areas")
    target_include_directories(pat_reference PRIVATE ${PAT_INCLUDES})
    target_link_libraries(pat_reference PRIVATE CGAL::CGAL)
    target_compile_definitions(pat_reference PRIVATE PAT_REFERENCE_CGAL=1)
else()
    message(STATUS "pat_reference: CGAL not found, point_areas stubbed")
    # stub/ first so it shadows igl/copyleft/cgal/point_areas.h
    target_include_directories(pat_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stub ${PAT_INCLUDES})
    target_compile_definitions(pat_reference PRIVATE PAT_REFERENCE_CGAL=0)
endif()
target_compile_options(pat_reference PRIVATE -w)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pat_reference PRIVATE OpenMP::OpenMP_CXX)
endif()
