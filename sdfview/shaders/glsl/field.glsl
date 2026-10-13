// field.glsl — selects the field a program evaluates: compile with FIELD_PAT or FIELD_GRID defined.
//   float fieldDistance(vec3 p)                       φ(p) - isovalue
//   float fieldDistanceGradient(vec3 p, out vec3 g)  φ(p) - isovalue and ∇φ(p)
#include "common.glsl"

uniform float field_isovalue;

#if defined(FIELD_PAT)
#include "field_pat.glsl"
float fieldDistance(vec3 p) { return patDistance(p) - field_isovalue; }
float fieldDistanceGradient(vec3 p, out vec3 g) { return patDistanceGradient(p, g) - field_isovalue; }
#elif defined(FIELD_GRID)
#include "field_grid.glsl"
float fieldDistance(vec3 p) { return gridDistance(p) - field_isovalue; }
float fieldDistanceGradient(vec3 p, out vec3 g) { return gridDistanceGradient(p, g) - field_isovalue; }
#else
#error "define FIELD_PAT or FIELD_GRID"
#endif
