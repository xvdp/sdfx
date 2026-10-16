// field.wgsl — selects the field a program evaluates (WGSL counterpart of field.glsl): assembled with FIELD_PAT or
// FIELD_GRID defined (sdfview.shaders.wgsl_source / web/core/shaders.js resolve the conditionals).
//   fieldDistance(p) = φ(p) - isovalue;  fieldDistanceGradient(p, &g);  fieldPart(p) (-1: no parts)
// FIELD_GLSL (generated GLSL, e.g. SuperFit assemblies) has no WGSL counterpart yet: such models are shown through
// their grid bake.
#include "common.wgsl"

#if defined(FIELD_PAT)
#include "field_pat.wgsl"
fn fieldDistance(p: vec3f) -> f32 { return patDistance(p) - U.field_isovalue; }
fn fieldDistanceGradient(p: vec3f, g: ptr<function, vec3f>) -> f32 { return patDistanceGradient(p, g) - U.field_isovalue; }
fn fieldPart(p: vec3f) -> i32 { return -1; }
#elif defined(FIELD_GRID)
#include "field_grid.wgsl"
fn fieldDistance(p: vec3f) -> f32 { return gridDistance(p) - U.field_isovalue; }
fn fieldDistanceGradient(p: vec3f, g: ptr<function, vec3f>) -> f32 { return gridDistanceGradient(p, g) - U.field_isovalue; }
fn fieldPart(p: vec3f) -> i32 { return -1; }
#else
#error "define FIELD_PAT or FIELD_GRID (FIELD_GLSL: generated GLSL, not available in WGSL)"
#endif
