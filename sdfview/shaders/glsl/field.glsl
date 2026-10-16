// field.glsl — selects the field a program evaluates: compile with FIELD_PAT, FIELD_GRID or FIELD_GLSL defined.
//   float fieldDistance(vec3 p)                       φ(p) - isovalue
//   float fieldDistanceGradient(vec3 p, out vec3 g)  φ(p) - isovalue and ∇φ(p)
//   int fieldPart(vec3 p)                            part shown at p (FIELD_GLSL: the selected part, or the nearest
//                                                    part of the assembly); -1 for fields without parts
#include "common.glsl"

uniform float field_isovalue;

#if defined(FIELD_PAT)
#include "field_pat.glsl"
float fieldDistance(vec3 p) { return patDistance(p) - field_isovalue; }
float fieldDistanceGradient(vec3 p, out vec3 g) { return patDistanceGradient(p, g) - field_isovalue; }
int fieldPart(vec3 p) { return -1; }
#elif defined(FIELD_GRID)
#include "field_grid.glsl"
float fieldDistance(vec3 p) { return gridDistance(p) - field_isovalue; }
float fieldDistanceGradient(vec3 p, out vec3 g) { return gridDistanceGradient(p, g) - field_isovalue; }
int fieldPart(vec3 p) { return -1; }
#elif defined(FIELD_GLSL)
// Generated code (scene field type "glsl", inserted by the host before the modules, e.g. superfit_sdf/glsl.py):
// assemblyDistance(p), assemblyPartDistance(i, p), assemblyParts(p, out id), assemblyNearestPart(p).
uniform int field_part;              // -1: the assembly; -2: its parts as separate solids (min); i >= 0: part i alone
uniform float field_gradient_step;   // central differences step

float glslDistance(vec3 p) {
    if (field_part >= 0) return assemblyPartDistance(field_part, p);
    if (field_part == -2) { int id; return assemblyParts(p, id); }
    return assemblyDistance(p);
}
float fieldDistance(vec3 p) { return glslDistance(p) - field_isovalue; }
float fieldDistanceGradient(vec3 p, out vec3 g) {
    float h = field_gradient_step;
    g = vec3(glslDistance(p + vec3(h, 0.0, 0.0)) - glslDistance(p - vec3(h, 0.0, 0.0)),
             glslDistance(p + vec3(0.0, h, 0.0)) - glslDistance(p - vec3(0.0, h, 0.0)),
             glslDistance(p + vec3(0.0, 0.0, h)) - glslDistance(p - vec3(0.0, 0.0, h))) / (2.0 * h);
    return glslDistance(p) - field_isovalue;
}
int fieldPart(vec3 p) { return field_part >= 0 ? field_part : assemblyNearestPart(p); }
#else
#error "define FIELD_PAT, FIELD_GRID or FIELD_GLSL"
#endif
