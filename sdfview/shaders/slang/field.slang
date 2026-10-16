// field.slang — the field a program evaluates (mirror of sdfview/shaders/glsl/field.glsl).
//
// GLSL and WGSL select the field type with the preprocessor (FIELD_PAT / FIELD_GRID before #include "field.glsl");
// Slang with an interface: each field type is a struct implementing IField (field_pat.slang: PatField,
// field_grid.slang: GridField) that holds what the GLSL keeps in uniforms and textures, and the programs are generic
// over it (surface<F : IField>, probe<F : IField>). slangpy specializes a generic from the argument's '_type'
// ({'_type': 'PatField', ...}) or by name (module.require_function('probe<PatField>')).
//   fieldDistance(f, iso, p)              φ(p) - isovalue
//   fieldDistanceGradient(f, iso, p, g)   φ(p) - isovalue and ∇φ(p)
//   fieldPart(f, p)                       -1 (the GLSL's "glsl" field type, a SuperFit assembly, has no Slang mirror)
module field;

public interface IField
{
    float distance(float3 p);
    float distanceGradient(float3 p, out float3 gradient);
}

public float fieldDistance<F : IField>(F field, float isovalue, float3 p)
{
    return field.distance(p) - isovalue;
}

public float fieldDistanceGradient<F : IField>(F field, float isovalue, float3 p, out float3 g)
{
    return field.distanceGradient(p, g) - isovalue;
}

public int fieldPart<F : IField>(F field, float3 p)
{
    return -1;
}
