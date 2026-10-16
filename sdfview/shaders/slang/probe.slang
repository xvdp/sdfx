// probe.slang — field evaluation at query points (mirror of probe.frag): (φ(q) - isovalue, ∇φ(q)).
// slangpy vectorizes `q` over a (n, 3) tensor, one thread per query, and returns (n, 4); the GLSL reads the queries
// from a texture and writes one texel per query.
module probe;

__exported import field;
__exported import field_grid;
__exported import field_pat;

public float4 probe<F : IField>(F field, float isovalue, float3 q)
{
    float3 g;
    float d = fieldDistanceGradient(field, isovalue, q, g);
    return float4(d, g);
}
