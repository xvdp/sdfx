// probe.frag — field evaluation at query points (parity tests / picking): texel i of u_queries holds query i,
// output texel i = (φ(q_i) - isovalue, ∇φ(q_i)). The output has the size of the query texture.
#include "common.glsl"
#include "field.glsl"

uniform sampler2D u_queries;
uniform int u_count;

layout(location = 0) out vec4 frag_out;

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    int idx = xy.y * textureSize(u_queries, 0).x + xy.x;
    if (idx >= u_count) { frag_out = vec4(0.0); return; }
    vec3 g;
    float d = fieldDistanceGradient(fetch_texel(u_queries, idx).xyz, g);
    frag_out = vec4(d, g);
}
