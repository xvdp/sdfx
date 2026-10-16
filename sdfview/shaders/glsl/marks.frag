// marks.frag — round sprites with a dark rim; u_alpha < 1 for the pass drawn without depth test (marks behind
// surfaces show faintly)
uniform float u_alpha;

in vec4 v_color;
layout(location = 0) out vec4 frag_color;

void main() {
    vec2 c = 2.0 * gl_PointCoord - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 col = r2 > 0.55 ? 0.3 * v_color.rgb : v_color.rgb;
    frag_color = vec4(col, v_color.a * u_alpha);
}
