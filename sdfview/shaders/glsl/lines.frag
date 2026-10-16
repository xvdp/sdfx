// lines.frag — flat color; u_alpha < 1 for the pass drawn without depth test
uniform float u_alpha;

in vec4 v_color;
layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(v_color.rgb, v_color.a * u_alpha);
}
