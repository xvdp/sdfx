// blit.frag — copy a color texture to the viewport (scaled, linear filtering)
uniform sampler2D u_image;
uniform vec4 u_viewport;
layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = texture(u_image, (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw);
}
