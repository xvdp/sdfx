// marks.vert — highlighted points (tori inspector): point sprites from vertex attributes, sized as points.vert
// (diameter 2 r · P[1][1] · (h / 2) / w_clip pixels, at least 3 pixels so small marks stay visible).
uniform mat4 u_view_proj;
uniform vec4 u_viewport;
uniform float u_proj11;

in vec3 in_position;
in float in_radius;
in vec4 in_color;

out vec4 v_color;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    gl_PointSize = clamp(in_radius * u_proj11 * u_viewport.w / gl_Position.w, 3.0, 64.0);
    v_color = in_color;
}
