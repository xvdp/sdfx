// lines.vert — line segments from vertex attributes (tori inspector: principal circles, directions, axis)
uniform mat4 u_view_proj;

in vec3 in_position;
in vec4 in_color;

out vec4 v_color;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_color = in_color;
}
