// cutplane.vert — a rectangle (triangle strip of 4 vertices, no vertex buffer): origin ± u ± v
uniform mat4 u_view_proj;
uniform vec3 u_plane_origin;
uniform vec3 u_plane_u;  // half extent along the first in-plane axis
uniform vec3 u_plane_v;  // half extent along the second in-plane axis

out vec3 v_world;

void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    v_world = u_plane_origin + c.x * u_plane_u + c.y * u_plane_v;
    gl_Position = u_view_proj * vec4(v_world, 1.0);
}
