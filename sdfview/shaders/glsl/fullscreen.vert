// fullscreen.vert — one triangle covering the viewport, no vertex buffer: vertex 0, 1, 2 -> (-1,-1), (3,-1), (-1,3)
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
}
