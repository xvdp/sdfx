// points.vert — point sprites read from textures by gl_VertexID (no vertex buffer).
// Sprite diameter in pixels: 2 r · P[1][1] · (h / 2) / w_clip  (projected world radius r, viewport height h).
#include "common.glsl"

uniform mat4 u_view_proj;
uniform vec4 u_viewport;
uniform float u_proj11;      // P[1][1] = cot(fov_y / 2)
uniform float u_point_radius;
uniform sampler2D u_positions;  // (x, y, z, outlier)
uniform sampler2D u_normals;    // (nx, ny, nz, 0)
uniform bool u_has_normals;

out vec3 v_normal;
out float v_outlier;

void main() {
    vec4 p = fetch_texel(u_positions, gl_VertexID);
    gl_Position = u_view_proj * vec4(p.xyz, 1.0);
    gl_PointSize = clamp(u_point_radius * u_proj11 * u_viewport.w / gl_Position.w, 1.0, 64.0);
    v_normal = u_has_normals ? fetch_texel(u_normals, gl_VertexID).xyz : vec3(0.0);
    v_outlier = p.w;
}
