// points.frag — round sprites; with normals, shaded by |n · view| (front/back tinted), else flat
uniform vec3 u_camera_dir;  // world-space view direction
uniform vec3 u_point_color;
uniform bool u_has_normals;

in vec3 v_normal;
in float v_outlier;
layout(location = 0) out vec4 frag_color;

void main() {
    vec2 c = 2.0 * gl_PointCoord - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 col = v_outlier > 0.5 ? vec3(0.85, 0.1, 0.1) : u_point_color;
    float shade = 1.0 - 0.35 * r2;  // sphere-like falloff
    if (u_has_normals) {
        float nd = dot(normalize(v_normal), -u_camera_dir);
        col = nd >= 0.0 ? col : mix(col, vec3(0.55), 0.6);  // facing away: grayed
        shade *= 0.45 + 0.55 * abs(nd);
    }
    frag_color = vec4(col * shade, 1.0);
}
