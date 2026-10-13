// camera.glsl — primary rays and depth for full-screen passes.
//   ray through a pixel: unproject the NDC points (x, y, -1) and (x, y, 1) with (P V)⁻¹
//   depth of a world point: z_ndc = (P V p).z / (P V p).w, window depth = ½ z_ndc + ½ (default depth range [0, 1])

uniform mat4 u_view_proj;      // P V
uniform mat4 u_inv_view_proj;  // (P V)⁻¹
uniform vec4 u_viewport;       // x, y, width, height in pixels

void cameraRay(in vec2 frag_xy, out vec3 ro, out vec3 rd) {
    vec2 ndc = 2.0 * (frag_xy - u_viewport.xy) / u_viewport.zw - 1.0;
    vec4 near = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 far = u_inv_view_proj * vec4(ndc, 1.0, 1.0);
    ro = near.xyz / near.w;
    rd = normalize(far.xyz / far.w - ro);
}

float fragDepth(in vec3 p) {
    vec4 clip = u_view_proj * vec4(p, 1.0);
    return clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
}

// slab test; t_near <= t_far when the ray (line) meets the box
bool intersectAABB(vec3 ro, vec3 rd, vec3 aabb_min, vec3 aabb_max, out float t_near, out float t_far) {
    vec3 inv_dir = 1.0 / rd;
    vec3 t0 = (aabb_min - ro) * inv_dir;
    vec3 t1 = (aabb_max - ro) * inv_dir;
    vec3 t_min = min(t0, t1);
    vec3 t_max = max(t0, t1);
    t_near = max(max(t_min.x, t_min.y), t_min.z);
    t_far = min(min(t_max.x, t_max.y), t_max.z);
    return t_near <= t_far && t_far >= 0.0;
}
