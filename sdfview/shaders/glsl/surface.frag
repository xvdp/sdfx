// surface.frag — sphere tracing of the level set φ = isovalue inside the scene box.
//
// March t ← t + κ |φ(o + t d)| (κ = step scale ≤ 1: the PAT blend is not exactly 1-Lipschitz). A hit is |φ| < ε,
// or a sign change between two samples, refined by bisection (robust to steps that overshoot the surface).
// Optional clip half-space n·x ≤ c: the ray segment is restricted to it. A ray that enters the solid at the clip
// plane marches on to the far side of the solid (the cut plane, drawn at the same place, covers that opening).
// Shading: Blinn-Phong with a head light; front and back faces (seen from inside) get different colors, or the color
// of the part at the hit (u_color_parts); alpha = u_opacity (the renderer blends passes with u_opacity < 1 without
// writing depth).
#include "common.glsl"
#include "camera.glsl"
#include "field.glsl"

uniform vec3 u_box_min;
uniform vec3 u_box_max;
uniform int u_max_steps;
uniform float u_epsilon;
uniform float u_step_scale;
uniform bool u_clip;
uniform vec4 u_clip_plane;  // (n, c): keep n·x <= c
uniform vec3 u_color_front;
uniform vec3 u_color_back;
uniform float u_opacity;  // < 1: blended pass (overlay of several models)
uniform bool u_color_parts;  // color by fieldPart (fields with parts, e.g. a SuperFit assembly)

layout(location = 0) out vec4 frag_color;

// part colors: golden-ratio hues, pastel
vec3 partColor(int i) {
    float hue = fract(0.13 + 0.618034 * float(i));
    vec3 k = clamp(abs(fract(hue + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), k, 0.55);
}

vec3 shade(vec3 p, vec3 n, vec3 rd) {
    bool back = dot(n, rd) > 0.0;
    n = back ? -n : n;
    vec3 base = back ? u_color_back : u_color_front;
    if (u_color_parts) {
        int part = fieldPart(p);
        if (part >= 0) base = back ? 0.6 * partColor(part) : partColor(part);
    }
    vec3 l = normalize(-rd + 0.3 * vec3(0.3, 1.0, 0.2));
    vec3 h = normalize(l - rd);
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(n, h), 0.0), 48.0);
    return base * (0.2 + 0.8 * diffuse) + 0.25 * specular;
}

void main() {
    vec3 ro, rd;
    cameraRay(gl_FragCoord.xy, ro, rd);
    float t0, t1;
    if (!intersectAABB(ro, rd, u_box_min, u_box_max, t0, t1)) discard;
    t0 = max(t0, 0.0);

    if (u_clip) {
        float denom = dot(u_clip_plane.xyz, rd);
        float s = u_clip_plane.w - dot(u_clip_plane.xyz, ro);  // n·(o + t d) <= c  <=>  t denom <= s
        if (denom > 0.0) {
            t1 = min(t1, s / denom);
        } else if (denom < 0.0) {
            t0 = max(t0, s / denom);
        } else if (s < 0.0) {
            discard;
        }
    }
    if (t0 >= t1) discard;

    float t = t0;
    float d = fieldDistance(ro + t * rd);
    bool hit = false;
    for (int i = 0; i < u_max_steps; i++) {
        if (abs(d) < u_epsilon) { hit = true; break; }
        float t_prev = t;
        float d_prev = d;
        t += u_step_scale * max(abs(d), 0.5 * u_epsilon);
        if (t > t1) break;
        d = fieldDistance(ro + t * rd);
        if ((d < 0.0) != (d_prev < 0.0)) {
            // bracketed: bisection on [t_prev, t]
            float a = t_prev, b = t, da = d_prev;
            for (int j = 0; j < 10; j++) {
                float m = 0.5 * (a + b);
                float dm = fieldDistance(ro + m * rd);
                if ((dm < 0.0) == (da < 0.0)) { a = m; da = dm; } else { b = m; }
            }
            t = 0.5 * (a + b);
            hit = true;
            break;
        }
    }
    if (!hit) discard;

    vec3 p = ro + t * rd;
    vec3 g;
    fieldDistanceGradient(p, g);
    frag_color = vec4(shade(p, normalize(g), rd), u_opacity);
    gl_FragDepth = fragDepth(p);
}
