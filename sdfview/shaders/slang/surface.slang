// surface.slang — sphere tracing of the level set φ = isovalue inside the scene box (mirror of surface.frag).
//
// The method and shading are documented in sdfview/shaders/glsl/surface.frag. Differences from the GLSL program:
//   - a compute function called once per pixel (slangpy: spy.grid((h, w)) gives pixel = (x, y)), not a fragment
//     shader: the host's fixed-function steps are written out here: `discard` returns without writing, the depth test
//     (less) reads depth[y, x], blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA, opacity < 1: no depth write) is a lerp;
//   - the uniforms are grouped in structs (Camera, Trace, Shading) and the field is a generic IField argument
//     (field.slang), in place of FIELD_PAT / FIELD_GRID compile variants.
// Image rows: y = 0 is the bottom row (window coordinates, as glReadPixels).
module surface;

import slangpy;
import common;
__exported import camera;
__exported import field;
__exported import field_grid;
__exported import field_pat;

public struct Trace  // GLSL: u_box_min, u_box_max, u_max_steps, u_epsilon, u_step_scale, u_clip, u_clip_plane,
{                    //       field_isovalue
    public float3 box_min;
    public float3 box_max;
    public int max_steps;
    public float epsilon;
    public float step_scale;
    public int clip;           // 0 / 1 (GLSL: bool u_clip)
    public float4 clip_plane;  // (n, c): keep n·x <= c
    public float isovalue;
}

public struct Shading  // GLSL: u_color_front, u_color_back, u_opacity, u_color_parts
{
    public float3 color_front;
    public float3 color_back;
    public float opacity;
    public int color_parts;  // 0 / 1
}

// part colors: golden-ratio hues, pastel
public float3 partColor(int i)
{
    float hue = frac(0.13 + 0.618034 * float(i));
    float3 k = clamp(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return lerp(float3(1.0), k, 0.55);
}

public float3 shade<F : IField>(F field, Shading sh, float3 p, float3 n, float3 rd)
{
    bool back = dot(n, rd) > 0.0;
    n = back ? -n : n;
    float3 base = back ? sh.color_back : sh.color_front;
    if (sh.color_parts != 0) {
        int part = fieldPart(field, p);
        if (part >= 0) base = back ? 0.6 * partColor(part) : partColor(part);
    }
    float3 l = normalize(-rd + 0.3 * float3(0.3, 1.0, 0.2));
    float3 h = normalize(l - rd);
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(n, h), 0.0), 48.0);
    return base * (0.2 + 0.8 * diffuse) + 0.25 * specular;
}

// main() of surface.frag up to the shading: false where the GLSL discards; else the hit point p on the ray (ro, rd)
public bool traceSurface<F : IField>(F field, Trace tr, Camera cam, float2 frag_xy, out float3 p, out float3 rd)
{
    float3 ro;
    cameraRay(cam, frag_xy, ro, rd);
    p = ro;
    float t0, t1;
    if (!intersectAABB(ro, rd, tr.box_min, tr.box_max, t0, t1)) return false;
    t0 = max(t0, 0.0);

    if (tr.clip != 0) {
        float denom = dot(tr.clip_plane.xyz, rd);
        float s = tr.clip_plane.w - dot(tr.clip_plane.xyz, ro);  // n·(o + t d) <= c  <=>  t denom <= s
        if (denom > 0.0) {
            t1 = min(t1, s / denom);
        } else if (denom < 0.0) {
            t0 = max(t0, s / denom);
        } else if (s < 0.0) {
            return false;
        }
    }
    if (t0 >= t1) return false;

    float t = t0;
    float d = fieldDistance(field, tr.isovalue, ro + t * rd);
    bool hit = false;
    for (int i = 0; i < tr.max_steps; i++) {
        if (abs(d) < tr.epsilon) { hit = true; break; }
        float t_prev = t;
        float d_prev = d;
        t += tr.step_scale * max(abs(d), 0.5 * tr.epsilon);
        if (t > t1) break;
        d = fieldDistance(field, tr.isovalue, ro + t * rd);
        if ((d < 0.0) != (d_prev < 0.0)) {
            // bracketed: bisection on [t_prev, t]
            float a = t_prev, b = t, da = d_prev;
            for (int j = 0; j < 10; j++) {
                float m = 0.5 * (a + b);
                float dm = fieldDistance(field, tr.isovalue, ro + m * rd);
                if ((dm < 0.0) == (da < 0.0)) { a = m; da = dm; } else { b = m; }
            }
            t = 0.5 * (a + b);
            hit = true;
            break;
        }
    }
    p = ro + t * rd;
    return hit;
}

// one surface pass at one pixel: trace, shade, depth test, write or blend
public void surface<F : IField>(int2 pixel, F field, Trace tr, Shading sh, Camera cam, RWTensor<float, 3> color,
                                RWTensor<float, 2> depth)
{
    float3 p, rd;
    if (!traceSurface(field, tr, cam, float2(pixel) + 0.5, p, rd)) return;  // discard
    float z = fragDepth(cam, p);
    if (!(z < depth[pixel.y, pixel.x])) return;  // depth test (less)

    float3 g;
    fieldDistanceGradient(field, tr.isovalue, p, g);
    float4 src = float4(shade(field, sh, p, normalize(g), rd), sh.opacity);
    if (sh.opacity >= 1.0) {
        for (int c = 0; c < 4; c++) color[pixel.y, pixel.x, c] = src[c];
        depth[pixel.y, pixel.x] = z;
    } else {  // blended: src·α + dst·(1 - α) on every channel, depth not written
        for (int c = 0; c < 4; c++) color[pixel.y, pixel.x, c] = lerp(color[pixel.y, pixel.x, c], src[c], sh.opacity);
    }
}
