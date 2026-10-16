// surface.wgsl — sphere tracing of the level set φ = isovalue inside the scene box (WGSL counterpart of
// fullscreen.vert + surface.frag; the same marching, bisection, clip half-space and shading).
// March t ← t + κ |φ(o + t d)|; a hit is |φ| < ε or a sign change between two samples, refined by bisection.
#include "common.wgsl"
#include "camera.wgsl"
#include "field.wgsl"

// one triangle covering the viewport, no vertex buffer: vertex 0, 1, 2 -> (-1,-1), (3,-1), (-1,3)
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(2.0 * p - 1.0, 0.0, 1.0);
}

// part colors: golden-ratio hues, pastel
fn partColor(i: i32) -> vec3f {
    let hue = fract(0.13 + 0.618034 * f32(i));
    let k = clamp(abs(fract(hue + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return mix(vec3f(1.0), k, 0.55);
}

fn shade(p: vec3f, n_in: vec3f, rd: vec3f) -> vec3f {
    let back = dot(n_in, rd) > 0.0;
    let n = select(n_in, -n_in, back);
    var base = select(U.u_color_front, U.u_color_back, back);
    if (U.u_color_parts != 0) {
        let part = fieldPart(p);
        if (part >= 0) { base = select(partColor(part), 0.6 * partColor(part), back); }
    }
    let l = normalize(-rd + 0.3 * vec3f(0.3, 1.0, 0.2));
    let h = normalize(l - rd);
    let diffuse = max(dot(n, l), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 48.0);
    return base * (0.2 + 0.8 * diffuse) + 0.25 * specular;
}

struct FragOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(@builtin(position) frag: vec4f) -> FragOut {
    let ray = cameraRay(frag.xy);
    let ro = ray.ro;
    let rd = ray.rd;
    let tt = intersectAABB(ro, rd, U.u_box_min, U.u_box_max);
    if (!(tt.x <= tt.y && tt.y >= 0.0)) { discard; }
    var t0 = max(tt.x, 0.0);
    var t1 = tt.y;

    if (U.u_clip != 0) {
        let denom = dot(U.u_clip_plane.xyz, rd);
        let s = U.u_clip_plane.w - dot(U.u_clip_plane.xyz, ro);  // n·(o + t d) <= c  <=>  t denom <= s
        if (denom > 0.0) {
            t1 = min(t1, s / denom);
        } else if (denom < 0.0) {
            t0 = max(t0, s / denom);
        } else if (s < 0.0) {
            discard;
        }
    }
    if (t0 >= t1) { discard; }

    var t = t0;
    var d = fieldDistance(ro + t * rd);
    var hit = false;
    for (var i = 0; i < U.u_max_steps; i++) {
        if (abs(d) < U.u_epsilon) { hit = true; break; }
        let t_prev = t;
        let d_prev = d;
        t += U.u_step_scale * max(abs(d), 0.5 * U.u_epsilon);
        if (t > t1) { break; }
        d = fieldDistance(ro + t * rd);
        if ((d < 0.0) != (d_prev < 0.0)) {
            // bracketed: bisection on [t_prev, t]
            var a = t_prev;
            var b = t;
            var da = d_prev;
            for (var j = 0; j < 10; j++) {
                let m = 0.5 * (a + b);
                let dm = fieldDistance(ro + m * rd);
                if ((dm < 0.0) == (da < 0.0)) { a = m; da = dm; } else { b = m; }
            }
            t = 0.5 * (a + b);
            hit = true;
            break;
        }
    }
    if (!hit) { discard; }

    let p = ro + t * rd;
    var g: vec3f;
    fieldDistanceGradient(p, &g);
    return FragOut(vec4f(shade(p, normalize(g), rd), U.u_opacity), fragDepth(p));
}
