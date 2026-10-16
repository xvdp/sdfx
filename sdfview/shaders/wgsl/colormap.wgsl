// colormap.wgsl — signed-distance colormap with isobands (WGSL counterpart of colormap.glsl; the formulas are there).
// As in the GLSL, sdfColor takes fw = fwidth(φ), computed by the fragment shader before any discard (WGSL requires
// derivatives in uniform control flow; in GLSL they are undefined after a discard in the 2x2 quad).

fn segment(edge0: f32, edge1: f32, x: f32) -> f32 {
    return step(edge0, x) * (1.0 - step(edge1, x));
}

fn normalize_scalar(minVal: f32, maxVal: f32, val: f32) -> f32 {
    return (val - minVal) / (maxVal - minVal);
}

fn colormap_PasadenaBlue(t: f32) -> vec3f {
    let blue = vec3f(0.0627451, 0.517647, 0.94902);
    let white = vec3f(1.0, 1.0, 1.0);
    let tp = pow(t, 7.0);
    return (1.0 - tp) * white + tp * blue;
}

fn colormap_grapefruit(t: f32) -> vec3f {
    let grapefruit = vec3f(0.992157, 0.431373, 0.337255);
    let white = vec3f(1.0, 1.0, 1.0);
    return mix(white, grapefruit, t);
}

// as the original: values outside [min, max] (t outside [0, 1]) map to black
fn colormap_SDF(t: f32) -> vec3f {
    return segment(0.0, 0.5, t) * colormap_PasadenaBlue(2.0 * (t - 0.0)) +
           segment(0.5, 1.0, t) * colormap_grapefruit(2.0 * (t - 0.5));
}

// scale a color, redistributing overflow so the hue is preserved
fn redistribute_rgb(color: vec3f) -> vec3f {
    let m = max(max(color.x, color.y), color.z);
    if (m <= 1.0) { return color; }
    let total = color.x + color.y + color.z;
    if (total >= 3.0) { return vec3f(1.0); }
    let x = (3.0 - total) / (3.0 * m - total);
    let gray = 1.0 - x * m;
    return vec3f(gray) + x * color;
}

fn darken_color(color: vec3f, t: f32) -> vec3f {
    return redistribute_rgb(t * color);
}

// color of a field value: colormap over [-range, range], isobands of frequency `freq`, zero contour of ~1.5 px
fn sdfColor(phi: f32, fw: f32, range: f32, freq: f32) -> vec3f {
    var dark_factor = 0.9;
    dark_factor *= 1.0 - 0.1 * smoothstep(-0.01, 0.01, sin(-freq * phi));
    let col = darken_color(colormap_SDF(normalize_scalar(-range, range, phi)), dark_factor);
    let line = 1.0 - smoothstep(0.5, 1.5, abs(phi) / max(fw, 1e-12));
    return mix(col, vec3f(0.1), line);
}
