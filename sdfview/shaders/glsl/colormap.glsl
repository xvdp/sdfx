// colormap.glsl — signed-distance colormap with isobands, from ext/points-as-tori/demo/shaders_3d.py
// (colormap_SDF, darken_color, cutplaneColor): t = (φ - min) / (max - min); t < ½: white→blue with weight (2t)^7
// (blue just inside the surface), t > ½: white→grapefruit, linear in 2t - 1; isobands darken by 10% where
// sin(-f φ) > 0. Added here: the zero level set drawn as a line ~1.5 px wide (screen-space derivative fwidth).

float segment(float edge0, float edge1, float x) {
    return step(edge0, x) * (1.0 - step(edge1, x));
}

float normalize_scalar(float minVal, float maxVal, float val) {
    return (val - minVal) / (maxVal - minVal);
}

vec3 colormap_PasadenaBlue(float t) {
    vec3 blue = vec3(0.0627451, 0.517647, 0.94902);
    vec3 white = vec3(1.0, 1.0, 1.0);
    float tp = pow(t, 7.0);
    return (1.0 - tp) * white + tp * blue;
}

vec3 colormap_grapefruit(float t) {
    vec3 grapefruit = vec3(0.992157, 0.431373, 0.337255);
    vec3 white = vec3(1.0, 1.0, 1.0);
    return mix(white, grapefruit, t);
}

// as the original: values outside [min, max] (t outside [0, 1]) map to black
vec3 colormap_SDF(float t) {
    return segment(0.0, 0.5, t) * colormap_PasadenaBlue(2.0 * (t - 0.0)) +
           segment(0.5, 1.0, t) * colormap_grapefruit(2.0 * (t - 0.5));
}

// https://stackoverflow.com/a/141943: scale a color, redistributing overflow so the hue is preserved
vec3 redistribute_rgb(vec3 color) {
    float r = color.x; float g = color.y; float b = color.z;
    float m = max(max(r, g), b);
    if (m <= 1.0) return color;
    float total = r + g + b;
    if (total >= 3.0) return vec3(1.0);
    float x = (3.0 - total) / (3.0 * m - total);
    float gray = 1.0 - x * m;
    return vec3(gray + x * r, gray + x * g, gray + x * b);
}

vec3 darken_color(vec3 color, float t) {
    return redistribute_rgb(t * color);
}

// color of a field value: colormap over [-range, range], isobands of frequency `freq`, zero contour of ~1.5 px
vec3 sdfColor(float phi, float range, float freq) {
    float dark_factor = 0.9;
    dark_factor *= 1.0 - 0.1 * smoothstep(-0.01, 0.01, sin(-freq * phi));
    vec3 col = darken_color(colormap_SDF(normalize_scalar(-range, range, phi)), dark_factor);
    float line = 1.0 - smoothstep(0.5, 1.5, abs(phi) / max(fwidth(phi), 1e-12));
    return mix(col, vec3(0.1), line);
}
