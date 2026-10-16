// field_torus.glsl — one torus as a field (scene field type "torus": the tori inspector shows the torus of one point).
//   g(q) = s (|(v - R, h)| - |r|),  v = |(q - c) × u| (distance to the axis),  h = (q - c)·u,  s = sign(r)
//   ∇g = s (ŵ_x e + ŵ_y u),  (ŵ_x, ŵ_y) = (v - R, h) / |(v - R, h)|,  e = ((q - c) - h u) / v  (radial unit)
// (the torus SDF of field_pat.glsl's accumulateTorusPoint, with its gradient written through e instead of the 2x3
// Jacobian G).

uniform vec3 torus_center;
uniform vec3 torus_axis;   // unit
uniform float torus_R;
uniform float torus_r;     // signed: s |r|

float torusDistanceGradient(in vec3 q, out vec3 gradient) {
    vec3 w = q - torus_center;
    float h = dot(w, torus_axis);
    vec3 radial = w - h * torus_axis;
    float v = length(radial);
    vec3 e = v > 1e-12 ? radial / v : vec3(0.0);
    vec2 d = vec2(v - torus_R, h);
    float len = max(length(d), 1e-12);
    float s = torus_r >= 0.0 ? 1.0 : -1.0;
    gradient = s * (d.x / len * e + d.y / len * torus_axis);
    return s * (len - abs(torus_r));
}

float torusDistance(in vec3 q) {
    vec3 gradient;
    return torusDistanceGradient(q, gradient);
}
