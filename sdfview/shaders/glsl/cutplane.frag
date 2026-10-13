// cutplane.frag — field values on a plane through the scene box: colormap, isobands, zero contour
#include "common.glsl"
#include "field.glsl"
#include "colormap.glsl"

uniform vec3 u_box_min;
uniform vec3 u_box_max;
uniform float u_colormap_range;
uniform float u_isoband_frequency;
uniform float u_opacity;

in vec3 v_world;
layout(location = 0) out vec4 frag_color;

void main() {
    vec3 slack = 1e-4 * (u_box_max - u_box_min);
    if (any(lessThan(v_world, u_box_min - slack)) || any(greaterThan(v_world, u_box_max + slack))) discard;
    float phi = fieldDistance(v_world);
    frag_color = vec4(sdfColor(phi, u_colormap_range, u_isoband_frequency), u_opacity);
}
