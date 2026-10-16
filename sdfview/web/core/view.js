// view.js — host-independent view state for the JavaScript hosts: settings (a subset of sdfview/settings.py
// ViewSettings, same meanings) and the choice of fields per model (as Renderer.models / model_field / resolve_field /
// surface_passes / field_part in sdfview/backends/moderngl/renderer.py).

export const MODEL_COLORS = { pat: [0.86, 0.70, 0.45], sup: [0.42, 0.62, 0.90], gt: [0.72, 0.72, 0.72] };
export const EXTRA_COLORS = [[0.52, 0.78, 0.50], [0.80, 0.55, 0.82], [0.92, 0.55, 0.45], [0.45, 0.80, 0.80]];

export function defaultSettings(overrides = {}) {
  return {
    showSurface: true,
    showCutplane: true,
    showPoints: false,
    surfaceField: 'auto',       // field name, or 'auto': the first model's cheapest field to trace
    cutField: 'auto',           // field name, or 'auto': the first model's exact field
    surfaceMode: 'single',      // 'single' | 'overlay' (every model; the selected one opaque)
    overlayOpacity: 1.0,        // < 1: other models blended without depth writes
    modelColors: null,          // {model: [r, g, b]}
    part: -1,                   // fields with parts: -1 whole, -2 all parts, i part i
    colorParts: false,
    isovalue: 0.0,
    k: null,                    // PAT neighbors (null: the scene's)
    lambdaScale: null,          // PAT λ scale (null: the scene's)
    // cut plane n·x = c through the box: normal = axis unless cutNormal; offset in [-1, 1] across the box
    cutAxis: 2,
    cutNormal: null,
    cutOffset: 0.0,
    clipSurface: false,         // hide the surface on the side n·x > c
    colormapRange: null,        // null: ½ of the box diagonal
    isobandFrequency: 60.0,
    cutOpacity: 1.0,
    pointRadius: null,          // null: the point cloud's radius
    pointColor: [0.15, 0.35, 0.75],
    maxSteps: 128,
    epsilon: null,              // null: 1e-4 of the box diagonal
    stepScale: 0.9,
    background: [1, 1, 1],
    colorFront: [0.85, 0.78, 0.62],
    colorBack: [0.55, 0.62, 0.85],
    ...overrides,
  };
}

export const modelOf = (f) => f.model ?? f.name;

export function models(scene) {
  const out = [];
  for (const f of scene.manifest.fields) if (!out.includes(modelOf(f))) out.push(modelOf(f));
  return out;
}

// surfaces: cheapest to trace first (generated GLSL, grid, PAT); cut plane / probes: exact first
const SURFACE_ORDER = { glsl: 0, grid: 1 };
const CUT_ORDER = { grid: 1 };

export function preferred(fields, forSurface, supported = null) {
  const usable = supported ? fields.filter((f) => supported.includes(f.type)) : fields;
  const rank = (f) => (forSurface ? SURFACE_ORDER[f.type] ?? 2 : CUT_ORDER[f.type] ?? 0);
  return usable.reduce((best, f) => (best === null || rank(f) < rank(best) ? f : best), null);
}

export function modelField(scene, model, forSurface = true, supported = null) {
  return preferred(scene.manifest.fields.filter((f) => modelOf(f) === model), forSurface, supported);
}

export function resolveField(scene, which, forSurface, supported = null) {
  if (which !== 'auto') {
    const f = scene.manifest.fields.find((g) => g.name === which || g.type === which);
    if (!f) throw new Error(`no field ${which} in the scene`);
    return f;
  }
  for (const m of models(scene)) {  // the first model with a field this host supports
    const f = modelField(scene, m, forSurface, supported);
    if (f) return f;
  }
  return null;
}

export function modelColor(scene, model, s) {
  if (s.modelColors && model in s.modelColors) return s.modelColors[model];
  if (model in MODEL_COLORS) return MODEL_COLORS[model];
  const others = models(scene).filter((m) => !(m in MODEL_COLORS));
  return EXTRA_COLORS[Math.max(0, others.indexOf(model)) % EXTRA_COLORS.length];
}

// [{field, front, back, opacity}] of the surfaces to draw
export function surfacePasses(scene, s, supported = null) {
  const primary = resolveField(scene, s.surfaceField, true, supported);
  if (!s.showSurface || !primary) return [];
  const ms = models(scene);
  const fields = [[primary, 1.0]];
  if (s.surfaceMode === 'overlay') {
    for (const m of ms) {
      if (m === modelOf(primary)) continue;
      const f = modelField(scene, m, true, supported);
      if (f) fields.push([f, s.overlayOpacity]);
    }
  }
  return fields.map(([field, opacity]) => {
    const front = ms.length > 1 ? modelColor(scene, modelOf(field), s) : s.colorFront;
    const back = ms.length > 1 ? front.map((x) => 0.55 * x) : s.colorBack;
    return { field, front, back, opacity };
  });
}

// cut plane n·x = c through the box (as cut_plane in sdfview/settings.py): unit normal n, offset c, rectangle
// origin and half-axes u, v (the rectangle covers the box's projection; fragments outside the box are discarded)
export function cutPlane(s, [lo, hi]) {
  const center = [0, 1, 2].map((i) => 0.5 * (lo[i] + hi[i]));
  const half = [0, 1, 2].map((i) => 0.5 * (hi[i] - lo[i]));
  let n = s.cutNormal ? [...s.cutNormal] : [0, 1, 2].map((i) => (i === s.cutAxis ? 1 : 0));
  const len = Math.hypot(...n);
  n = n.map((x) => x / len);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const reach = Math.abs(n[0]) * half[0] + Math.abs(n[1]) * half[1] + Math.abs(n[2]) * half[2];
  const c = dot(n, center) + s.cutOffset * reach;
  const origin = center.map((x, i) => x + (c - dot(n, center)) * n[i]);
  const k = n.map(Math.abs).indexOf(Math.min(...n.map(Math.abs)));  // the axis least aligned with n
  let u = cross(n, [0, 1, 2].map((i) => (i === k ? 1 : 0)));
  const ul = Math.hypot(...u);
  u = u.map((x) => x / ul);
  const v = cross(n, u);
  const r = Math.hypot(...half);
  return { n, c, origin, u: u.map((x) => x * r), v: v.map((x) => x * r) };
}

// the uniforms of the cut plane and of the surface's clip half-space (GLSL / WGSL names)
export function cutUniforms(s, box) {
  const { n, c, origin, u, v } = cutPlane(s, box);
  return {
    u_plane_origin: origin, u_plane_u: u, u_plane_v: v, u_clip: s.clipSurface && s.showCutplane,
    u_clip_plane: [...n, c], u_colormap_range: s.colormapRange ?? 0.5 * diagonal(box),
    u_isoband_frequency: s.isobandFrequency, u_opacity: s.cutOpacity,
  };
}

export const pointClouds = (scene) => scene.manifest.pointclouds ?? [];

export function fieldPart(field, s) {
  const n = field.glsl?.n_parts ?? 0;
  return s.part >= -2 && s.part < n ? s.part : -1;
}

export function boxOf(scene) {
  const m = scene.manifest;
  if (m.bounds) return [m.bounds.min, m.bounds.max];
  const boxes = m.fields.filter((f) => f.bounds).map((f) => [f.bounds.min, f.bounds.max]);
  if (!boxes.length) return [[-1, -1, -1], [1, 1, 1]];
  return [0, 1].map((j) => [0, 1, 2].map((i) => (j ? Math.max : Math.min)(...boxes.map((b) => b[j][i]))));
}

export const diagonal = ([lo, hi]) => Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
