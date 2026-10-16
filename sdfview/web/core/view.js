// view.js — host-independent view state for the JavaScript hosts: settings (a subset of sdfview/settings.py
// ViewSettings, same meanings) and the choice of fields per model (as Renderer.models / model_field / resolve_field /
// surface_passes / field_part in sdfview/backends/moderngl/renderer.py).

export const MODEL_COLORS = { pat: [0.86, 0.70, 0.45], sup: [0.42, 0.62, 0.90], gt: [0.72, 0.72, 0.72] };
export const EXTRA_COLORS = [[0.52, 0.78, 0.50], [0.80, 0.55, 0.82], [0.92, 0.55, 0.45], [0.45, 0.80, 0.80]];

export function defaultSettings(overrides = {}) {
  return {
    surfaceField: 'auto',       // field name, or 'auto': the first model's cheapest field to trace
    surfaceMode: 'single',      // 'single' | 'overlay' (every model; the selected one opaque)
    overlayOpacity: 1.0,        // < 1: other models blended without depth writes
    modelColors: null,          // {model: [r, g, b]}
    part: -1,                   // fields with parts: -1 whole, -2 all parts, i part i
    colorParts: false,
    isovalue: 0.0,
    k: null,                    // PAT neighbors (null: the scene's)
    lambdaScale: null,          // PAT λ scale (null: the scene's)
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
  if (!primary) return [];
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
