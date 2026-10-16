// app.js — the interactive page shared by the raw and three.js hosts: loads a scene (?scene=URL, a file chosen, or a
// file dropped), owns the orbit camera and the settings, and calls the host's renderer on change. Hosts differ only in
// their renderer module: createRenderer(canvas, scene) -> {draw(camera, settings, viewport), probe, release}.
//
// Mouse: left drag orbit, wheel zoom. Keys: G next surface field, V overlay models, T translucency, [ ] parts,
// U color parts, I / O isovalue, R reset view.

import { loadScene, parseScene } from '../scene.js';
import { OrbitCamera } from './camera.js';
import { boxOf, defaultSettings, diagonal } from './view.js';

export async function runViewer(createRenderer, { canvas, status, title }) {
  const settings = defaultSettings();
  let scene = null, renderer = null, camera = null, fields = ['auto'];
  let pending = false;

  const report = (text) => { if (status) status.textContent = text; };
  const redraw = () => {
    if (!renderer || pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      const t0 = performance.now();
      renderer.draw(camera, settings, [0, 0, w, h]);
      const parts = settings.part === -1 ? '' : settings.part === -2 ? ' · all parts' : ` · part ${settings.part}`;
      report(`${title} · surface ${settings.surfaceField}${settings.surfaceMode === 'overlay' ? ' + overlay' : ''}` +
             `${parts} · ${(performance.now() - t0).toFixed(1)} ms (submit)`);
    });
  };

  async function open(newScene) {
    if (renderer) renderer.release();
    scene = newScene;
    renderer = await createRenderer(canvas, scene);
    const [lo, hi] = boxOf(scene);
    camera = OrbitCamera.fit(lo, hi);
    fields = ['auto', ...scene.manifest.fields.filter((f) => renderer.supports(f)).map((f) => f.name)];
    settings.surfaceField = 'auto';
    redraw();
  }

  const url = new URLSearchParams(location.search).get('scene');
  if (url) await open(await loadScene(url)); else report(`${title} · choose or drop a .sdfx scene`);

  const fromFile = async (file) => open(parseScene(await file.arrayBuffer()));
  document.querySelector('input[type=file]')?.addEventListener('change', (e) => fromFile(e.target.files[0]));
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', (e) => { e.preventDefault(); fromFile(e.dataTransfer.files[0]); });

  let dragging = false, last = null;
  canvas.addEventListener('pointerdown', (e) => { dragging = true; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging || !camera) return;
    camera.orbit(e.clientX - last[0], e.clientY - last[1]);
    last = [e.clientX, e.clientY];
    redraw();
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); camera?.zoom(-Math.sign(e.deltaY)); redraw(); }, { passive: false });
  window.addEventListener('resize', redraw);
  window.addEventListener('keydown', (e) => {
    if (!scene) return;
    const k = e.key.toLowerCase();
    if (k === 'g') settings.surfaceField = fields[(fields.indexOf(settings.surfaceField) + 1) % fields.length];
    else if (k === 'v') settings.surfaceMode = settings.surfaceMode === 'overlay' ? 'single' : 'overlay';
    else if (k === 't') settings.overlayOpacity = settings.overlayOpacity >= 1 ? 0.45 : 1.0;
    else if (k === 'u') settings.colorParts = !settings.colorParts;
    else if (k === '[' || k === ']') {
      const n = Math.max(0, ...scene.manifest.fields.map((f) => f.glsl?.n_parts ?? 0));
      const order = [-1, -2, ...Array.from({ length: n }, (_, i) => i)];
      const i = Math.max(0, order.indexOf(settings.part));
      settings.part = order[(i + (k === ']' ? 1 : -1) + order.length) % order.length];
    } else if (k === 'i' || k === 'o') settings.isovalue += (k === 'i' ? -1 : 1) * 0.005 * diagonal(boxOf(scene));
    else if (k === 'r') camera = OrbitCamera.fit(...boxOf(scene));
    else return;
    redraw();
  });
}
