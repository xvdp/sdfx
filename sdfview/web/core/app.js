// app.js — the interactive page shared by the raw and three.js hosts: loads a scene (?scene=URL, a file chosen, or a
// file dropped), owns the orbit camera and the settings, and calls the host's renderer on change. Hosts differ only in
// their renderer module: createRenderer(canvas, scene) -> {draw(camera, settings, viewport), probe, release}.
// Keys and mouse: HELP below (the moderngl window's, sdfview/backends/moderngl/window.py); H shows / hides them.
//
// A new scene gets a new canvas: a canvas keeps the context type it was first given, and three.js' WebGL backend
// loses its context on dispose().

import { loadScene, parseScene } from '../scene.js';
import { OrbitCamera } from './camera.js';
import { boxOf, defaultSettings, diagonal, pointClouds, resolveField } from './view.js';

export const HELP = [
  ['mouse', 'left drag: orbit · right / middle drag: pan · wheel: zoom · drop a .sdfx file to open it'],
  ['S C P', 'surface · cut plane · points'],
  ['X Y Z  ↑ ↓', 'cut plane axis · move it (shift: faster)'],
  ['K  B', 'clip the surface at the plane · cut plane opacity'],
  ['G  F', 'next surface field · next cut plane field'],
  ['V  T', 'overlay all models · overlay transparency'],
  ['[ ]  U', 'parts of an assembly (whole, all parts, part 0, 1, ...) · color by part'],
  ['I O  N M', 'isovalue ∓ (shift: ×5) · PAT neighbors k ∓ (shift: ×8)'],
  ['R  H', 'reset view · this help'],
];

function helpPanel() {
  const panel = document.createElement('div');
  panel.style.cssText = 'position:fixed;left:12px;top:12px;padding:8px 12px;border-radius:6px;font:12px/1.5 ui-monospace,monospace;'
    + 'background:rgba(20,20,20,0.82);color:#eee;white-space:pre;pointer-events:none;z-index:10';
  panel.textContent = HELP.map(([k, v]) => `${k.padEnd(11)} ${v}`).join('\n');
  document.body.appendChild(panel);
  return panel;
}

export async function runViewer(createRenderer, { canvas, status, title }) {
  const settings = defaultSettings();
  let scene = null, renderer = null, camera = null, surfaceFields = ['auto'], cutFields = ['auto'];
  let pending = false;
  const help = helpPanel();

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
      const s = settings;
      const surf = s.showSurface ? `${s.surfaceField}${s.surfaceMode === 'overlay' ? ' + overlay' : ''}` : 'off';
      const cut = s.showCutplane && resolveField(scene, s.cutField, false) ? `${s.cutField} ${'xyz'[s.cutAxis]} ${s.cutOffset.toFixed(2)}` : 'off';
      const parts = s.part === -1 ? '' : s.part === -2 ? ' · all parts' : ` · part ${s.part}`;
      report(`${title} · surface ${surf} · cut ${cut}${s.showPoints ? ' · points' : ''}${parts} · `
             + `${(performance.now() - t0).toFixed(1)} ms (submit) · H: keys`);
    });
  };

  // input on the current canvas (re-attached when a scene replaces the canvas)
  let dragging = null, last = null;
  function listen(c) {
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('dragover', (e) => e.preventDefault());
    c.addEventListener('drop', (e) => { e.preventDefault(); fromFile(e.dataTransfer.files[0]); });
    c.addEventListener('pointerdown', (e) => { dragging = e.button; last = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointerup', () => { dragging = null; });
    c.addEventListener('pointermove', (e) => {
      if (dragging === null || !camera) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      if (dragging === 0) camera.orbit(dx, dy); else camera.pan(dx, dy, c.clientHeight);
      last = [e.clientX, e.clientY];
      redraw();
    });
    c.addEventListener('wheel', (e) => { e.preventDefault(); camera?.zoom(-Math.sign(e.deltaY)); redraw(); }, { passive: false });
  }

  async function open(newScene) {
    if (renderer) {
      renderer.release();
      const fresh = canvas.cloneNode(false);
      canvas.replaceWith(fresh);
      canvas = fresh;
      listen(canvas);
    }
    scene = newScene;
    renderer = await createRenderer(canvas, scene);
    camera = OrbitCamera.fit(...boxOf(scene));
    const names = scene.manifest.fields.filter((f) => renderer.supports(f)).map((f) => f.name);
    surfaceFields = ['auto', ...names];
    cutFields = ['auto', ...names];
    settings.surfaceField = settings.cutField = 'auto';
    settings.showPoints = settings.showPoints && pointClouds(scene).length > 0;
    redraw();
  }
  const fromFile = async (file) => open(parseScene(await file.arrayBuffer()));
  listen(canvas);
  document.querySelector('input[type=file]')?.addEventListener('change', (e) => fromFile(e.target.files[0]));

  const url = new URLSearchParams(location.search).get('scene');
  if (url) await open(await loadScene(url)); else report(`${title} · choose or drop a .sdfx scene · H: keys`);

  window.addEventListener('resize', redraw);
  window.addEventListener('keydown', (e) => {
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === 'h') { help.hidden = !help.hidden; return; }
    if (!scene) return;
    const s = settings;
    const next = (list, value) => list[(list.indexOf(value) + 1) % list.length];
    if (k === 's') s.showSurface = !s.showSurface;
    else if (k === 'c') s.showCutplane = !s.showCutplane;
    else if (k === 'p') s.showPoints = !s.showPoints;
    else if (k === 'x' || k === 'y' || k === 'z') { s.cutAxis = 'xyz'.indexOf(k); s.cutNormal = null; }
    else if (k === 'ArrowUp') s.cutOffset = Math.min(1, s.cutOffset + (e.shiftKey ? 0.1 : 0.02));
    else if (k === 'ArrowDown') s.cutOffset = Math.max(-1, s.cutOffset - (e.shiftKey ? 0.1 : 0.02));
    else if (k === 'k') s.clipSurface = !s.clipSurface;
    else if (k === 'b') s.cutOpacity = s.cutOpacity === 1 ? 0.5 : 1;
    else if (k === 'g') s.surfaceField = next(surfaceFields, s.surfaceField);
    else if (k === 'f') s.cutField = next(cutFields, s.cutField);
    else if (k === 'v') s.surfaceMode = s.surfaceMode === 'overlay' ? 'single' : 'overlay';
    else if (k === 't') s.overlayOpacity = s.overlayOpacity >= 1 ? 0.45 : 1.0;
    else if (k === 'u') s.colorParts = !s.colorParts;
    else if (k === '[' || k === ']') {
      const n = Math.max(0, ...scene.manifest.fields.map((f) => f.glsl?.n_parts ?? 0));
      const order = [-1, -2, ...Array.from({ length: n }, (_, i) => i)];
      const i = Math.max(0, order.indexOf(s.part));
      s.part = order[(i + (k === ']' ? 1 : -1) + order.length) % order.length];
    } else if (k === 'i' || k === 'o') s.isovalue += (k === 'i' ? -1 : 1) * 0.005 * diagonal(boxOf(scene)) * (e.shiftKey ? 5 : 1);
    else if (k === 'n' || k === 'm') {
      const pat = scene.manifest.fields.find((f) => f.type === 'pat');
      const k0 = s.k ?? pat?.params?.k ?? 32;
      s.k = Math.max(1, k0 + (k === 'n' ? -1 : 1) * (e.shiftKey ? 8 : 1));
    } else if (k === 'r') camera = OrbitCamera.fit(...boxOf(scene));
    else return;
    e.preventDefault();
    redraw();
  });
}
