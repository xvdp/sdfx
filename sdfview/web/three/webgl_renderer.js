// three/webgl_renderer.js — three.js host, WebGLRenderer: the GLSL modules of sdfview/shaders/glsl (the same programs
// as the raw WebGL2 host) in RawShaderMaterials, the scene arrays in DataTextures. The difference from the raw host is
// who owns the GL calls: three.js creates the textures, programs and draw state from objects.
//
//   1. context      new THREE.WebGLRenderer({ canvas })
//   2. upload       arrayTexture: DataTexture (RGBA32F / RGBA32I / R32I, 4096 texels per row, nearest);
//                   gridTexture: Data3DTexture (R32F, linear)
//   3. programs     RawShaderMaterial({ glslVersion: GLSL3, vertexShader, fragmentShader, uniforms }) with
//                   glslSource(..., 'three') (three.js adds '#version 300 es')
//   4. per frame    uniform values, renderer.render(sceneOfQuads, camera)
//   render / probe  WebGLRenderTarget + readRenderTargetPixels
//
// sdfMesh(field, ...): a THREE.Mesh tracing the field, to add to any three.js scene: its uniforms follow the three.js
// camera (onBeforeRender) and it writes depth, so it composes with the scene's meshes and points (web/three/compose.html).

import * as THREE from 'three';
import { columnMajor, invert, multiply } from '../core/camera.js';
import { fetchReader, fieldDefines, glslSource } from '../core/shaders.js';
import { boxOf, defaultSettings, diagonal, fieldPart, resolveField, surfacePasses } from '../core/view.js';

export const TEXTURE_WIDTH = 4096;
const SUPPORTED = ['pat', 'grid', 'glsl'];
const SHADERS = new URL('../../shaders/', import.meta.url);

export function arrayTexture(array, width = TEXTURE_WIDTH) {
  const { data, shape, dtype } = array;
  const cols = shape[shape.length - 1];
  const comps = cols % 4 === 0 ? 4 : cols;
  const count = Math.max(data.length / comps, 1);
  const w = Math.min(width, count), h = Math.ceil(count / w);
  const padded = new data.constructor(w * h * comps);
  padded.set(data);
  const int = dtype === 'int32';
  const format = comps === 4 ? (int ? THREE.RGBAIntegerFormat : THREE.RGBAFormat) : int ? THREE.RedIntegerFormat : THREE.RedFormat;
  const tex = new THREE.DataTexture(padded, w, h, format, int ? THREE.IntType : THREE.FloatType);
  tex.internalFormat = `${comps === 4 ? 'RGBA' : 'R'}32${int ? 'I' : 'F'}`;
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export function gridTexture(array) {
  const [d, h, w] = array.shape;
  const tex = new THREE.Data3DTexture(array.data, w, h, d);
  tex.format = THREE.RedFormat;
  tex.type = THREE.FloatType;
  tex.internalFormat = 'R32F';
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// a triangle covering the viewport (the vertex shader ignores the positions: gl_VertexID)
function fullscreenGeometry() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
  return g;
}

// uniforms of a field (names as in the GLSL modules), with the field's textures
function fieldUniforms(field, textures, s) {
  const p = field.params || {};
  const u = { field_isovalue: { value: (p.isovalue ?? 0) + s.isovalue } };
  if (field.type === 'pat') {
    for (const key of ['points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims']) u[`pat_${key}`] = { value: textures[`${field.name}/${key}`] };
    u.pat_k = { value: Math.min(s.k ?? p.k ?? 32, Math.max(32, p.k ?? 32)) };
    u.pat_lambda_scale = { value: s.lambdaScale ?? p.lambda_scale ?? 1.0 };
    u.pat_max_exp_arg = { value: p.max_exp_arg ?? 64.0 };
  } else if (field.type === 'grid') {
    u.grid_values = { value: textures[`${field.name}/values`] };
    u.grid_min = { value: field.bounds.min };
    u.grid_max = { value: field.bounds.max };
  } else if (field.type === 'glsl') {
    u.field_part = { value: fieldPart(field, s) };
    u.field_gradient_step = { value: 5e-4 * diagonal([field.bounds.min, field.bounds.max]) };
  }
  return u;
}

async function fieldMaterial(read, frag, field, uniforms) {
  const defines = fieldDefines(field);
  const prelude = field.type === 'glsl' ? field.glsl.code : null;
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: await glslSource(read, 'fullscreen.vert', 'three', defines),
    fragmentShader: await glslSource(read, frag, 'three', defines, prelude),
    uniforms,
  });
}

export function uploadTextures(scene) {
  const textures = {};
  for (const f of scene.manifest.fields) {
    const a = (key) => scene.arrays[f.arrays[key]];
    if (f.type === 'pat') for (const key of ['points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims']) textures[`${f.name}/${key}`] = arrayTexture(a(key));
    else if (f.type === 'grid') textures[`${f.name}/values`] = gridTexture(a('values'));
  }
  return textures;
}

// A mesh tracing `field` inside the box, for any three.js scene; settings as web/core/view.js defaultSettings().
export async function sdfMesh(scene, field, textures, settings = defaultSettings(), { read = fetchReader(SHADERS), color = null, opacity = 1 } = {}) {
  const box = boxOf(scene);
  const uniforms = {
    ...fieldUniforms(field, textures, settings),
    u_view_proj: { value: new Float32Array(16) }, u_inv_view_proj: { value: new Float32Array(16) },
    u_viewport: { value: [0, 0, 1, 1] }, u_box_min: { value: box[0] }, u_box_max: { value: box[1] },
    u_max_steps: { value: settings.maxSteps }, u_epsilon: { value: settings.epsilon ?? 1e-4 * diagonal(box) },
    u_step_scale: { value: settings.stepScale }, u_clip: { value: false }, u_clip_plane: { value: [0, 0, 1, 0] },
    u_color_front: { value: color ?? settings.colorFront }, u_color_back: { value: (color ?? settings.colorBack).map((x) => (color ? 0.55 * x : x)) },
    u_opacity: { value: opacity }, u_color_parts: { value: settings.colorParts },
  };
  const material = await fieldMaterial(read, 'surface.frag', field, uniforms);
  material.transparent = opacity < 1;
  material.depthWrite = opacity >= 1;
  const mesh = new THREE.Mesh(fullscreenGeometry(), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = opacity < 1 ? 1 : 0;
  // follow the camera three.js renders with (its matrices have the OpenGL conventions of web/core/camera.js)
  const vp = new THREE.Matrix4(), viewport = new THREE.Vector4();
  mesh.onBeforeRender = (renderer, _scene, camera) => {
    vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    uniforms.u_view_proj.value.set(vp.elements);  // three.js stores column-major, as GLSL expects
    uniforms.u_inv_view_proj.value.set(vp.clone().invert().elements);
    renderer.getCurrentViewport(viewport);
    uniforms.u_viewport.value = [viewport.x, viewport.y, viewport.z, viewport.w];
  };
  return mesh;
}

// our OrbitCamera (web/core/camera.js) as a three.js camera with the same matrices
function threeCamera(camera, viewport) {
  const cam = new THREE.PerspectiveCamera();
  const view = camera.viewMatrix(), proj = camera.projMatrix(viewport[2] / Math.max(viewport[3], 1));
  cam.matrixWorldInverse.fromArray(columnMajor(view));
  cam.matrixWorld.fromArray(columnMajor(invert(view)));
  cam.projectionMatrix.fromArray(columnMajor(proj));
  cam.projectionMatrixInverse.fromArray(columnMajor(invert(proj)));
  cam.matrixAutoUpdate = false;
  cam.matrixWorldAutoUpdate = false;
  return cam;
}

export async function createRenderer(canvas, scene, { read = fetchReader(SHADERS) } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
  renderer.autoClear = false;
  const textures = uploadTextures(scene);
  const meshes = new Map();  // `${field}|${opacity < 1}` -> mesh
  for (const f of scene.manifest.fields) {
    if (!SUPPORTED.includes(f.type)) continue;
    for (const opacity of [1, 0.5]) meshes.set(`${f.name}|${opacity < 1}`, await sdfMesh(scene, f, textures, defaultSettings(), { read, opacity }));
  }

  function frame(camera, s, viewport) {
    const group = new THREE.Scene();
    for (const p of surfacePasses(scene, s, SUPPORTED)) {
      const mesh = meshes.get(`${p.field.name}|${p.opacity < 1}`);
      const u = mesh.material.uniforms;
      Object.assign(u, Object.fromEntries(Object.entries(fieldUniforms(p.field, textures, s)).map(([k, v]) => [k, u[k] ? Object.assign(u[k], v) : v])));
      u.u_color_front.value = p.front;
      u.u_color_back.value = p.back;
      u.u_opacity.value = p.opacity;
      u.u_color_parts.value = s.colorParts;
      u.u_max_steps.value = s.maxSteps;
      u.u_step_scale.value = s.stepScale;
      if (s.epsilon != null) u.u_epsilon.value = s.epsilon;
      group.add(mesh);
    }
    renderer.setClearColor(new THREE.Color(...s.background), 1);
    renderer.setViewport(...viewport);
    renderer.clear();
    renderer.render(group, threeCamera(camera, viewport));
  }

  function draw(camera, s, viewport) {
    renderer.setRenderTarget(null);
    renderer.setSize(canvas.width, canvas.height, false);
    frame(camera, s, viewport);
  }

  async function render(camera, s, [w, h]) {
    const target = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true });
    renderer.setRenderTarget(target);
    frame(camera, s, [0, 0, w, h]);
    const out = new Uint8Array(4 * w * h);
    renderer.readRenderTargetPixels(target, 0, 0, w, h, out);
    renderer.setRenderTarget(null);
    target.dispose();
    return out;
  }

  async function probe(queries, fieldName = 'auto', s = defaultSettings()) {
    const field = resolveField(scene, fieldName, false, SUPPORTED);
    const n = queries.length / 3;
    const q4 = new Float32Array(Math.max(n, 1) * 4);
    for (let i = 0; i < n; i++) q4.set(queries.subarray(3 * i, 3 * i + 3), 4 * i);
    const qtex = arrayTexture({ data: q4, shape: [Math.max(n, 1), 4], dtype: 'float32' });
    const w = qtex.image.width, h = qtex.image.height;
    const material = await fieldMaterial(read, 'probe.frag', field, {
      ...fieldUniforms(field, textures, s), u_queries: { value: qtex }, u_count: { value: n },
    });
    material.depthTest = false;
    const mesh = new THREE.Mesh(fullscreenGeometry(), material);
    mesh.frustumCulled = false;
    const group = new THREE.Scene();
    group.add(mesh);
    const target = new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, depthBuffer: false });
    renderer.setRenderTarget(target);
    renderer.setViewport(0, 0, w, h);
    renderer.render(group, new THREE.OrthographicCamera());
    const out = new Float32Array(4 * w * h);
    renderer.readRenderTargetPixels(target, 0, 0, w, h, out);
    renderer.setRenderTarget(null);
    for (const o of [target, material, qtex]) o.dispose();
    return out.subarray(0, 4 * n);
  }

  function release() {
    for (const t of Object.values(textures)) t.dispose();
    for (const m of meshes.values()) { m.material.dispose(); m.geometry.dispose(); }
    renderer.dispose();
  }

  return { draw, render, probe, release, supports: (f) => SUPPORTED.includes(f.type), name: 'three-webgl', three: renderer };
}
