// three/webgpu_renderer.js — three.js host, WebGPURenderer: the field written in TSL (three.js' shading language,
// JavaScript building a node graph that three.js compiles to WGSL, or to GLSL on its WebGL2 fallback backend). The
// GLSL / WGSL modules are not used: WebGPURenderer takes node materials, not ShaderMaterials (three r186), and native
// code enters only as single functions (wgslFn / glslFn), without the module-level storage bindings field_pat needs.
//
// Chromium 141 (Dawn): three's slice-by-slice upload of the 3D grid fails validation unless Dawn's lazy clear is off
// (--disable-dawn-features=lazy_clear_resource_on_first_use, as web/test/run_probe.mjs launches it); see the note there.
//
// Field types: 'grid' (gridSample / gridDistance / gridDistanceGradient: the functions of field_grid.glsl, in TSL).
// 'pat' and 'glsl' are not ported to TSL; their grid bakes are drawn instead.
//
//   1. context      new THREE.WebGPURenderer({ canvas, forceWebGL }); await renderer.init()
//   2. upload       Data3DTexture (r32float, sampled with texture3DLoad: exact trilinear, as the WGSL host)
//   3. programs     NodeMaterial.fragmentNode = surface(); QuadMesh draws it; compute for probes
//   4. per frame    uniform nodes' values, quad.render(renderer)

import * as THREE from 'three/webgpu';
import {
  Break, Discard, Fn, If, Loop, abs, clamp, dot, float, floor, instanceIndex, instancedArray, int, ivec3, length,
  max, min, mix, normalize, pow, screenUV, texture3DLoad, uniform, vec2, vec3, vec4,
} from 'three/tsl';
import { columnMajor } from '../core/camera.js';
import { boxOf, defaultSettings, diagonal, resolveField, surfacePasses } from '../core/view.js';

const SUPPORTED = ['grid'];

// three r186 passes `swizzle: 'rgba'` in every GPUTextureViewDescriptor (the current spec: a string, default 'rgba',
// the identity). Chromium <= 141 typed that member as a dictionary and rejects the string (TypeError in createView).
// Where the browser rejects it, drop the member when it is the identity: no change in meaning, the default applies.
function viewSwizzleCompat(device) {
  const probe = device.createTexture({ size: [1, 1], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  try {
    probe.createView({ swizzle: 'rgba' });
  } catch (e) {
    const createView = GPUTexture.prototype.createView;
    GPUTexture.prototype.createView = function (desc) {
      if (desc && desc.swizzle === 'rgba') {
        desc = { ...desc };
        delete desc.swizzle;
      }
      return createView.call(this, desc);
    };
    GPUTexture.prototype.createView.sdfxCompat = true;
  } finally {
    probe.destroy();
  }
}

// field_grid.glsl in TSL: samples on the grid vertices, trilinear from 8 loads, outside the box φ(clamp p) + |p - clamp p|
function gridFunctions(tex, res, gridMin, gridMax) {
  const load = (i) => texture3DLoad(tex, i, int(0)).x;
  const gridSample = Fn(([p]) => {
    const x = p.sub(gridMin).div(gridMax.sub(gridMin)).mul(res.sub(1.0));
    const i0 = ivec3(clamp(floor(x), vec3(0.0), res.sub(2.0)));
    const t = x.sub(vec3(i0));
    const c00 = mix(load(i0), load(i0.add(ivec3(1, 0, 0))), t.x);
    const c10 = mix(load(i0.add(ivec3(0, 1, 0))), load(i0.add(ivec3(1, 1, 0))), t.x);
    const c01 = mix(load(i0.add(ivec3(0, 0, 1))), load(i0.add(ivec3(1, 0, 1))), t.x);
    const c11 = mix(load(i0.add(ivec3(0, 1, 1))), load(i0.add(ivec3(1, 1, 1))), t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
  });
  const gridDistance = Fn(([p]) => {
    const pc = clamp(p, gridMin, gridMax);
    return gridSample(pc).add(length(p.sub(pc)));
  });
  const gridDistanceGradient = Fn(([p]) => {  // vec4(φ, ∇φ): central differences, h = one cell
    const h = gridMax.sub(gridMin).div(res.sub(1.0));
    const dx = vec3(h.x, 0, 0), dy = vec3(0, h.y, 0), dz = vec3(0, 0, h.z);
    const g = vec3(gridDistance(p.add(dx)).sub(gridDistance(p.sub(dx))), gridDistance(p.add(dy)).sub(gridDistance(p.sub(dy))),
                   gridDistance(p.add(dz)).sub(gridDistance(p.sub(dz)))).div(h.mul(2.0));
    return vec4(gridDistance(p), g);
  });
  return { gridSample, gridDistance, gridDistanceGradient };
}

export async function createRenderer(canvas, scene, { forceWebGL = false } = {}) {
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, forceWebGL });
  await renderer.init();
  if (!forceWebGL && !GPUTexture.prototype.createView.sdfxCompat) viewSwizzleCompat(renderer.backend.device);
  renderer.autoClear = false;
  const box = boxOf(scene);

  // uniforms shared by the surface programs (names as in the GLSL / WGSL)
  const U = {
    u_inv_view_proj: uniform(new THREE.Matrix4()), u_box_min: uniform(new THREE.Vector3(...box[0])),
    u_box_max: uniform(new THREE.Vector3(...box[1])), u_epsilon: uniform(1e-4 * diagonal(box)),
    u_step_scale: uniform(0.9), u_color_front: uniform(new THREE.Vector3()), u_color_back: uniform(new THREE.Vector3()),
    field_isovalue: uniform(0.0),
  };
  const MAX_STEPS = 128;  // a loop bound fixed at build time here (the GLSL / WGSL read u_max_steps)

  const fields = {};
  for (const f of scene.manifest.fields) {
    if (f.type !== 'grid') continue;
    const a = scene.arrays[f.arrays.values];
    const [d, h, w] = a.shape;
    const tex = new THREE.Data3DTexture(a.data, w, h, d);
    tex.format = THREE.RedFormat;
    tex.type = THREE.FloatType;
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    const fn = gridFunctions(tex, vec3(w, h, d), uniform(new THREE.Vector3(...f.bounds.min)), uniform(new THREE.Vector3(...f.bounds.max)));
    fields[f.name] = { field: f, tex, ...fn };
  }

  // surface.frag in TSL: sphere tracing with bisection on a sign change, Blinn-Phong head light
  function surfaceMaterial(F) {
    const fieldDistance = (p) => F.gridDistance(p).sub(U.field_isovalue);
    const material = new THREE.NodeMaterial();
    material.fragmentNode = Fn(() => {
      const ndc = vec2(screenUV.x.mul(2.0).sub(1.0), float(1.0).sub(screenUV.y.mul(2.0)));  // screenUV: origin top left
      const near = U.u_inv_view_proj.mul(vec4(ndc, -1.0, 1.0));
      const far = U.u_inv_view_proj.mul(vec4(ndc, 1.0, 1.0));
      const ro = near.xyz.div(near.w).toVar();
      const rd = normalize(far.xyz.div(far.w).sub(ro)).toVar();
      const inv = vec3(1.0).div(rd);
      const ta = U.u_box_min.sub(ro).mul(inv), tb = U.u_box_max.sub(ro).mul(inv);
      const tmin = min(ta, tb), tmax = max(ta, tb);
      const t0 = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
      const t1 = min(min(tmax.x, tmax.y), tmax.z);
      If(t0.greaterThanEqual(t1), () => { Discard(); });
      const t = float(t0).toVar();
      const d = float(fieldDistance(ro.add(rd.mul(t)))).toVar();
      const hit = float(0.0).toVar();
      Loop(MAX_STEPS, () => {
        If(abs(d).lessThan(U.u_epsilon), () => { hit.assign(1.0); Break(); });
        const tPrev = float(t).toVar(), dPrev = float(d).toVar();
        t.addAssign(U.u_step_scale.mul(max(abs(d), U.u_epsilon.mul(0.5))));
        If(t.greaterThan(t1), () => { Break(); });
        d.assign(fieldDistance(ro.add(rd.mul(t))));
        If(d.lessThan(0.0).notEqual(dPrev.lessThan(0.0)), () => {  // bracketed: bisection on [tPrev, t]
          const a = float(tPrev).toVar(), b = float(t).toVar(), da = float(dPrev).toVar();
          Loop(10, () => {
            const m = a.add(b).mul(0.5);
            const dm = fieldDistance(ro.add(rd.mul(m))).toVar();
            If(dm.lessThan(0.0).equal(da.lessThan(0.0)), () => { a.assign(m); da.assign(dm); }).Else(() => { b.assign(m); });
          });
          t.assign(a.add(b).mul(0.5));
          hit.assign(1.0);
          Break();
        });
      });
      If(hit.lessThan(0.5), () => { Discard(); });
      const p = ro.add(rd.mul(t));
      const n0 = normalize(F.gridDistanceGradient(p).yzw);
      const back = dot(n0, rd).greaterThan(0.0);
      const n = back.select(n0.negate(), n0);
      const base = back.select(U.u_color_back, U.u_color_front);
      const l = normalize(rd.negate().add(vec3(0.3, 1.0, 0.2).mul(0.3)));
      const hv = normalize(l.sub(rd));
      const diffuse = max(dot(n, l), 0.0);
      const specular = pow(max(dot(n, hv), 0.0), 48.0);
      return vec4(base.mul(diffuse.mul(0.8).add(0.2)).add(specular.mul(0.25)), 1.0);
    })();
    return material;
  }
  const quads = {};
  for (const [name, F] of Object.entries(fields)) quads[name] = new THREE.QuadMesh(surfaceMaterial(F));

  function frame(camera, s, viewport) {
    const cam = camera.uniforms(viewport);
    U.u_inv_view_proj.value.fromArray(columnMajor(cam.u_inv_view_proj));
    U.u_epsilon.value = s.epsilon ?? 1e-4 * diagonal(box);
    U.u_step_scale.value = s.stepScale;
    U.field_isovalue.value = s.isovalue;
    renderer.setClearColor(new THREE.Color(...s.background), 1);
    renderer.clear();
    for (const p of surfacePasses(scene, s, SUPPORTED)) {
      U.u_color_front.value.set(...p.front);
      U.u_color_back.value.set(...p.back);
      quads[p.field.name].render(renderer);
    }
  }

  function draw(camera, s, viewport) {
    renderer.setRenderTarget(null);
    renderer.setSize(canvas.width, canvas.height, false);
    renderer.setViewport(...viewport);
    frame(camera, s, viewport);
  }

  async function render(camera, s, [w, h]) {
    const target = new THREE.RenderTarget(w, h);
    renderer.setRenderTarget(target);
    frame(camera, s, [0, 0, w, h]);
    const raw = new Uint8Array(await renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h));
    renderer.setRenderTarget(null);
    target.dispose();
    if (renderer.backend.isWebGLBackend) return raw;  // glReadPixels: packed rows, bottom-up
    // WebGPU backend: rows of a 256-byte multiple (the last row unpadded), top row first
    const row = 4 * w, stride = h > 1 ? (raw.length - row) / (h - 1) : row;
    const out = new Uint8Array(row * h);  // contract: rows bottom-up, as glReadPixels
    for (let y = 0; y < h; y++) out.set(raw.subarray(y * stride, y * stride + row), (h - 1 - y) * row);
    return out;
  }

  // probe: a TSL compute shader, one invocation per query, results in a storage buffer read back
  async function probe(queries, fieldName = 'auto', s = defaultSettings()) {
    const f = resolveField(scene, fieldName, false, SUPPORTED);
    const n = queries.length / 3;
    const q4 = new Float32Array(Math.max(n, 1) * 4);
    for (let i = 0; i < n; i++) q4.set(queries.subarray(3 * i, 3 * i + 3), 4 * i);
    const qbuf = instancedArray(q4, 'vec4');
    const out = instancedArray(Math.max(n, 1), 'vec4');
    U.field_isovalue.value = (f.params?.isovalue ?? 0) + s.isovalue;
    const F = fields[f.name];
    const kernel = Fn(() => {
      const r = F.gridDistanceGradient(qbuf.element(instanceIndex).xyz);
      out.element(instanceIndex).assign(vec4(r.x.sub(U.field_isovalue), r.yzw));
    })().compute(Math.max(n, 1));
    await renderer.computeAsync(kernel);
    const data = new Float32Array(await renderer.getArrayBufferAsync(out.value));
    return data.subarray(0, 4 * n);
  }

  function release() {
    for (const F of Object.values(fields)) F.tex.dispose();
    renderer.dispose();
  }

  return { draw, render, probe, release, supports: (fl) => SUPPORTED.includes(fl.type), name: forceWebGL ? 'three-tsl-webgl' : 'three-webgpu', three: renderer };
}
