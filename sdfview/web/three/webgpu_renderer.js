// three/webgpu_renderer.js — three.js host, WebGPURenderer: the field written in TSL (three.js' shading language,
// JavaScript building a node graph that three.js compiles to WGSL, or to GLSL on its WebGL2 fallback backend). The
// GLSL / WGSL modules are not used: WebGPURenderer takes node materials, not ShaderMaterials (three r186); native code
// enters as single functions (wgslFn / glslFn).
//
// Field types: 'grid' (gridSample / gridDistance / gridDistanceGradient: the functions of field_grid.glsl, in TSL).
// 'pat' and 'glsl' are not ported to TSL; their grid bakes are drawn instead.
//
// Chromium 141 (Dawn): three's slice-by-slice upload of the 3D grid fails validation unless Dawn's lazy clear is off
// (--disable-dawn-features=lazy_clear_resource_on_first_use, as web/test/run_probe.mjs launches it); see the note there.
//
//   1. context      new THREE.WebGPURenderer({ canvas, forceWebGL }); await renderer.init()
//   2. upload       Data3DTexture (r32float, sampled with texture3DLoad: exact trilinear, as the WGSL host);
//                   point clouds as instanced attributes
//   3. programs     NodeMaterials: surface (fragmentNode, writes depth) on a QuadMesh per field and opacity; cut plane
//                   and points (vertexNode: our matrices, not three.js' camera); compute for probes
//   4. per frame    uniform nodes' values; opaque surfaces, cut plane, points, translucent surfaces (one render call
//                   each, no clear in between: they share the depth buffer)

import * as THREE from 'three/webgpu';
import {
  Break, Discard, Fn, If, Loop, abs, clamp, depth, dot, float, floor, fwidth, instanceIndex, instancedArray,
  instancedBufferAttribute, int, ivec3, length, max, min, mix, normalize, positionGeometry, pow, screenUV, select, sin,
  smoothstep, step, texture3DLoad, uniform, varying, vec2, vec3, vec4,
} from 'three/tsl';
import { columnMajor } from '../core/camera.js';
import { boxOf, cutUniforms, defaultSettings, diagonal, pointClouds, resolveField, surfacePasses } from '../core/view.js';

const SUPPORTED = ['grid'];

// three r186 sets `swizzle: 'rgba'` in every GPUTextureViewDescriptor (the current spec: a string, default 'rgba',
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

// colormap.glsl in TSL (formulas there): colormap over [-range, range], isobands, zero contour ~1.5 px wide
const segment = (e0, e1, x) => step(e0, x).mul(float(1.0).sub(step(e1, x)));
const colormapPasadenaBlue = (t) => mix(vec3(1.0), vec3(0.0627451, 0.517647, 0.94902), pow(t, 7.0));
const colormapGrapefruit = (t) => mix(vec3(1.0), vec3(0.992157, 0.431373, 0.337255), t);
const colormapSDF = (t) => colormapPasadenaBlue(t.mul(2.0)).mul(segment(0.0, 0.5, t))
  .add(colormapGrapefruit(t.sub(0.5).mul(2.0)).mul(segment(0.5, 1.0, t)));
const redistributeRgb = (c) => {
  const m = max(max(c.x, c.y), c.z);
  const total = c.x.add(c.y).add(c.z);
  const x = float(3.0).sub(total).div(m.mul(3.0).sub(total));
  const spread = vec3(float(1.0).sub(x.mul(m))).add(c.mul(x));
  return select(m.lessThanEqual(1.0), c, select(total.greaterThanEqual(3.0), vec3(1.0), spread));
};
// fw = fwidth(φ), taken by the caller before any discard (derivatives need uniform control flow)
const sdfColor = (phi, fw, range, freq) => {
  const dark = float(0.9).mul(float(1.0).sub(smoothstep(-0.01, 0.01, sin(phi.mul(freq).negate())).mul(0.1)));
  const col = redistributeRgb(colormapSDF(phi.add(range).div(range.mul(2.0))).mul(dark));
  const line = float(1.0).sub(smoothstep(0.5, 1.5, abs(phi).div(max(fw, 1e-12))));
  return mix(col, vec3(0.1), line);
};

export async function createRenderer(canvas, scene, { forceWebGL = false } = {}) {
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, forceWebGL });
  await renderer.init();
  if (!forceWebGL && !GPUTexture.prototype.createView.sdfxCompat) viewSwizzleCompat(renderer.backend.device);
  renderer.autoClear = false;
  const webgpuClip = renderer.backend.isWebGPUBackend === true;  // clip z in [0, w] (WebGPU) or [-w, w] (WebGL)
  const box = boxOf(scene);

  // uniforms shared by the programs (names as in the GLSL / WGSL)
  const U = {
    u_view_proj: uniform(new THREE.Matrix4()), u_inv_view_proj: uniform(new THREE.Matrix4()),
    u_viewport: uniform(new THREE.Vector4()), u_proj11: uniform(1.0), u_camera_dir: uniform(new THREE.Vector3()),
    u_box_min: uniform(new THREE.Vector3(...box[0])), u_box_max: uniform(new THREE.Vector3(...box[1])),
    u_epsilon: uniform(1e-4 * diagonal(box)), u_step_scale: uniform(0.9), u_clip: uniform(0), u_clip_plane: uniform(new THREE.Vector4()),
    u_color_front: uniform(new THREE.Vector3()), u_color_back: uniform(new THREE.Vector3()), u_opacity: uniform(1.0),
    field_isovalue: uniform(0.0),
    u_plane_origin: uniform(new THREE.Vector3()), u_plane_u: uniform(new THREE.Vector3()), u_plane_v: uniform(new THREE.Vector3()),
    u_colormap_range: uniform(1.0), u_isoband_frequency: uniform(60.0), u_cut_opacity: uniform(1.0),
    u_point_radius: uniform(0.005), u_point_color: uniform(new THREE.Vector3()),
  };
  const MAX_STEPS = 128;  // a loop bound fixed at build time here (the GLSL / WGSL read u_max_steps)
  const fragDepth = (p) => {  // camera.glsl fragDepth: ½ z_ndc + ½
    const clip = U.u_view_proj.mul(vec4(p, 1.0));
    return clamp(clip.z.div(clip.w).mul(0.5).add(0.5), 0.0, 1.0);
  };
  const toClip = (clip) => (webgpuClip ? vec4(clip.xy, clip.z.add(clip.w).mul(0.5), clip.w) : clip);

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

  // surface.frag in TSL: clip half-space, sphere tracing with bisection on a sign change, Blinn-Phong head light, depth
  function surfaceMaterial(F, translucent) {
    const fieldDistance = (p) => F.gridDistance(p).sub(U.field_isovalue);
    const material = new THREE.NodeMaterial();
    material.transparent = translucent;
    material.depthWrite = !translucent;
    material.fragmentNode = Fn(() => {
      const ndc = vec2(screenUV.x.mul(2.0).sub(1.0), float(1.0).sub(screenUV.y.mul(2.0)));  // screenUV: origin top left
      const near = U.u_inv_view_proj.mul(vec4(ndc, -1.0, 1.0));
      const far = U.u_inv_view_proj.mul(vec4(ndc, 1.0, 1.0));
      const ro = near.xyz.div(near.w).toVar();
      const rd = normalize(far.xyz.div(far.w).sub(ro)).toVar();
      const inv = vec3(1.0).div(rd);
      const ta = U.u_box_min.sub(ro).mul(inv), tb = U.u_box_max.sub(ro).mul(inv);
      const tmin = min(ta, tb), tmax = max(ta, tb);
      const t0 = float(max(max(max(tmin.x, tmin.y), tmin.z), 0.0)).toVar();
      const t1 = float(min(min(tmax.x, tmax.y), tmax.z)).toVar();
      If(U.u_clip.equal(1), () => {  // keep n·x <= c:  t denom <= s
        const denom = dot(U.u_clip_plane.xyz, rd);
        const s = U.u_clip_plane.w.sub(dot(U.u_clip_plane.xyz, ro));
        If(denom.greaterThan(0.0), () => { t1.assign(min(t1, s.div(denom))); })
          .ElseIf(denom.lessThan(0.0), () => { t0.assign(max(t0, s.div(denom))); })
          .ElseIf(s.lessThan(0.0), () => { Discard(); });
      });
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
      const p = ro.add(rd.mul(t)).toVar();
      // gl_FragDepth / @builtin(frag_depth); r186's depth.assign returns a node without adding it to the shader's
      // statements: toStack() does
      depth.assign(fragDepth(p)).toStack();
      const n0 = normalize(F.gridDistanceGradient(p).yzw);
      const back = dot(n0, rd).greaterThan(0.0);
      const n = back.select(n0.negate(), n0);
      const base = back.select(U.u_color_back, U.u_color_front);
      const l = normalize(rd.negate().add(vec3(0.3, 1.0, 0.2).mul(0.3)));
      const hv = normalize(l.sub(rd));
      const diffuse = max(dot(n, l), 0.0);
      const specular = pow(max(dot(n, hv), 0.0), 48.0);
      return vec4(base.mul(diffuse.mul(0.8).add(0.2)).add(specular.mul(0.25)), U.u_opacity);
    })();
    return material;
  }
  const quads = {};
  for (const [name, F] of Object.entries(fields)) {
    for (const tr of [false, true]) quads[`${name}|${tr}`] = new THREE.QuadMesh(surfaceMaterial(F, tr));
  }

  // cutplane.vert + cutplane.frag in TSL: rectangle origin ± u ± v (4 vertices, 2 triangles), colormap of φ
  function cutPlaneMesh(F) {
    const material = new THREE.NodeMaterial();
    material.side = THREE.DoubleSide;  // three.js culls back faces by default; the GL hosts cull nothing
    const corner = positionGeometry.xy;  // (±1, ±1)
    const world = varying(U.u_plane_origin.add(U.u_plane_u.mul(corner.x)).add(U.u_plane_v.mul(corner.y)));
    material.vertexNode = toClip(U.u_view_proj.mul(vec4(world, 1.0)));
    material.fragmentNode = Fn(() => {
      const phi = F.gridDistance(world).sub(U.field_isovalue).toVar();
      const fw = fwidth(phi).toVar();  // before the discard
      const slack = U.u_box_max.sub(U.u_box_min).mul(1e-4);
      const lo = U.u_box_min.sub(slack), hi = U.u_box_max.add(slack);
      If(world.x.lessThan(lo.x).or(world.y.lessThan(lo.y)).or(world.z.lessThan(lo.z))
        .or(world.x.greaterThan(hi.x)).or(world.y.greaterThan(hi.y)).or(world.z.greaterThan(hi.z)), () => { Discard(); });
      return vec4(sdfColor(phi, fw, U.u_colormap_range, U.u_isoband_frequency), U.u_cut_opacity);
    })();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]), 3));
    g.setIndex([0, 1, 2, 2, 1, 3]);
    const mesh = new THREE.Mesh(g, material);
    mesh.frustumCulled = false;
    return mesh;
  }
  const cutMeshes = {};
  for (const [name, F] of Object.entries(fields)) cutMeshes[name] = cutPlaneMesh(F);
  const sceneOf = (mesh) => new THREE.Scene().add(mesh);  // renderer.render takes a scene root

  // points.vert + points.frag in TSL: one instanced quad per point (WebGPU points are 1 pixel), diameter
  // 2 r P[1][1] (h / 2) / w_clip pixels clamped to [1, 64]
  function pointsMesh(pc) {
    const positions = scene.arrays[pc.positions];
    const n = positions.shape[0];
    const pos = instancedBufferAttribute(new THREE.InstancedBufferAttribute(positions.data, 4));
    const nrm = pc.normals ? instancedBufferAttribute(new THREE.InstancedBufferAttribute(scene.arrays[pc.normals].data, 4)) : vec4(0.0);
    const material = new THREE.NodeMaterial();
    material.side = THREE.DoubleSide;
    const corner = positionGeometry.xy;
    material.vertexNode = Fn(() => {
      const clip = U.u_view_proj.mul(vec4(pos.xyz, 1.0)).toVar();
      const size = clamp(U.u_point_radius.mul(U.u_proj11).mul(U.u_viewport.w).div(clip.w), 1.0, 64.0);
      return toClip(vec4(clip.xy.add(corner.mul(size).div(U.u_viewport.zw).mul(clip.w)), clip.z, clip.w));
    })();
    const c = varying(corner), normal = varying(nrm.xyz), outlier = varying(pos.w);
    material.fragmentNode = Fn(() => {
      const r2 = dot(c, c);
      If(r2.greaterThan(1.0), () => { Discard(); });
      const col = select(outlier.greaterThan(0.5), vec3(0.85, 0.1, 0.1), U.u_point_color).toVar();
      const shade = float(1.0).sub(r2.mul(0.35)).toVar();
      if (pc.normals) {
        const nd = dot(normalize(normal), U.u_camera_dir.negate());
        col.assign(select(nd.greaterThanEqual(0.0), col, mix(col, vec3(0.55), 0.6)));
        shade.mulAssign(abs(nd).mul(0.55).add(0.45));
      }
      return vec4(col.mul(shade), 1.0);
    })();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0]), 3));
    const mesh = new THREE.Mesh(g, material);
    mesh.count = n;  // instances (WebGPURenderer)
    mesh.frustumCulled = false;
    return mesh;
  }
  const clouds = pointClouds(scene).map((pc) => [pc, pointsMesh(pc)]);
  const dummyCamera = new THREE.PerspectiveCamera();  // the materials above take their matrices from U, not from it

  function frame(camera, s, viewport) {
    const cam = camera.uniforms(viewport);
    U.u_view_proj.value.fromArray(columnMajor(cam.u_view_proj));
    U.u_inv_view_proj.value.fromArray(columnMajor(cam.u_inv_view_proj));
    U.u_viewport.value.set(...viewport);
    U.u_proj11.value = cam.u_proj11;
    U.u_camera_dir.value.set(...cam.u_camera_dir);
    U.u_epsilon.value = s.epsilon ?? 1e-4 * diagonal(box);
    U.u_step_scale.value = s.stepScale;
    const cut = cutUniforms(s, box);
    U.u_clip.value = cut.u_clip ? 1 : 0;
    U.u_clip_plane.value.set(...cut.u_clip_plane);
    renderer.setClearColor(new THREE.Color(...s.background), 1);
    renderer.clear();
    const passes = surfacePasses(scene, s, SUPPORTED);
    const surface = (p) => {
      U.field_isovalue.value = (p.field.params?.isovalue ?? 0) + s.isovalue;
      U.u_color_front.value.set(...p.front);
      U.u_color_back.value.set(...p.back);
      U.u_opacity.value = p.opacity;
      quads[`${p.field.name}|${p.opacity < 1}`].render(renderer);
    };
    for (const p of passes) if (p.opacity >= 1) surface(p);

    const cutField = s.showCutplane ? resolveField(scene, s.cutField, false, SUPPORTED) : null;
    if (cutField) {
      U.field_isovalue.value = (cutField.params?.isovalue ?? 0) + s.isovalue;
      U.u_plane_origin.value.set(...cut.u_plane_origin);
      U.u_plane_u.value.set(...cut.u_plane_u);
      U.u_plane_v.value.set(...cut.u_plane_v);
      U.u_colormap_range.value = cut.u_colormap_range;
      U.u_isoband_frequency.value = cut.u_isoband_frequency;
      U.u_cut_opacity.value = s.cutOpacity;
      const mesh = cutMeshes[cutField.name];
      mesh.material.transparent = s.cutOpacity < 1;  // blended, depth still written (as the GL hosts)
      renderer.render(sceneOf(mesh), dummyCamera);
    }
    if (s.showPoints) {
      for (const [pc, mesh] of clouds) {
        U.u_point_radius.value = s.pointRadius ?? pc.radius ?? 0.005;
        U.u_point_color.value.set(...s.pointColor);
        renderer.render(sceneOf(mesh), dummyCamera);
      }
    }
    for (const p of passes) if (p.opacity < 1) surface(p);
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
    for (const m of [...Object.values(cutMeshes), ...clouds.map(([, m]) => m)]) { m.material.dispose(); m.geometry.dispose(); }
    renderer.dispose();
  }

  return { draw, render, probe, release, supports: (fl) => SUPPORTED.includes(fl.type), name: forceWebGL ? 'three-tsl-webgl' : 'three-webgpu', three: renderer };
}
