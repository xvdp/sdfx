// webgpu/renderer.js — raw WebGPU host: the same steps as webgl2/renderer.js and the moderngl Renderer, with the WGSL
// modules (sdfview/shaders/wgsl, the same functions as the GLSL):
//
//   1. device       navigator.gpu.requestAdapter() / requestDevice(); canvas context configured with the device
//   2. upload       PAT arrays -> storage buffers (the scene rows as stored: no 2D texel layout), grid -> r32float 3D
//                   texture (trilinear in the shader: textureLoad, exact weights)
//   3. pipelines    wgslSource('surface.wgsl' | 'probe.wgsl', field defines) -> render / compute pipelines (layout
//                   'auto'); bind groups: 0 uniform block (common.wgsl Uniforms), 1 field data, 2 probe buffers
//   4. per frame    uniforms written per surface pass, one render pass, one full-screen triangle per surface
//   render(...)     the same frame offscreen, read back as RGBA8 (tests, images)
//   probe(q)        compute shader, one invocation per query, results read back from a mapped buffer
//
// Field types: 'pat', 'grid' ('glsl' fields: generated GLSL, no WGSL yet; their grid bake is drawn instead).

import { columnMajor } from '../core/camera.js';
import { fetchReader, fieldDefines, wgslSource } from '../core/shaders.js';
import { boxOf, defaultSettings, diagonal, resolveField, surfacePasses } from '../core/view.js';

const SUPPORTED = ['pat', 'grid'];
const SHADERS = new URL('../../shaders/', import.meta.url);

// common.wgsl Uniforms: name -> [byte offset, type]
export const UNIFORMS = {
  u_view_proj: [0, 'mat4'], u_inv_view_proj: [64, 'mat4'], u_viewport: [128, 'vec4'],
  u_box_min: [144, 'vec3'], u_max_steps: [156, 'i32'], u_box_max: [160, 'vec3'], u_epsilon: [172, 'f32'],
  u_color_front: [176, 'vec3'], u_step_scale: [188, 'f32'], u_color_back: [192, 'vec3'], u_opacity: [204, 'f32'],
  grid_min: [208, 'vec3'], field_isovalue: [220, 'f32'], grid_max: [224, 'vec3'], pat_k: [236, 'i32'],
  pat_lambda_scale: [240, 'f32'], pat_max_exp_arg: [244, 'f32'], u_count: [248, 'i32'], u_color_parts: [252, 'i32'],
};
const UNIFORM_BYTES = 256;

export function packUniforms(values) {
  const buf = new ArrayBuffer(UNIFORM_BYTES);
  const f = new Float32Array(buf), i = new Int32Array(buf);
  for (const [name, v] of Object.entries(values)) {
    const spec = UNIFORMS[name];
    if (!spec) continue;
    const [offset, type] = spec;
    const at = offset / 4;
    if (type === 'mat4') f.set(columnMajor(v), at);
    else if (type === 'i32') i[at] = Number(v);
    else if (type === 'f32') f[at] = v;
    else f.set(v, at);
  }
  return buf;
}

export async function createRenderer(canvas, scene, { read = fetchReader(SHADERS) } = {}) {
  // 1. device
  if (!navigator.gpu) throw new Error('WebGPU is not available (a secure context and a supporting browser are needed)');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // 2. upload
  const storage = (array) => {
    const b = device.createBuffer({ size: Math.max(16, array.data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, array.data);
    return b;
  };
  const resources = {};  // field name -> [GPUBuffer | GPUTextureView, ...] in binding order
  for (const f of scene.manifest.fields) {
    const a = (key) => scene.arrays[f.arrays[key]];
    if (f.type === 'pat') {
      resources[f.name] = ['points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims'].map((k) => storage(a(k)));
    } else if (f.type === 'grid') {
      const [d, h, w] = a('values').shape;
      const tex = device.createTexture({ size: [w, h, d], dimension: '3d', format: 'r32float',
                                         usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture: tex }, a('values').data, { bytesPerRow: 4 * w, rowsPerImage: h }, [w, h, d]);
      resources[f.name] = [tex.createView()];
    }
  }
  const box = boxOf(scene);

  // 3. pipelines (per field; opaque and translucent variants of the surface)
  const modules = new Map();
  async function shaderModule(program, field) {
    const key = `${program}|${field.name}`;
    if (!modules.has(key)) {
      const code = await wgslSource(read, program, fieldDefines(field));
      const module = device.createShaderModule({ code, label: key });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) {
        const lines = code.split('\n');
        throw new Error(errors.map((m) => `${key}:${m.lineNum}:${m.linePos} ${m.message}\n  ${lines[m.lineNum - 1]}`).join('\n'));
      }
      modules.set(key, module);
    }
    return modules.get(key);
  }
  const surfacePipelines = new Map();
  async function surfacePipeline(field, translucent, colorFormat) {
    const key = `${field.name}|${translucent}|${colorFormat}`;
    if (!surfacePipelines.has(key)) {
      const module = await shaderModule('surface.wgsl', field);
      const blend = translucent
        ? { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } }
        : undefined;
      surfacePipelines.set(key, await device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: colorFormat, blend }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: !translucent, depthCompare: 'less' },
      }));
    }
    return surfacePipelines.get(key);
  }
  for (const f of scene.manifest.fields) {  // draw() records synchronously: build the pipelines now
    if (!SUPPORTED.includes(f.type)) continue;
    for (const fmt of [format, 'rgba8unorm']) for (const tr of [false, true]) await surfacePipeline(f, tr, fmt);
  }

  function fieldUniforms(field, s) {
    const p = field.params || {};
    const u = { field_isovalue: (p.isovalue ?? 0) + s.isovalue };
    if (field.type === 'pat') {
      u.pat_k = Math.min(s.k ?? p.k ?? 32, Math.max(32, p.k ?? 32));
      u.pat_lambda_scale = s.lambdaScale ?? p.lambda_scale ?? 1.0;
      u.pat_max_exp_arg = p.max_exp_arg ?? 64.0;
    } else if (field.type === 'grid') {
      u.grid_min = field.bounds.min;
      u.grid_max = field.bounds.max;
    }
    return u;
  }

  function fieldBindGroup(pipeline, field) {
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: resources[field.name].map((r, binding) => ({ binding, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    });
  }

  // 4. per frame
  let depth = null;
  function encodeFrame(colorView, colorFormat, width, height, camera, s, viewport) {
    if (!depth || depth.width !== width || depth.height !== height) {
      depth?.destroy();
      depth = device.createTexture({ size: [width, height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, clearValue: [...s.background, 1], loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setViewport(...viewport, 0, 1);
    const cam = camera.uniforms(viewport);
    const eps = s.epsilon ?? 1e-4 * diagonal(box);
    const passes = surfacePasses(scene, s, SUPPORTED);
    const ordered = [...passes.filter((p) => p.opacity >= 1), ...passes.filter((p) => p.opacity < 1)];
    const transient = [];
    for (const p of ordered) {
      const pipeline = surfacePipelines.get(`${p.field.name}|${p.opacity < 1}|${colorFormat}`);
      const ubuf = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      transient.push(ubuf);
      device.queue.writeBuffer(ubuf, 0, packUniforms({
        ...cam, ...fieldUniforms(p.field, s), u_box_min: box[0], u_box_max: box[1], u_max_steps: s.maxSteps,
        u_epsilon: eps, u_step_scale: s.stepScale, u_color_front: p.front, u_color_back: p.back,
        u_opacity: p.opacity, u_color_parts: s.colorParts,
      }));
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] }));
      pass.setBindGroup(1, fieldBindGroup(pipeline, p.field));
      pass.draw(3);
    }
    pass.end();
    return [encoder, transient];
  }

  function draw(camera, s, viewport) {
    const tex = context.getCurrentTexture();
    const [encoder, transient] = encodeFrame(tex.createView(), format, tex.width, tex.height, camera, s, viewport);
    device.queue.submit([encoder.finish()]);
    for (const b of transient) b.destroy();
  }

  // offscreen frame -> RGBA8 rows, bottom row first (as WebGL's readPixels and moderngl's Renderer.render flipped)
  async function render(camera, s, [w, h]) {
    const target = device.createTexture({ size: [w, h], format: 'rgba8unorm',
                                          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const [encoder, transient] = encodeFrame(target.createView(), 'rgba8unorm', w, h, camera, s, [0, 0, w, h]);
    const rowBytes = Math.ceil((4 * w) / 256) * 256;
    const buffer = device.createBuffer({ size: rowBytes * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyTextureToBuffer({ texture: target }, { buffer, bytesPerRow: rowBytes }, [w, h]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(4 * w * h);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * rowBytes, y * rowBytes + 4 * w), 4 * w * (h - 1 - y));
    buffer.unmap();
    for (const b of [...transient, buffer]) b.destroy();
    target.destroy();
    return out;
  }

  // (n, 3) queries -> Float32Array (n, 4): (φ - isovalue, ∇φ) from the compute shader
  async function probe(queries, fieldName = 'auto', s = defaultSettings()) {
    const field = resolveField(scene, fieldName, false, SUPPORTED);
    const n = queries.length / 3;
    const q4 = new Float32Array(Math.max(n, 1) * 4);
    for (let i = 0; i < n; i++) q4.set(queries.subarray(3 * i, 3 * i + 3), 4 * i);
    const module = await shaderModule('probe.wgsl', field);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'cs_main' } });
    const ubuf = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ubuf, 0, packUniforms({ ...fieldUniforms(field, s), u_count: n }));
    const qbuf = device.createBuffer({ size: q4.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(qbuf, 0, q4);
    const obuf = device.createBuffer({ size: q4.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rbuf = device.createBuffer({ size: q4.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] }));
    pass.setBindGroup(1, fieldBindGroup(pipeline, field));
    pass.setBindGroup(2, device.createBindGroup({ layout: pipeline.getBindGroupLayout(2),
      entries: [{ binding: 0, resource: { buffer: qbuf } }, { binding: 1, resource: { buffer: obuf } }] }));
    pass.dispatchWorkgroups(Math.ceil(Math.max(n, 1) / 64));
    pass.end();
    encoder.copyBufferToBuffer(obuf, 0, rbuf, 0, q4.byteLength);
    device.queue.submit([encoder.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rbuf.getMappedRange().slice(0, n * 16));
    rbuf.unmap();
    for (const b of [ubuf, qbuf, obuf, rbuf]) b.destroy();
    return out;
  }

  function release() {
    for (const rs of Object.values(resources)) for (const r of rs) if (r instanceof GPUBuffer) r.destroy();
    depth?.destroy();
    device.destroy();
  }

  return { draw, render, probe, release, supports: (f) => SUPPORTED.includes(f.type), name: 'webgpu' };
}
