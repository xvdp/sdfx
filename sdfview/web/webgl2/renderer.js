// webgl2/renderer.js — raw WebGL2 host: the moderngl Renderer (sdfview/backends/moderngl/renderer.py) step by step,
// with the same GLSL modules (sdfview/shaders/glsl, target 'webgl2' = GLSL ES 3.00) and the same texture layout.
//
//   1. context      canvas.getContext('webgl2'); EXT_color_buffer_float (float render targets: probes),
//                   OES_texture_float_linear (linear filtering of R32F grids; nearest without it)
//   2. upload       arrayTexture: (n, 4·t) rows -> 2D texture of RGBA32F / RGBA32I texels, 4096 per row (R32I for
//                   one column); gridTexture: (D, H, W) -> 3D R32F texture (W, H, D)
//   3. programs     glslSource(fullscreen.vert, surface.frag | probe.frag, field defines, generated code)
//   4. per frame    camera / field / surface uniforms, one full-screen triangle per surface pass
//   render(...)     the same frame into an offscreen RGBA8 target, read back (tests, images)
//   probe(q)        fieldDistanceGradient at query points: render probe.frag into an RGBA32F target, read back
//
//   const r = await createRenderer(canvas, scene);  r.draw(camera, settings, [0, 0, w, h]);  await r.probe(q, 'pat')

import { fetchReader, fieldDefines, glslSource } from '../core/shaders.js';
import { boxOf, diagonal, fieldPart, resolveField, surfacePasses } from '../core/view.js';
import { columnMajor } from '../core/camera.js';

export const TEXTURE_WIDTH = 4096;
const SUPPORTED = ['pat', 'grid', 'glsl'];
const SHADERS = new URL('../../shaders/', import.meta.url);

export function arrayTexture(gl, array, width = TEXTURE_WIDTH) {
  const { data, shape, dtype } = array;
  const cols = shape[shape.length - 1];
  const comps = cols % 4 === 0 ? 4 : cols;
  if (comps !== 1 && comps !== 4) throw new Error(`array texture rows must have 1 or a multiple of 4 values: ${shape}`);
  const count = Math.max(data.length / comps, 1);
  const w = Math.min(width, count), h = Math.ceil(count / w);
  const padded = new data.constructor(w * h * comps);
  padded.set(data);
  const isInt = dtype === 'int32';
  const formats = {
    'float32,4': [gl.RGBA32F, gl.RGBA, gl.FLOAT], 'float32,1': [gl.R32F, gl.RED, gl.FLOAT],
    'int32,4': [gl.RGBA32I, gl.RGBA_INTEGER, gl.INT], 'int32,1': [gl.R32I, gl.RED_INTEGER, gl.INT],
  };
  const [internal, format, type] = formats[`${dtype},${comps}`];
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, padded);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  tex.isInt = isInt;
  return tex;
}

export function gridTexture(gl, array, linear) {
  const [d, h, w] = array.shape;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, w, h, d, 0, gl.RED, gl.FLOAT, array.data);
  const filter = linear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
  return tex;
}

function compile(gl, type, source) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

function link(gl, vert, frag) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vert));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  prog.uniforms = {};  // name -> {location, type}: unused uniforms are removed by the compiler, setters skip them
  for (let i = 0; i < gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i++) {
    const info = gl.getActiveUniform(prog, i);
    prog.uniforms[info.name] = { location: gl.getUniformLocation(prog, info.name), type: info.type };
  }
  return prog;
}

function setUniform(gl, prog, name, value) {
  const u = prog.uniforms[name];
  if (!u) return;
  const v = typeof value === 'boolean' ? Number(value) : value;
  switch (u.type) {
    case gl.FLOAT: return gl.uniform1f(u.location, v);
    case gl.FLOAT_VEC2: return gl.uniform2fv(u.location, v);
    case gl.FLOAT_VEC3: return gl.uniform3fv(u.location, v);
    case gl.FLOAT_VEC4: return gl.uniform4fv(u.location, v);
    case gl.FLOAT_MAT4: return gl.uniformMatrix4fv(u.location, false, columnMajor(v));
    default: return gl.uniform1i(u.location, v);  // int, bool, samplers (texture unit)
  }
}

export async function createRenderer(canvas, scene, { read = fetchReader(SHADERS) } = {}) {
  // 1. context
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 is not available');
  const floatTargets = !!gl.getExtension('EXT_color_buffer_float');
  const floatLinear = !!gl.getExtension('OES_texture_float_linear');

  // 2. upload: every field's arrays once
  const textures = {};
  for (const f of scene.manifest.fields) {
    const arrays = (key) => scene.arrays[f.arrays[key]];
    if (f.type === 'pat') {
      for (const key of ['points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims']) textures[`${f.name}/${key}`] = arrayTexture(gl, arrays(key));
    } else if (f.type === 'grid') {
      textures[`${f.name}/values`] = gridTexture(gl, arrays('values'), floatLinear);
    }
  }
  const box = boxOf(scene);
  const vao = gl.createVertexArray();

  // 3. programs, compiled once per (stage files, field, defines)
  const programs = new Map();
  async function program(vert, frag, field) {
    const key = `${vert}|${frag}|${field ? field.name : ''}`;
    if (!programs.has(key)) {
      const defines = field ? fieldDefines(field) : {};
      const prelude = field && field.type === 'glsl' ? field.glsl.code : null;
      const vs = await glslSource(read, vert, 'webgl2', defines);
      const fs = await glslSource(read, frag, 'webgl2', defines, prelude);
      programs.set(key, link(gl, vs, fs));
    }
    return programs.get(key);
  }
  // compile the surface programs now (draw() is synchronous)
  for (const f of scene.manifest.fields) if (SUPPORTED.includes(f.type)) await program('fullscreen.vert', 'surface.frag', f);

  function bindField(prog, field, s, unit = 0) {
    const p = field.params || {};
    setUniform(gl, prog, 'field_isovalue', (p.isovalue ?? 0) + s.isovalue);
    if (field.type === 'pat') {
      for (const key of ['points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims']) {
        if (!prog.uniforms[`pat_${key}`]) continue;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, textures[`${field.name}/${key}`]);
        setUniform(gl, prog, `pat_${key}`, unit++);
      }
      const kMax = Math.max(32, p.k ?? 32);
      setUniform(gl, prog, 'pat_k', Math.min(s.k ?? p.k ?? 32, kMax));
      setUniform(gl, prog, 'pat_lambda_scale', s.lambdaScale ?? p.lambda_scale ?? 1.0);
      setUniform(gl, prog, 'pat_max_exp_arg', p.max_exp_arg ?? 64.0);
    } else if (field.type === 'grid') {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_3D, textures[`${field.name}/values`]);
      setUniform(gl, prog, 'grid_values', unit++);
      setUniform(gl, prog, 'grid_min', field.bounds.min);
      setUniform(gl, prog, 'grid_max', field.bounds.max);
    } else if (field.type === 'glsl') {
      setUniform(gl, prog, 'field_part', fieldPart(field, s));
      setUniform(gl, prog, 'field_gradient_step', 5e-4 * diagonal([field.bounds.min, field.bounds.max]));
    }
    return unit;
  }

  // 4. per frame
  function drawSurface(pass, s, cam, eps) {
    const prog = programs.get(`fullscreen.vert|surface.frag|${pass.field.name}`);
    gl.useProgram(prog);
    bindField(prog, pass.field, s);
    for (const [k, v] of Object.entries(cam)) setUniform(gl, prog, k, v);
    setUniform(gl, prog, 'u_box_min', box[0]);
    setUniform(gl, prog, 'u_box_max', box[1]);
    setUniform(gl, prog, 'u_max_steps', s.maxSteps);
    setUniform(gl, prog, 'u_epsilon', eps);
    setUniform(gl, prog, 'u_step_scale', s.stepScale);
    setUniform(gl, prog, 'u_clip', false);
    setUniform(gl, prog, 'u_clip_plane', [0, 0, 1, 0]);
    setUniform(gl, prog, 'u_color_front', pass.front);
    setUniform(gl, prog, 'u_color_back', pass.back);
    setUniform(gl, prog, 'u_opacity', pass.opacity);
    setUniform(gl, prog, 'u_color_parts', s.colorParts);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function draw(camera, s, viewport, framebuffer = null) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(...viewport);
    gl.clearColor(...s.background, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    const cam = camera.uniforms(viewport);
    const eps = s.epsilon ?? 1e-4 * diagonal(box);
    const passes = surfacePasses(scene, s, SUPPORTED);
    for (const p of passes) if (p.opacity >= 1) drawSurface(p, s, cam, eps);
    const translucent = passes.filter((p) => p.opacity < 1);
    if (translucent.length) {  // blended over the opaque ones, without writing depth
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      for (const p of translucent) drawSurface(p, s, cam, eps);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }
  }

  // (n, 3) queries -> Float32Array (n, 4): (φ - isovalue, ∇φ) from the shader
  async function probe(queries, fieldName = 'auto', s = null) {
    if (!floatTargets) throw new Error('probe needs EXT_color_buffer_float');
    const settings = s ?? (await import('../core/view.js')).defaultSettings();
    const field = resolveField(scene, fieldName, false, SUPPORTED);
    const n = queries.length / 3;
    const q4 = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) q4.set(queries.subarray(3 * i, 3 * i + 3), 4 * i);
    const qtex = arrayTexture(gl, { data: q4, shape: [n, 4], dtype: 'float32' });
    const w = Math.min(TEXTURE_WIDTH, Math.max(n, 1)), h = Math.ceil(Math.max(n, 1) / w);
    const out = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, out);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out, 0);
    const prog = await program('fullscreen.vert', 'probe.frag', field);
    gl.useProgram(prog);
    const unit = bindField(prog, field, settings);
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, qtex);
    setUniform(gl, prog, 'u_queries', unit);
    setUniform(gl, prog, 'u_count', n);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const data = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(out);
    gl.deleteTexture(qtex);
    return data.subarray(0, n * 4);
  }

  // a frame into an offscreen RGBA8 target -> rows, bottom row first (as moderngl's Renderer.render, unflipped)
  async function render(camera, s, [w, h]) {
    const color = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, color);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h);
    const depthRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);
    draw(camera, s, [0, 0, w, h], fbo);
    const out = new Uint8Array(4 * w * h);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteRenderbuffer(color);
    gl.deleteRenderbuffer(depthRb);
    return out;
  }

  function release() {
    for (const t of Object.values(textures)) gl.deleteTexture(t);
    for (const p of programs.values()) gl.deleteProgram(p);
    gl.deleteVertexArray(vao);
  }

  return { draw, render, probe, release, supports: (f) => SUPPORTED.includes(f.type), name: 'webgl2' };
}
