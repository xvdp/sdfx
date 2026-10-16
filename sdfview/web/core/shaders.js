// shaders.js — assembles GLSL and WGSL programs from sdfview/shaders/{glsl,wgsl} for the JavaScript hosts, with the
// rules of sdfview/shaders/__init__.py (glsl_source, wgsl_source, preprocess); the output is identical (tested).
//
//   const src = await glslSource(read, 'surface.frag', 'webgl2', { FIELD_PAT: null, BVH_K_MAX: 32 }, prelude)
//   const wgsl = await wgslSource(read, 'surface.wgsl', { FIELD_GRID: null })
//
// `read(path)` returns the text of a module, path relative to sdfview/shaders ('glsl/common.glsl'): in a browser
// fetchReader(baseUrl), in Node a reader over the file system.

export const HEADERS = {
  gl330: '#version 330 core\n',
  webgl2:
    '#version 300 es\n' +
    'precision highp float;\nprecision highp int;\n' +
    'precision highp sampler2D;\nprecision highp isampler2D;\nprecision highp sampler3D;\n',
  // three.js RawShaderMaterial (glslVersion GLSL3) prepends '#version 300 es' itself
  three:
    'precision highp float;\nprecision highp int;\n' +
    'precision highp sampler2D;\nprecision highp isampler2D;\nprecision highp sampler3D;\n',
};

export function fetchReader(baseUrl) {
  const cache = new Map();
  return (path) => {
    if (!cache.has(path)) {
      cache.set(path, fetch(new URL(path, baseUrl)).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.text();
      }));
    }
    return cache.get(path);
  };
}

function defineLines(defines) {
  return Object.entries(defines || {}).map(([k, v]) => (v === null || v === true ? `#define ${k}` : `#define ${k} ${v}`));
}

// ---- GLSL: #include resolved here, everything else by GLSL's preprocessor

const INCLUDE = /^\s*#include\s+"([^"]+)"\s*$/gm;

async function expandGlsl(read, name, seen) {
  if (seen.has(name)) return `// (already included: ${name})\n`;
  seen.add(name);
  const text = await read(`glsl/${name}`);
  let out = '';
  let last = 0;
  for (const m of text.matchAll(INCLUDE)) {  // in order, as Python's re.sub
    out += text.slice(last, m.index) + (await expandGlsl(read, m[1], seen));
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

export async function glslSource(read, name, target = 'webgl2', defines = {}, prelude = null) {
  if (!(target in HEADERS)) throw new Error(`unknown target ${target}`);
  const lines = [HEADERS[target], ...defineLines(defines)];
  if (prelude) lines.push(prelude);
  return lines.join('\n') + '\n' + (await expandGlsl(read, name, new Set()));
}

// ---- WGSL: includes and conditionals resolved here (WGSL has no preprocessor)

const DIRECTIVE = /^\s*#\s*(include|ifdef|ifndef|if|elif|else|endif|error)\b\s*(.*?)\s*$/;
const DEFINED = /^!?\s*defined\s*\(\s*(\w+)\s*\)$/;

function condition(expr, defines) {
  for (const [op, combine] of [['||', 'some'], ['&&', 'every']]) {
    if (expr.includes(op)) return expr.split(op).map((e) => e.trim())[combine]((e) => condition(e, defines));
  }
  const m = DEFINED.exec(expr);
  if (!m) throw new Error(`unsupported #if expression '${expr}' (use defined(X), !defined(X), ||, &&)`);
  return (m[1] in defines) !== expr.startsWith('!');
}

export async function preprocess(read, dir, name, defines, seen = new Set()) {
  if (seen.has(name)) return `// (already included: ${name})\n`;
  seen.add(name);
  const lines = (await read(`${dir}/${name}`)).split('\n');
  const out = [];
  const stack = [];  // [active, branchTaken] per open #if
  let active = true;
  for (const line of lines) {
    const m = DIRECTIVE.exec(line);
    if (!m) {
      if (active) out.push(line);
      continue;
    }
    const [, kind, arg] = m;
    if (kind === 'ifdef' || kind === 'ifndef' || kind === 'if') {
      const cond = kind === 'ifdef' ? arg in defines : kind === 'ifndef' ? !(arg in defines) : condition(arg, defines);
      stack.push([active, active && cond]);
      active = active && cond;
    } else if (kind === 'elif') {
      const [parent, taken] = stack[stack.length - 1];
      const cond = parent && !taken && condition(arg, defines);
      stack[stack.length - 1] = [parent, taken || cond];
      active = cond;
    } else if (kind === 'else') {
      const [parent, taken] = stack[stack.length - 1];
      active = parent && !taken;
      stack[stack.length - 1] = [parent, true];
    } else if (kind === 'endif') {
      active = stack.pop()[0];
    } else if (!active) {
      continue;
    } else if (kind === 'include') {
      out.push((await preprocess(read, dir, arg.replace(/"/g, ''), defines, seen)).replace(/\n+$/, ''));
    } else if (kind === 'error') {
      throw new Error(`${name}: #error ${arg}`);
    }
  }
  if (stack.length) throw new Error(`${name}: unterminated #if`);
  return out.join('\n') + '\n';
}

export async function wgslSource(read, name, defines = {}, prelude = null) {
  const lines = Object.entries(defines || {})
    .filter(([, v]) => v !== null && v !== true)
    .map(([k, v]) => `const ${k} = ${v};`);
  if (prelude) lines.push(prelude);
  return lines.join('\n') + '\n' + (await preprocess(read, 'wgsl', name, defines));
}

// defines of a scene field (as Renderer._field_defines in sdfview/backends/moderngl/renderer.py)
export function fieldDefines(field) {
  const p = field.params || {};
  if (field.type === 'pat') {
    const k = Math.max(32, p.k ?? 32);
    return { FIELD_PAT: null, BVH_STACK_SIZE: Math.max(64, (p.bvh_depth ?? 62) + 2), BVH_K_MAX: k };
  }
  if (field.type === 'grid') return { FIELD_GRID: null };
  if (field.type === 'glsl') return { FIELD_GLSL: null };
  throw new Error(`unsupported field type ${field.type}`);
}
