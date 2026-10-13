// scene.js — reader for .sdfx scenes (format: sdfview/scene.py), browser or Node, no dependencies.
//
//   const scene = parseScene(arrayBuffer)            // or: await loadScene(url)
//   scene.manifest.fields, scene.arrays['pat/points'] -> {data: Float32Array, shape: [n, 4], dtype: 'float32'}
//
// Typed arrays are views into the buffer (no copy): every array starts at a 16-byte aligned offset.

const MAGIC = 'SDFX';
const TYPED = { float32: Float32Array, int32: Int32Array, uint32: Uint32Array, uint8: Uint8Array };

export function parseScene(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) throw new Error('not an sdfx scene (bad magic)');
  const version = view.getUint32(4, true);
  if (version > 1) throw new Error(`scene version ${version} is newer than this reader (1)`);
  const jsonLength = view.getUint32(8, true);
  const manifest = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 16, jsonLength)));
  const base = 16 + jsonLength;
  const arrays = {};
  for (const [name, spec] of Object.entries(manifest.arrays)) {
    const T = TYPED[spec.dtype];
    if (!T) throw new Error(`${name}: unsupported dtype ${spec.dtype}`);
    const data = new T(buffer, base + spec.byteOffset, spec.byteLength / T.BYTES_PER_ELEMENT);
    arrays[name] = { data, shape: spec.shape, dtype: spec.dtype };
  }
  return { manifest, arrays };
}

export async function loadScene(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return parseScene(await response.arrayBuffer());
}
