// camera.js — orbit camera, the same formulas as sdfview/camera.py (OpenGL conventions: right-handed view space
// looking down -z, clip z in [-w, w]). Matrices are row-major arrays of 16 numbers, m[4 * row + col], as the NumPy
// ones; columnMajor(m) gives the order GLSL uniforms (WebGL uniformMatrix4fv) and WGSL mat4x4f expect.
//
//   eye  = target + distance (cos(pitch) (sin(yaw) e1 + cos(yaw) e2) + sin(pitch) up)
//   view = lookAt(eye, target, up);  proj = perspective(fovY, aspect, near, far)

const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

const sub = (a, b) => a.map((x, i) => x - b[i]);
const add = (a, b) => a.map((x, i) => x + b[i]);
const scale = (a, s) => a.map((x) => x * s);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const normalize = (a) => scale(a, 1 / norm(a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function lookAt(eye, target, up) {
  const f = normalize(sub(target, eye));
  let s = cross(f, up);
  if (norm(s) < 1e-9) s = cross(f, [up[2], up[0], up[1]]);  // looking along up
  s = normalize(s);
  const u = cross(s, f);
  return [
    s[0], s[1], s[2], -dot(s, eye),
    u[0], u[1], u[2], -dot(u, eye),
    -f[0], -f[1], -f[2], dot(f, eye),
    0, 0, 0, 1,
  ];
}

export function perspective(fovYDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovYDeg * Math.PI) / 360);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
    0, 0, -1, 0,
  ];
}

export function multiply(a, b) {
  const m = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) for (let k = 0; k < 4; k++) m[4 * r + c] += a[4 * r + k] * b[4 * k + c];
  return m;
}

export function invert(m) {  // general 4x4 inverse (cofactors)
  const inv = new Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const d = 1 / det;
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d; inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d; inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d; inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d; inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d; inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d; inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d; inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d; inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
  return inv;
}

export const columnMajor = (m) => new Float32Array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15].map((i) => m[i]));

export class OrbitCamera {
  constructor({ target = [0, 0, 0], distance = 3, yaw = 0.6, pitch = 0.4, fovY = 40, up = 'y', radius = 1 } = {}) {
    Object.assign(this, { target, distance, yaw, pitch, fovY, radius });
    this.up = typeof up === 'string' ? AXES[up] : up;
  }

  static fit(boundsMin, boundsMax, up = 'y') {
    const radius = 0.5 * norm(sub(boundsMax, boundsMin));
    const cam = new OrbitCamera({ target: scale(add(boundsMin, boundsMax), 0.5), up, radius });
    cam.distance = radius / Math.sin((cam.fovY * Math.PI) / 360);  // bounding sphere just inside the view cone
    return cam;
  }

  basis() {
    const up = normalize(this.up);
    const e1 = normalize(cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    return [e1, cross(e1, up), up];
  }

  eye() {
    const [e1, e2, up] = this.basis();
    const cp = Math.cos(this.pitch);
    const d = add(add(scale(e1, cp * Math.sin(this.yaw)), scale(e2, cp * Math.cos(this.yaw))), scale(up, Math.sin(this.pitch)));
    return add(this.target, scale(d, this.distance));
  }

  viewMatrix() { return lookAt(this.eye(), this.target, this.up); }

  projMatrix(aspect) {
    const near = Math.max(this.distance - 2 * this.radius, 1e-3 * this.distance);
    return perspective(this.fovY, aspect, near, this.distance + 2 * this.radius);
  }

  orbit(dx, dy, speed = 0.01) {
    this.yaw -= dx * speed;
    this.pitch = Math.min(1.55, Math.max(-1.55, this.pitch + dy * speed));
  }

  zoom(steps) { this.distance *= 0.9 ** steps; }

  // uniforms shared by every host's surface program (as Renderer._camera_uniforms in the moderngl backend)
  uniforms(viewport) {
    const vp = multiply(this.projMatrix(viewport[2] / Math.max(viewport[3], 1)), this.viewMatrix());
    return { u_view_proj: vp, u_inv_view_proj: invert(vp), u_viewport: viewport };
  }
}
