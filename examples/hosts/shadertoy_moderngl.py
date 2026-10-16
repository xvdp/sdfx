"""
HOST — a single-file, Shadertoy-style moderngl viewer (uniforms iResolution, iTime, iTimeDelta, iFrame, iMouse;
a full-screen triangle; mainImage) that draws the results of each framework of this repository:

	1. a Shadertoy file (mainImage), e.g. SuperFit's assembly exported as code:
	       python -m sdfview superfit A.pkl --shadertoy examples/out/sup.glsl
	       python examples/hosts/shadertoy_moderngl.py examples/out/sup.glsl [--part -2]
	2. any field of a .sdfx scene, traced by the mainImage below with sdfview's GLSL field modules:
	       type "pat"  (Points as Tori: tori + BVH k-NN blend, exact)    examples/pat/01_fit.py
	       type "grid" (any torch SDF baked on a grid: PAT, SuperFit, GT)
	       type "glsl" (generated code: a SuperFit assembly and its parts)
	       python examples/hosts/shadertoy_moderngl.py examples/out/pat/tooth.sdfx --field pat_grid
	       python examples/hosts/shadertoy_moderngl.py examples/out/compare/knot.sdfx --field sup.sup --part -2

	window: left drag sets iMouse (orbit), Esc quits. Headless (EGL, no display): --png out.png [--time t]

What a field needs from the host (sdfview/shaders/glsl/field.glsl; sdfview/backends/moderngl/renderer.py does the same):
	program   glsl_source('field.glsl', 'gl330', defines, prelude) + your fragment code (no second #version):
	          defines FIELD_PAT (+ BVH_K_MAX >= k, BVH_STACK_SIZE >= bvh_depth + 2) | FIELD_GRID | FIELD_GLSL;
	          prelude = field['glsl']['code'] for FIELD_GLSL (generated code, inserted before the modules)
	functions fieldDistance(p) = φ(p) - isovalue, fieldDistanceGradient(p, out g), fieldPart(p)
	uniforms  all: field_isovalue
	          pat:  pat_points, pat_tori, pat_bvh_bounds (sampler2D RGBA32F), pat_bvh_nodes (isampler2D RGBA32I),
	                pat_bvh_prims (isampler2D R32I): array_texture(ctx, array), 4096 texels per row, NEAREST;
	                pat_k, pat_lambda_scale, pat_max_exp_arg
	          grid: grid_values (sampler3D R32F, LINEAR, clamped: grid_texture(ctx, values)), grid_min, grid_max
	          glsl: field_part (-1 whole, -2 all parts, i part i), field_gradient_step
"""

import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import moderngl  # noqa: E402
import numpy as np  # noqa: E402

VERT = """#version 330 core
void main() {  // one triangle covering the viewport, no vertex buffer
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
}
"""

SHADERTOY_UNIFORMS = """
uniform vec3 iResolution;   // viewport size in pixels (w, h, 1)
uniform float iTime;        // seconds since start
uniform float iTimeDelta;   // seconds since the previous frame
uniform int iFrame;         // frame index
uniform vec4 iMouse;        // xy: position while a button is held (pixels, origin bottom left); zw: click position,
                            // z > 0 while held
"""

MAIN = """
out vec4 shadertoy_color;
void main() { mainImage(shadertoy_color, gl_FragCoord.xy); }
"""

# mainImage for a .sdfx field: orbit camera from iMouse (or iTime when idle), sphere tracing, shading by part
FIELD_IMAGE = """
uniform vec3 box_min;   // scene bounds: camera framing
uniform vec3 box_max;
uniform bool color_parts;

vec3 partColor(int i) {
    float hue = fract(0.13 + 0.618034 * float(i));
    vec3 k = clamp(abs(fract(hue + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), k, 0.55);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 c = 0.5 * (box_min + box_max);
    float radius = 0.5 * length(box_max - box_min);
    vec2 m = iMouse.z > 0.0 ? iMouse.xy / iResolution.xy : vec2(0.1 + 0.02 * iTime, 0.6);
    float yaw = 6.2831853 * m.x, pitch = 3.0 * (m.y - 0.5);
    vec3 ro = c + 2.6 * radius * vec3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
    vec3 f = normalize(c - ro), r = normalize(cross(f, vec3(0.0, 1.0, 0.0))), u = cross(r, f);
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 rd = normalize(uv.x * r + uv.y * u + 2.0 * f);

    float t = 0.0, eps = 2e-4 * radius;
    bool hit = false;
    for (int i = 0; i < 256; i++) {  // sphere tracing; step 0.8: the PAT blend is not exactly 1-Lipschitz
        float d = fieldDistance(ro + t * rd);
        if (abs(d) < eps) { hit = true; break; }
        t += 0.8 * abs(d);
        if (t > 5.0 * radius) break;
    }
    vec3 col = vec3(0.96);
    if (hit) {
        vec3 p = ro + t * rd, g;
        fieldDistanceGradient(p, g);
        vec3 n = normalize(g);
        int part = fieldPart(p);
        vec3 base = (color_parts && part >= 0) ? partColor(part) : vec3(0.86, 0.70, 0.45);
        col = base * (0.25 + 0.75 * max(dot(n, -rd), 0.0));
    }
    fragColor = vec4(col, 1.0);
}
"""


class ShaderToy:
	"""A Shadertoy-style program: fragment code with mainImage, the Shadertoy uniforms, optional field bindings."""

	def __init__(self, ctx: moderngl.Context, source: str, field_name=None, part: int = -1, color_parts=False):
		self.ctx = ctx
		self.textures = []
		self.box = None
		if source.endswith('.sdfx'):
			frag = self._field_program(source, field_name)
		else:
			with open(source) as f:
				code = f.read()
			defines = f'#define PART {part}\n#define COLOR_PARTS {int(color_parts)}\n'
			frag = '#version 330 core\n' + defines + SHADERTOY_UNIFORMS + code + MAIN
		self.prog = ctx.program(vertex_shader=VERT, fragment_shader=frag)
		self.vao = ctx.vertex_array(self.prog, [])
		if self.field is not None:
			self._bind_field(part, color_parts)

	def _field_program(self, path: str, field_name):
		from sdfview.scene import Scene
		from sdfview.shaders import glsl_source

		self.scene = Scene.load(path)
		fields = {f['name']: f for f in self.scene.fields}
		self.field = fields[field_name] if field_name else self.scene.fields[0]
		f, params = self.field, self.field.get('params', {})
		print(f'field {f["name"]} ({f["type"]}); fields in the scene: {list(fields)}')
		prelude = None
		if f['type'] == 'pat':
			k = int(params.get('k', 32))
			defines = {'FIELD_PAT': None, 'BVH_K_MAX': max(32, k), 'BVH_STACK_SIZE': max(64, params['bvh_depth'] + 2)}
		elif f['type'] == 'grid':
			defines = {'FIELD_GRID': None}
		elif f['type'] == 'glsl':
			defines, prelude = {'FIELD_GLSL': None}, f['glsl']['code']
		else:
			raise ValueError(f'unsupported field type {f["type"]}')
		lo = self.scene.bounds_min if self.scene.bounds_min is not None else f['bounds']['min']
		hi = self.scene.bounds_max if self.scene.bounds_max is not None else f['bounds']['max']
		self.box = (tuple(float(x) for x in lo), tuple(float(x) for x in hi))
		return glsl_source('field.glsl', 'gl330', defines, prelude) + SHADERTOY_UNIFORMS + FIELD_IMAGE + MAIN

	field = None

	def _bind_field(self, part: int, color_parts: bool):
		from sdfview.backends.moderngl.renderer import array_texture, grid_texture

		f, params, prog = self.field, self.field.get('params', {}), self.prog
		arrays = self.scene.field_arrays(f)
		prog['field_isovalue'] = float(params.get('isovalue', 0.0))
		if f['type'] == 'pat':
			for unit, key in enumerate(('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims')):
				self.textures.append(array_texture(self.ctx, arrays[key]))
				self.textures[-1].use(unit)
				prog[f'pat_{key}'] = unit
			prog['pat_k'] = int(params.get('k', 32))
			prog['pat_lambda_scale'] = float(params.get('lambda_scale', 1.0))
			prog['pat_max_exp_arg'] = float(params.get('max_exp_arg', 64.0))
		elif f['type'] == 'grid':
			self.textures.append(grid_texture(self.ctx, arrays['values']))
			self.textures[-1].use(0)
			prog['grid_values'] = 0
			prog['grid_min'], prog['grid_max'] = tuple(f['bounds']['min']), tuple(f['bounds']['max'])
		elif f['type'] == 'glsl':
			prog['field_part'] = part if -2 <= part < f['glsl']['n_parts'] else -1
			prog['field_gradient_step'] = 5e-4 * float(np.linalg.norm(np.subtract(self.box[1], self.box[0])))
		prog['box_min'], prog['box_max'] = self.box
		if 'color_parts' in prog:
			prog['color_parts'] = bool(color_parts)

	def render(self, size, t: float, dt: float, frame: int, mouse=(0.0, 0.0, 0.0, 0.0)) -> None:
		values = {'iResolution': (float(size[0]), float(size[1]), 1.0), 'iTime': t, 'iTimeDelta': dt,
				  'iFrame': frame, 'iMouse': tuple(float(x) for x in mouse)}  # fmt: skip
		for name, value in values.items():
			if name in self.prog:  # unused uniforms are removed by the compiler
				self.prog[name] = value
		for unit, tex in enumerate(self.textures):  # rebind: another program may have used the units
			tex.use(unit)
		self.vao.render(moderngl.TRIANGLES, vertices=3)


def headless(a):
	from PIL import Image

	ctx = moderngl.create_standalone_context(backend='egl')
	w, h = a.size
	fbo = ctx.simple_framebuffer((w, h))
	fbo.use()
	toy = ShaderToy(ctx, a.source, a.field, a.part, a.color_parts)
	t = time.perf_counter()
	toy.render((w, h), a.time, 0.0, 0)
	data = fbo.read(components=3)
	print(f'rendered {w}x{h} in {time.perf_counter() - t:.2f} s')
	Image.frombytes('RGB', (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).save(a.png)
	print(f'wrote {a.png}')


def window(a):
	import moderngl_window as mglw

	class App(mglw.WindowConfig):
		gl_version = (3, 3)
		title = f'shadertoy host — {os.path.basename(a.source)}'
		window_size = tuple(a.size)
		resizable = True
		vsync = True

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.toy = ShaderToy(self.ctx, a.source, a.field, a.part, a.color_parts)
			self.mouse = [0.0, 0.0, 0.0, 0.0]
			self.frame = 0

		def on_render(self, t: float, dt: float) -> None:
			self.ctx.clear(1.0, 1.0, 1.0)
			self.toy.render(self.wnd.buffer_size, t, dt, self.frame, self.mouse)
			self.frame += 1

		def _pixels(self, x, y):  # window coordinates (origin top left) -> framebuffer pixels (origin bottom left)
			sx = self.wnd.buffer_size[0] / self.wnd.size[0]
			return x * sx, (self.wnd.size[1] - y) * sx

		def on_mouse_press_event(self, x, y, button):
			px, py = self._pixels(x, y)
			self.mouse = [px, py, px, py]

		def on_mouse_drag_event(self, x, y, dx, dy):
			self.mouse[0], self.mouse[1] = self._pixels(x, y)

		def on_mouse_release_event(self, x, y, button):
			self.mouse[2] = -abs(self.mouse[2])  # Shadertoy: z < 0 after release

	mglw.run_window_config(App, args=['--window', a.window])


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('source', help='Shadertoy fragment file (mainImage) or a scene (.sdfx)')
	ap.add_argument('--field', default=None, help='field of the scene (default: the first)')
	ap.add_argument('--part', type=int, default=-1, help='assembly: -1 whole, -2 all parts, i part i')
	ap.add_argument('--color-parts', action='store_true', help='assembly: color by part')
	ap.add_argument('--size', type=int, nargs=2, default=(960, 640))
	ap.add_argument('--png', default=None, help='render one frame to this file without a window (EGL)')
	ap.add_argument('--time', type=float, default=0.0, help='iTime of the --png frame')
	ap.add_argument('--window', default='pyglet', help="moderngl-window backend ('pyglet', 'glfw')")
	a = ap.parse_args()
	headless(a) if a.png else window(a)


if __name__ == '__main__':
	main()
