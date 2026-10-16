# sdfview — parallel hosts

The same viewer written against six graphics / compute interfaces, with the same module names, function names, uniform
names and data layout in each, so that files can be read side by side. Every host implements one contract (the
moderngl `Renderer`'s):

| member | Python hosts (moderngl, slangpy) | JavaScript hosts (WebGL2, WebGPU, three.js) |
|---|---|---|
| create | `Renderer(ctx \| None, scene)` | `await createRenderer(canvas, scene, options)` |
| draw | `draw(camera, settings, viewport)` (moderngl: into the bound framebuffer) | `draw(camera, settings, [x, y, w, h])` into the canvas |
| render | `render(camera, settings, (w, h))` → `(h, w, 4)` uint8, **first row = top** | `await render(camera, settings, [w, h])` → RGBA8, **rows bottom-up** (glReadPixels order) |
| probe | `probe(queries (n, 3), field, settings)` → `(n, 4)` float32 | `await probe(Float32Array(3 n), field, settings)` → `Float32Array(4 n)` |
| result of probe | `(φ(q) - isovalue, ∇φ(q))` per query | same |
| release | `release()` | `release()` |

Field choice per model, surface passes, model colors and the scene box are host independent: `sdfview/view.py`
(Python) and `web/core/view.js` (JavaScript), the same rules.

## Files

| module | GLSL (`shaders/glsl`) | WGSL (`shaders/wgsl`) | Slang (`shaders/slang`) | TSL (`web/three/webgpu_renderer.js`) |
|---|---|---|---|---|
| constants, array access | `common.glsl` | `common.wgsl` (+ the uniform struct) | `points_as_tori/slang/common.slang` | — |
| BVH k-NN | `bvh.glsl` | `bvh.wgsl` | `points_as_tori/slang/bvh.slang` | — |
| PAT field | `field_pat.glsl` | `field_pat.wgsl` | `field_pat.slang` (`PatField`) | — |
| grid field | `field_grid.glsl` | `field_grid.wgsl` | `field_grid.slang` (`GridField`) | `gridFunctions()` |
| field selection | `field.glsl` (`#if FIELD_*`) | `field.wgsl` (`#if FIELD_*`) | `field.slang` (`interface IField`) | one material per field |
| camera | `camera.glsl` | `camera.wgsl` | `camera.slang` (`Camera`) | inline in `surfaceMaterial()` |
| surface | `fullscreen.vert` + `surface.frag` | `surface.wgsl` (`vs_main`, `fs_main`) | `surface.slang` (`surface<F>`, compute) | `surfaceMaterial()` |
| colormap | `colormap.glsl` | `colormap.wgsl` | — | `sdfColor()` |
| cut plane | `cutplane.vert` + `cutplane.frag` | `cutplane.wgsl` | — | `cutPlaneMesh()` |
| points | `points.vert` + `points.frag` (point sprites) | `points.wgsl` (instanced quads) | — | `pointsMesh()` (instanced quads) |
| probe | `fullscreen.vert` + `probe.frag` | `probe.wgsl` (`cs_main`, compute) | `probe.slang` (`probe<F>`) | `Fn(...).compute(n)` |

Function names are the same in every language (`fetchBVHNode`, `bvhKNearestNeighbors`, `accumulateTorusPoint`,
`patDistanceGradient`, `gridSample`, `gridDistanceGradient`, `fieldDistance`, `fieldDistanceGradient`, `cameraRay`,
`fragDepth`, `intersectAABB`, `shade`, `partColor`, `sdfColor`, `colormap_SDF`).

Hosts:

| host | file | shaders | field types | draws |
|---|---|---|---|---|
| moderngl | `backends/moderngl/renderer.py` | GLSL, target `gl330` | pat, grid, glsl | surfaces, cut plane, points |
| slangpy | `backends/slangpy/renderer.py` | Slang | pat, grid | surfaces |
| WebGL2 | `web/webgl2/renderer.js` | GLSL, target `webgl2` | pat, grid, glsl | surfaces, cut plane, points |
| WebGPU | `web/webgpu/renderer.js` | WGSL | pat, grid | surfaces, cut plane, points |
| three.js WebGLRenderer | `web/three/webgl_renderer.js` | GLSL, target `three`, in `RawShaderMaterial`s | pat, grid, glsl | surfaces, cut plane, points |
| three.js WebGPURenderer | `web/three/webgpu_renderer.js` | TSL (compiled by three.js to WGSL, or GLSL with `forceWebGL`) | grid | surfaces, cut plane, points |

The tori inspector (a click: the blend at the clicked point, a torus, its principal curvatures; NOTES.md) is in the
moderngl window only for now; what it draws comes from `sdfview/inspector.py` (positions, sizes, colors, the box of
the torus), host independent, with `field_torus.glsl`, `marks.*`, `lines.*` on the GLSL side.

Viewers: the same keys and mouse in the moderngl window and the browser pages (`HELP_LINES` in
`backends/moderngl/window.py`, `HELP` in `web/core/app.js`); H shows / hides them on screen. The cut plane is
optional everywhere: key C, `ViewSettings.show_cutplane` / `showCutplane`, CLI `--no-cut`.

Pages: `web/webgl2/`, `web/webgpu/`, `web/three/webgl.html`, `web/three/webgpu.html` (`?backend=webgl`: the TSL on
three.js' WebGL2 backend), `web/three/compose.html` (an SDF as one object of a three.js scene). `python -m sdfview serve
scene.sdfx` prints one URL per page.

## Steps

| step | moderngl | WebGL2 | WebGPU | three.js WebGL | three.js WebGPU (TSL) | slangpy |
|---|---|---|---|---|---|---|
| 1. context | `moderngl.create_context()` / standalone (EGL) | `canvas.getContext('webgl2')`, `EXT_color_buffer_float` | `requestAdapter()`, `requestDevice()`, `context.configure` | `new WebGLRenderer({canvas})` | `new WebGPURenderer({canvas, forceWebGL})`, `await init()` | `spy.create_device(include_paths)`, `Module.load_from_file` |
| 2. upload | 2D textures RGBA32F / RGBA32I, 4096 texels per row; 3D R32F | same | storage buffers (rows as stored); 3D `r32float` texture | `DataTexture` / `Data3DTexture` (same formats) | `Data3DTexture` (grid) | `spy.Tensor` per array, rows of 4; grid `(D, H, W)` |
| 3. programs | `glsl_source(stage, 'gl330', defines, prelude)` | `glslSource(..., 'webgl2', ...)` | `wgslSource(name, defines)` → render / compute pipelines | `glslSource(..., 'three', ...)` → `RawShaderMaterial` | `NodeMaterial.fragmentNode = Fn(...)` | generic `surface<F>` / `probe<F>`, specialized by `{'_type': ...}`; `.constants({BVH_K_MAX, BVH_STACK_SIZE})` |
| 4. per frame | uniforms; opaque surfaces (full-screen triangle), cut plane (strip of 4), points (`POINTS`), translucent surfaces | same | one uniform buffer per draw, one render pass, the same order; points as instanced quads | the same objects ordered by `renderOrder` 0-3 | uniform nodes; `QuadMesh.render` per surface, `renderer.render` for the cut plane and points, the same order | color / depth tensors; one `surface(spy.grid((h, w)), ...)` call per pass |
| render | FBO RGBA8 + depth, `fbo.read` | FBO, `readPixels` | offscreen texture, `copyTextureToBuffer` | `WebGLRenderTarget`, `readRenderTargetPixels` | `RenderTarget`, `readRenderTargetPixelsAsync` (rows padded to 256 B: unpadded, flipped) | `color.to_numpy()` |
| probe | `probe.frag` into an RGBA32F texture, one texel per query | same | compute shader, mapped read-back buffer | `probe.frag` into a float render target | `instancedArray` + `computeAsync` + `getArrayBufferAsync` | `probe(field, iso, Tensor (n, 3))` → `(n, 4)` |

Fixed-function steps written out in the compute hosts (slangpy): `discard` (return), the depth test (less, against a
depth tensor), blending (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`: a lerp by the opacity, depth not written), float → unorm8
(`round(255 clamp(c))`).

## Names and layout

Uniforms (GLSL loose uniforms; WGSL members of `struct Uniforms`, group 0 binding 0; Slang members of `Camera`,
`Trace`, `Shading` and the field structs; TSL uniform nodes):

| GLSL / WGSL | Slang | meaning |
|---|---|---|
| `u_view_proj`, `u_inv_view_proj`, `u_viewport` | `Camera.view_proj`, `.inv_view_proj`, `.viewport` | P V, (P V)⁻¹, x y w h |
| `u_box_min`, `u_box_max`, `u_max_steps`, `u_epsilon`, `u_step_scale` | `Trace.box_min`, ... | sphere tracing |
| `u_clip`, `u_clip_plane` | `Trace.clip`, `.clip_plane` | half-space n·x ≤ c |
| `field_isovalue` | `Trace.isovalue` / probe argument | level set |
| `u_color_front`, `u_color_back`, `u_opacity`, `u_color_parts` | `Shading.*` | shading |
| `pat_points`, `pat_tori`, `pat_bvh_bounds`, `pat_bvh_nodes`, `pat_bvh_prims` | `PatField.points`, ... | PAT arrays |
| `pat_k`, `pat_lambda_scale`, `pat_max_exp_arg` | `PatField.k`, `.lambda_scale`, `.max_exp_arg` | PAT parameters |
| `grid_values`, `grid_min`, `grid_max` | `GridField.values`, `.grid_min`, `.grid_max` (+ `.res`) | grid |
| `u_plane_origin`, `u_plane_u`, `u_plane_v`, `u_colormap_range`, `u_isoband_frequency` | — | cut plane |
| `u_positions`, `u_normals`, `u_has_normals`, `u_point_radius`, `u_point_color`, `u_proj11`, `u_camera_dir` | — | points |

WGSL uniform block, byte offsets (WGSL uniform-buffer alignment: a `vec3f` is 16-byte aligned and a scalar fills its
fourth slot): `u_view_proj` 0, `u_inv_view_proj` 64, `u_viewport` 128, `u_box_min` 144, `u_max_steps` 156, `u_box_max` 160,
`u_epsilon` 172, `u_color_front` 176, `u_step_scale` 188, `u_color_back` 192, `u_opacity` 204, `grid_min` 208,
`field_isovalue` 220, `grid_max` 224, `pat_k` 236, `pat_lambda_scale` 240, `pat_max_exp_arg` 244, `u_count` 248,
`u_color_parts` 252, `u_clip_plane` 256, `u_plane_origin` 272, `u_clip` 284, `u_plane_u` 288, `u_colormap_range` 300,
`u_plane_v` 304, `u_isoband_frequency` 316, `u_camera_dir` 320, `u_point_radius` 332, `u_point_color` 336, `u_proj11` 348,
`u_has_normals` 352; 368 bytes, a multiple of the struct's 16-byte alignment (`web/webgpu/renderer.js` `UNIFORMS`).
`u_positions`, `u_normals` are storage buffers (group 1) in WGSL.

Arrays: the scene stores every array as rows of 4 values (`scene.py`); an element of 4 t values is t rows. GL: row j
is texel (j mod W, j div W) of a 2D texture (`fetch_texel`); WGSL: entry j of a storage buffer of `vec4`; Slang: row j
of a `(rows, 4)` tensor. Grid: `values[z][y][x]`, `(D, H, W)`, samples on the vertices.

Matrices: mathematical (row-major) in Python and Slang (`mul(M, v)`; slangpy `spy.float4x4(numpy)`); column-major in
GLSL, WGSL and three.js (`columnMajor()` in `web/core/camera.js`). Window coordinates: origin bottom left (GL,
Slang pixel grid); WebGPU's `@builtin(position)` has its origin top left: `cameraRay` flips y. Clip space: the
matrices are OpenGL's (z in [-w, w]); WebGPU clips z to [0, w], so its vertex shaders (and the TSL ones on the WebGPU
backend) output z' = ½ (z + w): window depth ½ z_ndc + ½, the value `fragDepth` writes for traced surfaces, so
rasterized geometry and surfaces compare in one depth buffer.

## Deviations

- Grid sampling: the GL hosts filter with the hardware (`texture()`, linear; the weights may be rounded to 8 bits on
  GPUs); WGSL, Slang and TSL interpolate from 8 exact loads (`r32float` is not filterable in WebGPU without the
  `float32-filterable` feature).
- `glsl` fields (SuperFit assemblies: generated GLSL) exist only in the GLSL hosts; the others draw the field's grid
  bake (`--bake N`).
- Cut plane and points: every host except slangpy (a compute host: the plane would be a per-pixel ray-plane hit,
  the sprites a scatter with atomic depth; not written).
- Points: GL rasterizes point sprites (`gl_PointSize`, `gl_PointCoord`); WebGPU's point primitives are 1 pixel wide, so
  WGSL and TSL draw one instanced quad (6 vertices) per point, of the same size: diameter 2 r P₁₁ (h / 2) / w_clip px,
  clamped to [1, 64].
- The cut plane's zero contour uses `fwidth(φ)`, taken before the fragment can discard (outside the box): derivatives
  are undefined after a discard in the 2x2 quad (GLSL) and must be in uniform control flow (WGSL). Before this order
  llvmpipe drew dark pixels along the box edges, SwiftShader did not.
- TSL: loops have a bound fixed when the material is built (128 steps); `u_max_steps` is not read.
- Precision: GLSL ES leaves sin / cos / tan precision to the implementation; SwiftShader (Chromium without a GPU):
  sin, cos 1.9e-4, tan 7.6e-5 (relative), which bounds the SuperFit field's error in the WebGL hosts there (1.7e-4).

## three.js notes (r186)

- `WebGPURenderer` accepts node materials only; native WGSL / GLSL enters as single functions (`wgslFn`, `glslFn`)
  without module-level storage bindings, so the PAT field (BVH traversal over storage buffers) is not in TSL here.
- r186 sets `swizzle: 'rgba'` in every texture view descriptor (the current WebGPU spec: a string, the identity);
  Chromium ≤ 141 typed that member as a dictionary and rejects it. `webgpu_renderer.js` drops the identity swizzle
  only where the browser rejects it (a probe at start).
- Chromium 141 (Dawn): a partial `writeTexture` into a 3D texture with `RENDER_ATTACHMENT` usage fails validation
  in Dawn's lazy zero-fill (a 2D view of a 3D texture); three.js uploads 3D textures one slice at a time with that
  usage. The headless tests launch Chromium with `--disable-dawn-features=lazy_clear_resource_on_first_use`.
- r186 requests a compatibility-mode adapter (`featureLevel: 'compatibility'`).
- Meshes are back-face culled by default (`side = FrontSide`); the GL hosts cull nothing: the cut plane and point
  quads use `DoubleSide` (with the normal along z the plane faced away and was culled).
- `depth.assign(value)` returns a node (`depthBase(value)`) without adding it to the shader's statements; inside a
  `Fn` it needs `.toStack()` (NodeMaterial.setupDepth does the same). Without it the traced surface kept the quad's
  depth and hid the cut plane.
- The WebGL backend's `dispose()` calls `WEBGL_lose_context.loseContext()`: a canvas cannot host a second renderer.
  The pages create a fresh canvas per scene.
- TSL capabilities for a PAT port, measured (WebGPU backend / WebGL backend): a local `array('int', 64).toVar()` stack
  with dynamic-index reads and writes inside a `Loop(condition)` while-loop gives the expected result on both
  (a depth-first walk of a 63-node binary tree: Σ node ids = 1953 on both); a storage buffer (`instancedArray`) read
  in a material's fragment shader with a varying or uniform index: correct on WebGPU, element 0 for every index on
  the WebGL backend (r186). Random access from a material on both backends therefore goes through textures
  (`textureLoad`), the GL hosts' layout.

## Deferred rendering (not implemented; where it would go)

Every host draws forward: each surface pass traces and shades. A deferred variant splits this in two:

1. geometry pass, per surface: trace, write a G-buffer instead of a color —
   `depth` (window depth, as now), `normal` RGBA16F (n.xyz, part id), `material` RGBA8 (model index, opacity,
   front/back flag);
2. lighting pass, once per frame: one full-screen pass reads the G-buffer and shades.

The lighting cost then no longer scales with the number of surface passes, and screen-space effects (ambient
occlusion, outlines, several lights) read the same buffers; translucent passes stay forward (one sample per pixel
in a G-buffer). Per host: WebGL2 `drawBuffers` (MRT) with `EXT_color_buffer_float`; WebGPU several color targets
of one render pass; three.js `WebGLRenderTarget(w, h, {count: 3})`; TSL `mrt({...})` + `pass()`; moderngl
`ctx.framebuffer([tex0, tex1, tex2], depth)`; Slang extra `RWTensor` outputs.

## Tests

| test | what | measured (this machine: llvmpipe, SwiftShader in headless Chromium 141) |
|---|---|---|
| `tests/sdfview/test_shader_assembly.py` | Python and JS assemblers give identical programs; WGSL preprocessor rules | 85 programs identical |
| `tests/sdfview/test_hosts.py` | every web host: probe vs torch; images vs moderngl: surface; surface clipped by a cut plane; PAT cut plane + points | PAT max\|Δφ\| 4.8e-7, max\|Δ∇φ\| 6.1e-5; grid 2.7e-7; SuperFit (WebGL) 1.7e-4; images: pixels with \|Δ\| > 8/255: surface 0, clip 0.0018, cut + points 0.0051; with a ±1 px tolerance 0, 0.0011, 0.0022 |
| `tests/sdfview/test_slangpy_host.py` | slangpy: probe vs torch; images vs moderngl (also a translucent overlay) | PAT 4.2e-7 / 6.1e-5; grid 3.6e-7; images max\|Δ\| ≤ 1/255 |
| `tests/sdfview/test_gl.py` | moderngl host | |

The image residuals: pixels on edges one pixel apart (box faces, sprite borders), 1-2 px sprites covered by one
rasterizer and not the other, overlapping sprites at equal depth, and the zero contour's width (GLSL allows coarse or
fine derivatives for `fwidth`). `test_hosts.py` needs node and Playwright (Chromium) and fetches three.js with
`npm pack`; `SDFX_HOSTS=webgpu,...` restricts the hosts while developing.
