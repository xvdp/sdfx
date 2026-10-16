// camera.slang — primary rays and depth (mirror of sdfview/shaders/glsl/camera.glsl).
//   ray through a pixel: unproject the NDC points (x, y, -1) and (x, y, 1) with (P V)⁻¹
//   depth of a world point: z_ndc = (P V p).z / (P V p).w, window depth = ½ z_ndc + ½ (default depth range [0, 1])
// Matrices are row-major as written in mathematics (slangpy: spy.float4x4(numpy (4, 4))), applied as mul(M, v); the
// GLSL receives the same matrices column-major and writes M * v.
module camera;

public struct Camera
{
    public float4x4 view_proj;      // P V        (GLSL: u_view_proj)
    public float4x4 inv_view_proj;  // (P V)⁻¹    (GLSL: u_inv_view_proj)
    public float4 viewport;         // x, y, width, height in pixels (GLSL: u_viewport)
}

// frag_xy: window coordinates of the pixel center, origin bottom left (gl_FragCoord.xy)
public void cameraRay(Camera cam, float2 frag_xy, out float3 ro, out float3 rd)
{
    float2 ndc = 2.0 * (frag_xy - cam.viewport.xy) / cam.viewport.zw - 1.0;
    float4 near = mul(cam.inv_view_proj, float4(ndc, -1.0, 1.0));
    float4 far = mul(cam.inv_view_proj, float4(ndc, 1.0, 1.0));
    ro = near.xyz / near.w;
    rd = normalize(far.xyz / far.w - ro);
}

public float fragDepth(Camera cam, float3 p)
{
    float4 clip = mul(cam.view_proj, float4(p, 1.0));
    return clamp(0.5 * clip.z / clip.w + 0.5, 0.0, 1.0);
}

// slab test; t_near <= t_far when the ray (line) meets the box
public bool intersectAABB(float3 ro, float3 rd, float3 aabb_min, float3 aabb_max, out float t_near, out float t_far)
{
    float3 inv_dir = 1.0 / rd;
    float3 t0 = (aabb_min - ro) * inv_dir;
    float3 t1 = (aabb_max - ro) * inv_dir;
    float3 t_min = min(t0, t1);
    float3 t_max = max(t0, t1);
    t_near = max(max(t_min.x, t_min.y), t_min.z);
    t_far = min(min(t_max.x, t_max.y), t_max.z);
    return t_near <= t_far && t_far >= 0.0;
}
