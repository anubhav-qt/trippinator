// Trippinator — Compositor Fragment Shader
// Final compositing, tonemapping, and color grading.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct CompositorParams {
    time: f32,
    exposure: f32,
    gamma: f32,
    saturation: f32,
    vignette_strength: f32,
    _padding: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(0) @binding(0)
var<uniform> params: CompositorParams;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var input_sampler: sampler;

// ACES filmic tonemapping
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    var color = textureSample(input_texture, input_sampler, uv).rgb;

    // Exposure
    color *= params.exposure;

    // Tonemapping
    color = aces_tonemap(color);

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / params.gamma));

    // Saturation adjustment
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(vec3<f32>(luminance), color, params.saturation);

    // Vignette
    let dist = distance(uv, vec2<f32>(0.5));
    let vignette = 1.0 - smoothstep(0.3, 0.9, dist) * params.vignette_strength;
    color *= vignette;

    return vec4<f32>(color, 1.0);
}
