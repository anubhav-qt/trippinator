// Trippinator — Feedback Fragment Shader
// Samples the previous frame with warp/distortion, blends with current.
//
// I_t = (1 - λ) * P_t + λ * W(I_{t-1})

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FeedbackParams {
    time: f32,
    feedback_strength: f32,
    warp_amount: f32,
    warp_rotation: f32,
    warp_zoom: f32,
    color_decay: f32,
    energy: f32,
    temporal_memory: f32,
};

@group(0) @binding(0)
var<uniform> params: FeedbackParams;

@group(0) @binding(1)
var prev_frame: texture_2d<f32>;

@group(0) @binding(2)
var prev_sampler: sampler;

@group(0) @binding(3)
var current_frame: texture_2d<f32>;

@group(0) @binding(4)
var current_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let center = vec2<f32>(0.5, 0.5);

    // Warp the UV for previous frame sampling
    var warped_uv = uv - center;

    // Apply zoom
    warped_uv *= 1.0 / params.warp_zoom;

    // Apply rotation
    let ca = cos(params.warp_rotation);
    let sa = sin(params.warp_rotation);
    warped_uv = vec2<f32>(
        warped_uv.x * ca - warped_uv.y * sa,
        warped_uv.x * sa + warped_uv.y * ca
    );

    // Additional warp based on position
    let warp_noise = sin(uv.x * 6.28 + params.time) * cos(uv.y * 6.28 + params.time * 0.7);
    warped_uv += vec2<f32>(warp_noise, -warp_noise) * params.warp_amount;

    warped_uv += center;

    // Sample previous frame with warp
    let prev_color = textureSample(prev_frame, prev_sampler, warped_uv);

    // Sample current procedural frame
    let curr_color = textureSample(current_frame, current_sampler, uv);

    // Blend with feedback
    let lambda = params.feedback_strength;
    var result = (1.0 - lambda) * curr_color + lambda * prev_color;

    // Apply color decay
    result = result * vec4<f32>(vec3<f32>(params.color_decay), 1.0);

    return result;
}
