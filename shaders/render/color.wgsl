// Trippinator — Color Pass
//
// Palette maps the HDR feedback result by spectral balance, applies a slow always-on
// hue drift, tone maps, and dithers to kill banding on the portrait panel's gradients.

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    dt: f32,

    band_energy: array<vec4<f32>, 2>,

    rms: f32,
    energy_velocity: f32,
    attack: f32,
    decay: f32,

    bass_w: f32,
    mid_w: f32,
    treble_w: f32,
    spectral_centroid: f32,

    symmetry: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,

    warp_radial: f32,
    feedback_decay: f32,
    hue_shift: f32,
    grandness: f32,

    inject_gain: f32,
    exposure: f32,
    organic: f32,
    core_scale: f32,

    warp_rotate: f32,
    warp_spiral: f32,
    warp_shear: f32,
    warp_turb: f32,

    level_norm: f32,
    bg_ambient: f32,
    bg_gate_lo: f32,
    bg_gate_hi: f32,

    profile_drift: f32,
    profile_pulse: f32,
    vacuum: f32,
    vacuum_radius: f32,

    ripple_age: f32,
    ripple_amp: f32,
    bg_outflow: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var hdr_frame: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn hue_rotate(color: vec3<f32>, turns: f32) -> vec3<f32> {
    let angle = turns * 6.2831853;
    let k = vec3<f32>(0.57735, 0.57735, 0.57735); // rotate around the luma axis
    let cos_a = cos(angle);
    return color * cos_a + cross(k, color) * sin(angle) + k * dot(k, color) * (1.0 - cos_a);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn dither(uv: vec2<f32>) -> f32 {
    return (fract(sin(dot(uv * u.resolution, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var color = textureSample(hdr_frame, samp, uv).rgb;

    // Spectral-balance palette anchor: warm for bass, mid-tone for mids, cool for treble.
    // See DESIGN.md "Spectral balance" — this is what makes a bassy track and a trebly
    // one read as different moods of the same visual, not just different frame content.
    let warm = vec3<f32>(1.0, 0.35, 0.15);
    let mid = vec3<f32>(0.4, 0.9, 0.3);
    let cool = vec3<f32>(0.2, 0.55, 1.0);
    let anchor = warm * u.bass_w + mid * u.mid_w + cool * u.treble_w;

    color = mix(color, color * anchor * 1.6, 0.55);

    // Hue rotation is a rotation about the luma axis: it preserves length but can push
    // individual channels negative, which the tonemapper then clamps to black. That is
    // what produced the dark core on saturated frames, so clamp before tone mapping.
    color = max(hue_rotate(color, u.hue_shift), vec3<f32>(0.0));

    color = aces_tonemap(color * u.exposure);
    color += vec3<f32>(dither(uv));

    return vec4<f32>(color, 1.0);
}
