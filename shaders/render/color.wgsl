// Trippinator — Color Pass
//
// Palette maps the HDR feedback result into the song's own colours, tone maps, and
// dithers to kill banding on the panel's gradients.
//
// The palette is built from `palette_hue` and `palette_spread`, both derived on the CPU
// from slow spectral character and then held still. What this replaced was a fixed
// warm/mid/cool triple passed through a continuous hue rotation, which is a filter rather
// than a palette: rotating every pixel by the same angle changes what colour the image is
// without changing how it is coloured, and it made the colour a property of when you were
// watching rather than of what was playing.

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
    palette_hue: f32,
    grandness: f32,

    inject_gain: f32,
    exposure: f32,
    organic: f32,
    core_radius: f32,

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
    chroma: f32,

    palette_spread: f32,
    palette_sat: f32,
    orb_radius: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var hdr_frame: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// Fully saturated hue wheel, then desaturated toward white by `s`. Kept as HSV rather
// than as a rotation of a fixed triple because the palette needs an absolute hue the song
// can name, not an offset from an arbitrary starting colour.
fn hsv(h: f32, s: f32) -> vec3<f32> {
    let p = abs(fract(vec3<f32>(h) + vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
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

    // Two anchors a `palette_spread` apart on the hue circle: the song's primary colour
    // and its partner. Spectral balance chooses between them, so a bassy passage and a
    // trebly one still read as different moods — but as two colours *of this song* rather
    // than as two points on a global wheel. See DESIGN.md "Spectral balance" for why that
    // differentiation is needed, and "Colour comes from the song, not the clock" for why
    // it is now scoped to the track.
    let low = hsv(u.palette_hue, u.palette_sat);
    let high = hsv(fract(u.palette_hue + u.palette_spread), u.palette_sat * 0.85);

    // Bass sits at the primary hue, treble at the partner, mids between.
    let tilt = clamp(u.treble_w + 0.5 * u.mid_w, 0.0, 1.0);
    let anchor = mix(low, high, tilt);

    color = mix(color, color * anchor * 1.6, 0.55);

    // Both anchors are non-negative, so unlike the luma-axis rotation this replaced there
    // is nothing here that can push a channel below zero and have the tonemapper clamp it
    // to a dark core.
    color = aces_tonemap(color * u.exposure);
    color += vec3<f32>(dither(uv));

    return vec4<f32>(color, 1.0);
}
