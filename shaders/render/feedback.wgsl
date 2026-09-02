// Trippinator — Feedback Pass
//
// Samples the previous frame through a kaleidoscope warp, decays it, and adds new
// audio-driven injection. Combined into one pass because injection needs the same warped
// coordinate space as the feedback sample — see crates/render/src/lib.rs module docs.
//
// All four injection layers render simultaneously. Each is kept exactly as it behaved
// when it was a selectable mode — same shapes, tints, radii, and band bindings — so the
// merged image is a true superposition, not a re-tune.
//
// Two invariants this file must preserve:
//  1. Injection is energy-per-SECOND and is multiplied by `dt`. Adding a per-frame
//     constant makes brightness frame-rate dependent and saturates the loop to white.
//  2. The warp applies a per-frame *step*, never a function of absolute `time` that
//     compounds. The feedback texture already carries the accumulated history.

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    dt: f32,

    band_energy: array<vec4<f32>, 2>, // 7 bands used, packed as 2x vec4

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

    zoom: f32,
    feedback_decay: f32,
    hue_shift: f32,
    grandness: f32,

    inject_gain: f32,
    exposure: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

const TAU: f32 = 6.2831853;
const PI: f32 = 3.14159265;

fn band(i: u32) -> f32 {
    if (i < 4u) { return u.band_energy[0][i]; }
    return u.band_energy[1][i - 4u];
}

// Soft radial falloff: 1.0 at the center, 0.0 beyond `radius`.
// Written this way because smoothstep(hi, lo, x) with reversed edges is undefined in WGSL.
fn falloff(d: f32, radius: f32) -> f32 {
    return 1.0 - smoothstep(0.0, max(radius, 1e-4), d);
}

// Kaleidoscope: fold the sample coordinate into a wedge of order `symmetry`, mirrored.
fn warp_coord(p: vec2<f32>) -> vec2<f32> {
    // Normally a slow inward drift; on a genuinely grand moment this drops below 1.0
    // and the whole image blooms outward. Driven by the audio's own baseline, never by
    // a periodic function of time.
    let grand_zoom = 1.0 - 0.045 * u.grandness;
    let z = u.zoom * grand_zoom;

    let wedge = TAU / f32(u.symmetry);
    // atan2 returns -PI..PI; shift to 0..TAU before folding so the wedge is even.
    var a = (atan2(p.y, p.x) + PI) % wedge;
    a = abs(a - wedge * 0.5);
    return vec2<f32>(cos(a), sin(a)) * length(p) * z;
}

// ---------------------------------------------------------------------------
// Injection layers. Each returns an energy-per-second rate; they are summed.
// ---------------------------------------------------------------------------

// Four concentric wave rings sitting outside the central orb, separated from it by a
// gap. Each ring is a closed wave function r(theta) = base + amp*sin(lobes*theta + phase),
// bound to a different band so they articulate independently — different lobe count,
// different drift rate, different tint. They never move in lockstep, which is what stops
// them reading as four copies of one animation.
fn wave_rings(p: vec2<f32>, orb_radius: f32) -> vec3<f32> {
    let r = length(p);
    let a = atan2(p.y, p.x);
    var acc = vec3<f32>(0.0);

    // bass, mid, high-mid, brilliance — four distinct voices across the spectrum.
    var band_of_ring = array<u32, 4>(1u, 3u, 4u, 6u);
    let gap = 0.16;

    for (var i = 0u; i < 4u; i = i + 1u) {
        let fi = f32(i);
        let e = band(band_of_ring[i]);

        let base_r = orb_radius + gap + 0.17 * fi;
        // Starts at 5, not 3: a 3-lobe ring is literally a rounded triangle, and on
        // loud bass its amplitude pushes it far enough out to read as a stray outline
        // rather than as ring texture. Low-order lobes look like shapes; higher ones
        // look like surface.
        let lobes = 5.0 + fi * 2.0;
        let phase = u.time * (0.13 + 0.07 * fi) + fi * 1.7;
        let ring_r = base_r + sin(a * lobes + phase) * (0.015 + 0.055 * e);

        let tint = mix(vec3<f32>(1.0, 0.5, 0.2), vec3<f32>(0.35, 0.75, 1.0), fi / 3.0);
        acc += tint * falloff(abs(r - ring_r), 0.012 + 0.010 * e) * (0.12 + e * 1.1);
    }
    return acc;
}

// Soft central orb, sized by bass + rms, ringed by the four wave rings.
fn layer_orb(p: vec2<f32>) -> vec3<f32> {
    let radius = 0.10 + 0.30 * u.bass_w * (0.4 + u.rms * 2.0);
    let orb = vec3<f32>(1.0, 0.55, 0.25) * falloff(length(p), radius) * (0.6 + u.attack * 0.8);
    return orb + wave_rings(p, radius);
}

// Seven arcs around a fixed ring, one per band.
fn layer_spectral_ring(p: vec2<f32>) -> vec3<f32> {
    let a = atan2(p.y, p.x) + PI;
    let idx = u32((a / TAU) * 7.0) % 7u;
    return vec3<f32>(0.3, 0.7, 1.0) * falloff(abs(length(p) - 0.5), 0.06) * band(idx) * 1.1;
}

// A handful of slowly orbiting emitter points.
fn layer_point_emitters(p: vec2<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let fi = f32(i);
        let ang = u.time * (0.06 + 0.015 * fi) + fi * 1.7;
        let center = vec2<f32>(cos(ang), sin(ang)) * (0.28 + 0.11 * fi);
        acc += vec3<f32>(0.6, 0.3, 1.0)
             * falloff(length(p - center), 0.07)
             * (0.25 + band(2u + i));
    }
    return acc * 0.8;
}

// One node per band on a slow Lissajous path. Low bands are broad warm pools, high bands
// tight cool points, so the spectrum reads as spatial structure rather than a level meter.
fn layer_constellation(p: vec2<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let t = u.time * 0.05;
    for (var i = 0u; i < 7u; i = i + 1u) {
        let fi = f32(i);
        let pos = vec2<f32>(
            sin(t * (0.7 + fi * 0.13) + fi * 2.1) * 0.62,
            cos(t * (0.5 + fi * 0.11) + fi * 1.3) * 0.80
        );
        let tint = mix(vec3<f32>(1.0, 0.4, 0.2), vec3<f32>(0.4, 0.8, 1.0), fi / 6.0);
        acc += tint * falloff(length(p - pos), 0.16 - 0.017 * fi) * band(i);
    }
    return acc * 0.9;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / u.resolution.y;
    var p = (uv - vec2<f32>(0.5)) * 2.0;
    p.x *= aspect;

    let warped = warp_coord(p);
    var sample_uv = warped / vec2<f32>(aspect, 1.0) * 0.5 + vec2<f32>(0.5);
    sample_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let decayed = textureSample(prev_frame, samp, sample_uv).rgb * u.feedback_decay;

    let injected = layer_orb(p)
                 + layer_spectral_ring(p)
                 + layer_point_emitters(p)
                 + layer_constellation(p);

    return vec4<f32>(decayed + injected * u.inject_gain * u.dt, 1.0);
}
