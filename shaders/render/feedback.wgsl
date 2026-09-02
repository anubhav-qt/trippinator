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
// Those four make up the central mandala, drawn in a frame shrunk by `u.core_radius`.
// Around it sits `layer_background`: a quiet ambient field that always tracks the music,
// plus a grand tier gated on `grandness` that is absent the rest of the time. The mandala
// says what is playing; the background says how it feels, and when something matters.
//
// Three invariants this file must preserve:
//  1. Injection is energy-per-SECOND and is multiplied by `dt`. Adding a per-frame
//     constant makes brightness frame-rate dependent and saturates the loop to white.
//  2. The warp applies a per-frame *step*, never a function of absolute `time` that
//     compounds. The feedback texture already carries the accumulated history.
//  3. Every function of the polar angle must be periodic with period TAU, and the warp
//     must be continuous in screen space. `atan2` has a branch cut at +/-PI, so
//     `sin(a * lobes)` with a fractional `lobes`, `fbm2(vec2(a, ...))`, or any blend
//     against a raw angle all jump across that cut. A one-pixel jump would be invisible
//     in a single frame, but this is a feedback loop: the discontinuity is written into
//     the texture, re-warped, and re-written, so within a second it is a permanent hard
//     seam across the image. Fold and blend *positions* (`vec2(cos a, sin a)`, which is
//     periodic by construction), crossfade between *integer* harmonics, and sample noise
//     on the unit circle via `ang_noise` — never on the angle itself.
//
// `u.organic` scales every departure from strict geometry: the imperfect fold, the
// breathing mirror order, the flow-field warp, ring rotation and shape morphing, orb
// churn and dispersion, blob eccentricity. It arrives already mapped into a hand-tuned
// safe range (0.10..0.70) by a song-feel function on the Rust side — see
// `ORGANIC_MIN`/`song_feel` in crates/render/src/lib.rs. At 0 this file would reduce to
// the strict-geometry version it grew out of.

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
@group(0) @binding(1) var prev_frame: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

const TAU: f32 = 6.2831853;
const PI: f32 = 3.14159265;

// How fast a ripple wavefront travels outward, in frame units per second. At this rate a
// wave launched at the centre reaches the end of a 16:9 frame in about two seconds.
const RIPPLE_SPEED: f32 = 0.52;

// NOTE ON UNITS: `p` is aspect-corrected, so the frame is always +/-1.0 tall and
// +/-aspect wide. On a 1080x1920 portrait panel that is +/-0.5625 wide, with the corners
// at r = 1.147; on a 1920x1080 landscape one it is +/-1.778 wide, corners at r = 2.04.
// Radii written as if the frame were +/-1 in both axes — which is the natural mistake —
// land off the side of a portrait screen entirely.
//
// NOTE ON ORIENTATION: this shader serves portrait and landscape from one code path, and
// the aspect correction above is the reason it cannot do that by simply being rotated 90
// degrees. The correction normalizes the *height* to 1 either way, so rotating the panel
// does not swap the axes' roles — it changes which axis is the short one AND changes the
// scale of everything measured against it. So nothing here may assume the tall axis is y.
// Anything shaped to the panel is written with `oriented()` and `long_half()` below, in
// terms of "along the long axis" and "across it", and works out either way.

fn band(i: u32) -> f32 {
    if (i < 4u) { return u.band_energy[0][i]; }
    return u.band_energy[1][i - 4u];
}

// Soft radial falloff: 1.0 at the center, 0.0 beyond `radius`.
// Written this way because smoothstep(hi, lo, x) with reversed edges is undefined in WGSL.
fn falloff(d: f32, radius: f32) -> f32 {
    return 1.0 - smoothstep(0.0, max(radius, 1e-4), d);
}

// Half-extent of the frame's long axis: 1.0 on a portrait 16:9 panel, 1.778 on a
// landscape one. The short axis's half-extent is the other way round, and is not needed
// here — `core_radius` and `orb_radius` arrive already scaled against it by the CPU,
// which is also where `core_frac` is interpreted.
fn long_half() -> f32 {
    return max(u.resolution.x, u.resolution.y) / u.resolution.y;
}

// Build a vector from a component along the frame's long axis and one across it. This is
// what keeps the panel-shaped constants orientation-agnostic: `oriented(1.15, 0.55)` is
// "spread wide the long way, narrow the short way" on either orientation.
fn oriented(along: f32, across: f32) -> vec2<f32> {
    if (u.resolution.x > u.resolution.y) {
        return vec2<f32>(along, across);
    }
    return vec2<f32>(across, along);
}

fn rot(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, s, -s, c);
}

// ---------------------------------------------------------------------------
// Cheap value noise. Used only for *slow, low-amplitude* modulation — never as
// per-pixel texture, which would read as static rather than as motion.
// ---------------------------------------------------------------------------

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// The second octave is rotated as well as scaled. Value noise is built on an
// axis-aligned lattice, so stacking octaves on the same grid lines their features up and
// the sum shows a visible crosshatch — which the feedback loop then amplifies into a
// standing grid. An irrational-ish rotation between octaves breaks that up.
fn fbm2(p: vec2<f32>) -> f32 {
    let r = mat2x2<f32>(0.80, 0.60, -0.60, 0.80);
    return vnoise(p) * 0.65 + vnoise(r * p * 2.03 + vec2<f32>(11.7, 3.1)) * 0.35;
}

// Signed 2D flow field in roughly -1..1.
fn flow(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        fbm2(p) - 0.5,
        fbm2(p + vec2<f32>(5.2, 1.3)) - 0.5
    ) * 2.0;
}

// Noise around a circle of radius `k`, so it is periodic in the angle by construction.
// Sampling `fbm2` on the raw angle instead wraps discontinuously at the atan2 branch cut
// and bakes a permanent radial seam into the feedback — see invariant 3.
fn ang_noise(a: f32, k: f32, t: f32) -> f32 {
    return fbm2(vec2<f32>(cos(a), sin(a)) * k + vec2<f32>(t, -t * 0.7));
}

// ---------------------------------------------------------------------------
// Warp
// ---------------------------------------------------------------------------

// Mirror fold at one *integer* order. Only integer orders are continuous: the fold is
// its own mirror image at every wedge boundary, but it closes on itself at the atan2
// branch cut only when `order` divides TAU exactly. A fractional order leaves one short
// wedge, and in a feedback loop that hairline becomes a permanent fault across the image.
// To vary the order, crossfade two of these — see `warp_coord`.
fn fold_pos(q: vec2<f32>, order: f32, frame: f32) -> vec2<f32> {
    let wedge = TAU / order;
    // atan2 returns -PI..PI; shift to 0..TAU before folding so the wedge is even.
    var a = (atan2(q.y, q.x) + PI + frame) % wedge;
    a = abs(a - wedge * 0.5);
    return vec2<f32>(cos(a), sin(a)) * length(q);
}

// Kaleidoscope with the rigidity taken out of it, and with the *kind* of motion coming
// from the music rather than from constants in this file.
//
// Every rate below arrives as a uniform in units per SECOND and is multiplied by `u.dt`.
// Two things follow from that, and both are the point:
//
//  - The motion is frame-rate independent. The six constants these replaced were applied
//    per frame, so the whole image moved at a speed set by how fast the GPU happened to
//    be running.
//  - The character of the warp is a per-track variable rather than a fixed recipe. It
//    used to be one motion — flow bend, one-signed swirl, always inward — so every track
//    on every run got the same inward spiral and only the amplitude changed. Now the
//    archetype sets the sign and balance of six independent components (see
//    `ProfileParams` in crates/render/src/lib.rs), never-repeating slow phases wander
//    them inside a bounded range, and the phases are entropy-seeded per run. Sustained
//    music blooms outward, turns, and flows; transient music tunnels inward and spirals.
//
// The rest of the fold is unchanged:
//  - the whole fold frame rotates slowly, so the wedge seams are never parked;
//  - the mirror order breathes: two adjacent integer folds are crossfaded, so the count
//    slides between orders continuously instead of only snapping on a keypress;
//  - the fold is only *mostly* applied, so the mirror is imperfect and the two halves
//    of a wedge drift out of exact agreement;
//  - a slow flow field, a differential rotation, and a slight tumbling anisotropy in the
//    zoom bend the trails, so they spiral and pool rather than running dead-straight
//    to the center.
//
// Everything here is still a per-frame step of the sample coordinate. Nothing integrates
// `time` into a growing offset — the feedback texture holds the history.
// Two sample coordinates for one pixel: the kaleidoscope fold that builds the core, and
// a plain outward drift for the field around it. They are returned together and blended
// in *colour* by `fs_main`, never here — see the note on `folded` below.
struct Warp {
    fold: vec2<f32>,
    drift: vec2<f32>,
};

fn warp_coord(p: vec2<f32>) -> Warp {
    let org = u.organic;

    // Log-zoom rate. Positive tunnels inward, negative blooms outward, and which one a
    // track gets is the archetype's decision. The grand-moment bloom and the transient
    // kick are already folded in on the CPU, where they are clamped against the limit at
    // which a sustained outward push would outrun injection and empty the frame.
    let z = exp(u.warp_radial * u.dt);

    var q = p;
    let r0 = length(q);

    // ONE GATE FOR EVERY DIRECTIONAL TERM.
    //
    // Enumerate what this function does to the coordinate and check each for symmetry
    // about the origin: the ripple and the vacuum are scalar multiplies, the uniform and
    // differential rotations are rotations about the origin, the tumbling anisotropy is an
    // ellipse centred on the origin. None of those can make one side of the orb brighter
    // than the other. Exactly two terms can:
    //
    //   * the flow bend, which is a local translation, and
    //   * the kaleidoscope fold, which aims every sample into a single angular wedge.
    //
    // Both are held at exactly zero across the orb and its surround and ramp in outside
    // it. Nothing else is gated, so the middle stays live rather than frozen.
    //
    // The flow bend's ramp used to be `smoothstep(0.0, 0.35, r0)` — sized against the
    // frame, written before the orb was a consideration. At 1.5x the orb radius that is
    // already at 20% strength, and a few percent of translation per frame is invisible in
    // one frame and a bright displaced arc after ninety-five of them. Gate against the orb,
    // not against the frame.
    // The gate has to clear the orb's LIGHT, not the orb's radius. Those are very
    // different distances: the orb is a compact emitter feeding a trail ninety-five frames
    // deep, so it lights the whole inner region well past its own edge. Starting the fold
    // at 1.8x the radius put the ramp straight into that halo, and a partial fold aims
    // what it touches into a wedge — which turns the orb's own light into a bright arc
    // sitting beside it, in the orb's own colour. That is what the gold crescent was.
    //
    // There is a real trade-off being made here and it is worth stating: a kaleidoscope
    // fold aims light into wedges, and the orb is the brightest light in the frame, so
    // "fold close to the centre" and "a clean isolated dot at the centre" cannot both be
    // had. The fold is pushed out to where the mandala's other layers live and the centre
    // is left alone.
    //
    // Floored against the mandala radius as well as the orb's, so it does not creep inward
    // whenever the orb shrinks on quiet material.
    let fold_in = max(u.orb_radius, 1e-3);
    let gate_lo = max(fold_in * 3.0, u.core_radius * 0.45);
    let dir_gate = smoothstep(gate_lo, gate_lo * 2.0, r0);

    // Flow-field bend. Amplitude is deliberately ~1% of the frame: at this scale it
    // reads as the image being made of moving fluid, and much larger smears the trails
    // into mush within a few frames of feedback.
    //
    // Ramped in from the origin, because the flow field's value *at* the origin is not
    // zero: without the ramp `q +=` translates the whole image by that constant every
    // frame. One frame of it is a fifth of a percent and invisible, but the trail is ~95
    // frames deep and each is drawn at a further offset, so it accumulates into a smear
    // roughly a fifth of the frame wide sitting beside the core — which reads as the
    // centre not being centred. The ramp makes the origin a fixed point of the warp.
    q += flow(q * 1.7 + vec2<f32>(u.time * 0.035, u.time * -0.028))
       * (u.warp_turb * u.dt)
       * dir_gate;

    // Ripple: a wave packet launched from the centre and travelling outward, applied as
    // a radial velocity so it compresses the space ahead of the front and rarefies it
    // behind. The front's radius is its age times its speed, so this is one wave moving
    // out rather than a standing pattern that happens to oscillate — a standing one reads
    // as the image wobbling, not as something having left the middle.
    //
    // Onsets launch small ones continuously, which is what makes the orb ripple; a
    // rupture launches a large one. Both fade with the wave's own age.
    let front = u.ripple_age * RIPPLE_SPEED;
    let dr = r0 - front;
    let packet = exp(-dr * dr * 70.0) * exp(-u.ripple_age * 1.1);
    q *= clamp(1.0 + u.ripple_amp * packet * sin(dr * 24.0) * u.dt, 0.55, 1.45);

    // Vacuum: the space itself recoiling away from a hole in the middle. Sampling from a
    // *smaller* radius than the pixel sits at means the content that was there is carried
    // outward — so the material around the hole is genuinely evacuated into a ring rather
    // than a black disc being drawn over a structure that is still present underneath.
    // `fs_main` does the other two halves of that: annihilating the history inside the
    // hole, and masking injection so nothing is drawn back into it.
    if (u.vacuum_radius > 1e-4) {
        // Contained deliberately. The recoil dies out by ~2.4x the hole radius, so what
        // reads is the middle being torn out and the material around it thrown clear —
        // not the whole frame being blown outward, which is a different and much less
        // interesting event.
        let push = smoothstep(0.0, u.vacuum_radius * 0.55, r0)
                 * (1.0 - smoothstep(u.vacuum_radius, u.vacuum_radius * 2.4, r0));
        q *= 1.0 - 2.2 * u.vacuum * push * u.dt;
    }

    // Uniform rotation. Zero for the transient-led archetype, which is why this did not
    // exist before; sustained material gets a slow overall turn, and dense material turns
    // the other way, so the two do not read as the same motion at different speeds.
    q = rot(u.warp_rotate * u.dt) * q;

    // Differential rotation: outer radii turn faster than inner ones, which is what
    // turns concentric trails into spiral arms. Signed, so it can wind either way.
    q = rot(u.warp_spiral * u.dt * (0.25 + r0)) * q;

    // Fold frame: slow rotation plus a wander, so seams travel instead of sitting.
    // Wrapped to TAU because it feeds a `%` against the wedge: left to accumulate for
    // minutes it costs mantissa bits, and the modulo then quantizes into visible steps.
    // TAU is a whole number of wedges at integer order, so wrapping changes nothing else.
    let frame = (u.time * 0.021 * org + fbm2(vec2<f32>(u.time * 0.03, 7.0)) * 0.9 * org) % TAU;

    let len = length(q);
    // The unfolded position, in the same rotated frame as the fold. Built from cos/sin
    // rather than carried as an angle so it stays continuous across the branch cut.
    let raw = vec2<f32>(cos(atan2(q.y, q.x) + PI + frame),
                        sin(atan2(q.y, q.x) + PI + frame)) * len;

    // Breathing mirror order: crossfade the two integer folds either side of `sym`.
    // Each is continuous, so the blend is too — which a fractional order would not be.
    let sym = max(2.0, f32(u.symmetry) + 0.45 * org * sin(u.time * 0.037));
    let n0 = floor(sym);
    let pf = mix(
        fold_pos(q, n0, frame),
        fold_pos(q, n0 + 1.0, frame),
        smoothstep(0.0, 1.0, fract(sym))
    );

    // Imperfect mirror: blending back toward the unfolded position leaves the symmetry
    // legible but stops the halves being exact copies. Floored well above 0.5 so the two
    // can never cancel — at exactly 0.5 an antipodal pair sums to zero and the warp
    // collapses a whole ray to the origin.
    let fold_amt = clamp(1.0 - 0.16 * org * (0.5 + 0.5 * sin(u.time * 0.043 + 1.2)), 0.62, 1.0);
    var folded = mix(raw, pf, fold_amt);
    // Blending two directions shortens the vector; restore the original length so the
    // blend only ever changes direction and never smuggles in an extra zoom.
    folded *= len / max(length(folded), 1e-5);

    // The polar frame is degenerate at the origin: atan2 is undefined at r = 0 and
    // violently sensitive just outside it, so the fold writes a pixel of garbage there
    // every frame and the outward zoom smears it into a permanent radial spoke. Fading
    // the warp back to the identity across the innermost few percent removes the source
    // rather than the symptom.
    //
    // Note what this deliberately does NOT do: blend the fold out again at large radii by
    // mixing the two *coordinates*. That mixes two very different mappings, and the blend
    // band between them reads as a hard faceted outline with the whole composition dragged
    // off centre. The fold is confined to the core by blending the two sampled *colours*
    // instead, which has no geometry to distort — see `fs_main`.
    //
    // The identity region is sized to the ORB, not to a fixed 5% of the frame. That
    // constant was the reason the core looked like a lit 3D ball rather than a flat dot:
    // the orb spans r = 0.045 to 0.11 here, so the fold was fully active across almost all
    // of it, and `fold_pos` aims every sample into a single wedge — meaning every pixel
    // inside the orb drew its history from one angular sector. That is a light coming from
    // one side, and because the fold frame rotates slowly, a light that slowly orbits.
    //
    // Inside the orb the warp is now only zoom and rotation about the centre, both of
    // which are radially symmetric, so a radially symmetric injection stays radially
    // symmetric however long it accumulates.
    // Same gate as the flow bend: a partially applied fold is a coordinate pulled toward
    // one direction by an amount that varies with radius, which across the orb's rim is a
    // brightness gradient down one side. No amount of making the injection symmetric fixes
    // that, because the asymmetry is in the sampling.
    folded = mix(q, folded, dir_gate);

    // Tumbling anisotropy in the zoom: the image breathes as a slowly turning ellipse
    // instead of a perfect circle. `warp_shear` adds a steady stretch along the same
    // turning axis on top of that wobble — a sustained pull in one direction, which is
    // what makes the drift archetype's motion read as weather rather than as a spin.
    let ecc = 0.02 * org * sin(u.time * 0.026) + u.warp_shear * u.dt;
    let aniso = vec2<f32>(1.0 + ecc, 1.0 - ecc);
    let axis = rot(u.time * 0.017 * org);

    // The drift coordinate keeps the flow bend, the swirl and the zoom but skips the fold
    // entirely, so the outer field has motion of its own and no mirror symmetry. Its
    // length is the pixel's own radius, so unlike the folded coordinate it never leaves
    // the frame and never has to be mirrored back in — which is what was stamping copies
    // of the core into the corners.
    //
    // On top of that it carries an always-outward rate of its own, so the background
    // streams outward and fades on the way rather than sitting where it was injected and
    // pulsing in place. It is applied here and not in `z` so the core is unaffected: the
    // mandala can still tunnel inward while the field around it flows out.
    // The zoom and the anisotropy are held at identity across the orb. Both are symmetric
    // about the origin, so neither was ever a source of the crescent — they are held for
    // two different reasons. The zoom, blooming outward, drains the centre faster than
    // injection refills it and hollows the orb into a dark disc with a bright rim. The
    // anisotropy is a 2% ellipse, invisible per frame, which over the depth of the trail
    // makes a round dot visibly oval.
    //
    // Held here rather than by blending the finished coordinate back toward `p`, which is
    // what this replaces: that also cancelled the ripple through the middle, and its own
    // transition band left a fraction of the flow bend alive across the orb's rim.
    let core_hold = 1.0 - smoothstep(fold_in * 1.25, fold_in * 2.20, r0);
    let z_held = mix(z, 1.0, core_hold);
    let aniso_held = mix(aniso, vec2<f32>(1.0), core_hold);
    let drift_z = z_held * exp(-u.bg_outflow * u.dt * (1.0 - core_hold));

    let fold_out = axis * ((transpose(axis) * folded) * aniso_held) * z_held;
    let drift_out = axis * ((transpose(axis) * q) * aniso_held) * drift_z;

    // Inside the orb the warp is the IDENTITY. Not "mostly radial", not "symmetric" —
    // the pixel samples exactly itself, so that region is its own injection decayed to
    // equilibrium and nothing else. That makes it a clean radial dot by construction.
    //
    // Making the fold symmetric there was not enough on its own. A symmetric warp is
    // still a warp: with the radial rate blooming outward, the centre drains faster than
    // injection refills it, so the orb came out as a dark hole with a bright rim — which
    // is what read as a glass bubble with a highlight on one side. Holding the coordinate
    // still removes the drain rather than compensating for it.
    //
    // Consequence worth knowing: a ripple now emanates from the orb's edge rather than
    // from the exact centre, because there is no motion inside the hold region to carry
    // it.
    return Warp(fold_out, drift_out);
}
// The four concentric "wave rings" that used to live here are gone deliberately.
//
// They were drawn relative to the orb, at its radius plus a gap, which made them a set
// of shells hugging it — and a shell around a bright centre reads as a bubble with the
// orb suspended inside it, which is not what the middle of this is meant to be. Closing
// the gap so they sat directly on the orb made it worse rather than better: the rings
// are angularly lobed, so at the orb's rim they shaded one side of it and turned a flat
// dot into a lit sphere.
//
// If something is wanted between the orb and the spectral ring, it must not be
// concentric with the orb and must not carry an angular term near its rim. Recover the
// old implementation from git history rather than rewriting it from this description.


fn layer_orb(p: vec2<f32>) -> vec3<f32> {
    // Sized off the normalized level rather than raw RMS, so the orb reaches the same
    // size on a quiet master as on a loud one.
    //
    // And sized by whichever voice this kind of music actually leads with, rather than by
    // the bass unconditionally. Keying the centrepiece to a band that a sustained,
    // guitar-led track barely occupies left the orb near its minimum radius for the whole
    // song, which is the third of the three reasons that material came out looking thin.
    // Computed on the CPU now, because the warp needs the same number — see the fold
    // identity region in `warp_coord`. Converted back into the mandala's own space, which
    // is what this function works in.
    let radius = u.orb_radius / max(u.core_radius, 1e-4);

    let r = length(p);

    // THE ORB IS AN EXPLICIT EXCEPTION TO THE `organic` PREFERENCE. Everywhere else in
    // this file, elements are deliberately given lobed outlines, independent rotation and
    // shape morphing so nothing reads as machined. The orb is not: it is asked to be a
    // flat, perfectly radial dot, brightest at the exact centre and dimming monotonically
    // to its boundary, and it is the one element where that is the requirement.
    //
    // The lobed outline that used to be here — three angular terms modulating `r_eff` —
    // is why. An angular radius makes `t` below a function of angle, so the body is
    // brighter on one side, and at the sizes the orb actually runs at that does not read
    // as an organic outline. It reads as a shaded sphere. Do not put it back.
    let r_eff = radius;

    // One tint across the whole body, and a brightness that is a function of RADIUS AND
    // NOTHING ELSE — no angular term in the radius, the brightness, the hue or the channel
    // weighting — so every pixel at the same distance from the centre is identical by
    // construction rather than by tuning.
    //
    // Two earlier attempts at "interest" inside the orb are gone for the same reason, and
    // must not come back: interior churn and per-channel limb dispersion both put
    // different colours in different places inside the body, and at any size where the orb
    // is small that reads as two dots side by side instead of as texture. Whatever variety
    // the orb has comes from the tint moving as a whole, never from one side differing
    // from the other.
    let tint = mix(vec3<f32>(1.00, 0.55, 0.25), vec3<f32>(0.45, 0.75, 1.00), u.treble_w);

    // Normalized radius, 0 at the centre and 1 at the boundary.
    let t = clamp(r / max(r_eff, 1e-3), 0.0, 1.0);
    let f = 1.0 - t;

    // Strictly decreasing in t: gentle through the middle, closing faster toward the rim
    // so the boundary stays legible instead of hazing out.
    let profile = f * f * (0.35 + 0.65 * f);
    // A tight hot centre on top, so the very middle is the brightest point in the image.
    let core = pow(f, 6.0);

    let orb = (tint * profile + vec3<f32>(1.00, 0.94, 0.88) * core * 0.90)
            * (0.55 + u.attack * 0.8);
    return orb;
}

// Seven arcs around a ring, one per band. This is the layer the kaleidoscope turns into
// the big background shapes, so it is where stationary geometry showed up worst: hard
// sector boundaries at fixed angles on a fixed-radius circle. Now the sector frame
// rotates, the boundaries crossfade instead of being quantized, the radius carries
// travelling harmonics plus a slow noise wobble, and the stroke thickness breathes — so
// those background shapes turn, deform, and hand energy between neighbours.
fn layer_spectral_ring(p: vec2<f32>) -> vec3<f32> {
    let org = u.organic;
    let spin = u.time * 0.033 * org;
    let a = atan2(p.y, p.x) + PI + spin;

    // Fractional sector position; blend between the two neighbouring bands rather than
    // snapping. Hard `u32` bucketing is what made these read as cut cardboard.
    let s = (a / TAU) * 7.0;
    let i0 = u32(floor(s)) % 7u;
    let i1 = (i0 + 1u) % 7u;
    let f = fract(s);
    let e = mix(band(i0), band(i1), smoothstep(0.35, 0.65, f));

    // Travelling deformation of the ring radius: two counter-rotating harmonics plus a
    // slow noise term, so the outline is never the same circle twice.
    let wob = sin(a * 3.0 - u.time * 0.19) * 0.030
            + sin(a * 5.0 + u.time * 0.11) * 0.018
            + (ang_noise(a, 1.3, u.time * 0.07) - 0.5) * 0.055;
    let radius = 0.5 + wob * org + 0.02 * u.grandness;

    // Thickness breathes with the band, so arcs swell and thin along their length.
    let thick = 0.06 * (1.0 + 0.5 * org * sin(a * 2.0 + u.time * 0.09)) * (0.8 + 0.6 * e);

    return vec3<f32>(0.3, 0.7, 1.0) * falloff(abs(length(p) - radius), thick) * e * 1.1;
}

// A handful of slowly orbiting emitter points. Circular orbits at constant speed are the
// giveaway that these are parametric, so the orbits are now slowly tumbling ellipses with
// a drifting wander on top, and each point is an oriented smear rather than a dot.
fn layer_point_emitters(p: vec2<f32>) -> vec3<f32> {
    let org = u.organic;
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let fi = f32(i);
        let ang = u.time * (0.06 + 0.015 * fi) + fi * 1.7;
        let e = band(2u + i);

        let ecc = 1.0 + 0.35 * org * sin(u.time * 0.043 + fi * 1.1);
        let orbit = vec2<f32>(cos(ang) * ecc, sin(ang) / ecc) * (0.28 + 0.11 * fi);
        var center = rot(u.time * (0.02 + 0.011 * fi) * org) * orbit;
        center += flow(vec2<f32>(fi * 4.0, u.time * 0.06)) * 0.05 * org;

        // Oriented smear: squash the sampling distance along a slowly turning axis, so
        // the emitter is a comet-ish streak that rotates, not a symmetric dot.
        var d = rot(ang * 0.7 + fi) * (p - center);
        d.x /= 1.0 + 0.6 * org;

        acc += vec3<f32>(0.6, 0.3, 1.0) * falloff(length(d), 0.07) * (0.25 + e);
    }
    return acc * 0.8;
}

// One node per band on a slow Lissajous path. Low bands are broad warm pools, high bands
// tight cool points, so the spectrum reads as spatial structure rather than a level meter.
// The pools are now anisotropic and tumbling with a noise-frayed edge, which is what
// separates "a soft cloud" from "a radial gradient".
fn layer_constellation(p: vec2<f32>) -> vec3<f32> {
    let org = u.organic;
    var acc = vec3<f32>(0.0);
    let t = u.time * 0.05;
    for (var i = 0u; i < 7u; i = i + 1u) {
        let fi = f32(i);
        var pos = oriented(
            cos(t * (0.5 + fi * 0.11) + fi * 1.3) * 0.80,
            sin(t * (0.7 + fi * 0.13) + fi * 2.1) * 0.62
        );
        pos += flow(vec2<f32>(fi * 7.3, u.time * 0.04)) * 0.09 * org;

        // Tumbling ellipse, eccentricity drifting on its own slow cycle.
        let stretch = 1.0 + 0.55 * org * (0.5 + 0.5 * sin(u.time * 0.031 + fi * 0.9));
        var d = rot(u.time * (0.024 + 0.008 * fi) * org + fi) * (p - pos);
        d = vec2<f32>(d.x / stretch, d.y * stretch);

        // Ragged edge: perturb the measured distance rather than the falloff radius, so
        // the boundary frays without the blob changing size.
        let dist = length(d)
                 * (1.0 + 0.22 * org * (fbm2(d * 6.0 + vec2<f32>(u.time * 0.15, fi)) - 0.5));

        let tint = mix(vec3<f32>(1.0, 0.4, 0.2), vec3<f32>(0.4, 0.8, 1.0), fi / 6.0);
        acc += tint * falloff(dist, 0.16 - 0.017 * fi) * band(i);
    }
    return acc * 0.9;
}

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

// Everything outside the core mandala.
//
// The division of labour: the mandala runs continuously and visualizes whatever is
// playing, so it has to work on anything. The background is the opposite — near-silent
// through ordinary playing, opening up only on the passages the analysis calls grand.
// That is what makes it read as an event rather than as wallpaper, and it is why every
// effect here is multiplied by `gate` rather than merely modulated by it: at ordinary
// levels this layer contributes nothing at all.
//
// It runs in two tiers. An ambient tier is always on and tracks the track quietly, so an
// ordinary passage still has a moving field around it rather than black. A grand tier is
// multiplied by a `grandness` gate and is genuinely absent the rest of the time.
//
// Both tiers are configured by the song archetype rather than by constants here, because
// a single tuning could not serve both cases. Sustained, bass-light material — an
// atmospheric lead over a held chord — used to get almost no background at all: the
// ambient level keyed off absolute RMS, so a dynamic older master never cleared it, and
// the two edge pools keyed off the bass and brilliance bands, which is exactly the part
// of the spectrum that kind of music does not occupy. It now takes a much higher ambient
// base, opens its grand tier at a lower gate, and reads its edge pools from the mid bands
// where its energy actually is.
//
// The grand tier is:
//  - curtains: large ridged folds of light, domain-warped so they drift like aurora;
//  - a swell: a broad wave whose radius is pushed outward by the moment itself rather
//    than by a timer, so it advances as the passage builds and falls back after;
//  - deep-field blooms: a few very large, very soft pools far out toward the corners;
//  - filaments: thin bright threads that only survive at the top of the gate.
fn layer_background(p: vec2<f32>) -> vec3<f32> {
    let r = length(p);

    // Opens just past the mandala's outer edge, as a multiple of the live core size so it
    // tracks whatever the core is set to instead of needing to be re-tuned alongside it.
    // The room this leaves is mostly along the long axis: at a large core the mandala
    // already spans the short axis, so the sides are legitimately full and the two ends of
    // the panel are where the background lives. Which ends those are depends on the
    // orientation, hence `oriented()` throughout below.
    let outside = smoothstep(u.core_radius * 0.82, u.core_radius * 1.02, r);
    if (outside <= 0.0) {
        return vec3<f32>(0.0);
    }

    let org = u.organic;
    let warm = vec3<f32>(1.00, 0.42, 0.18);
    let cool = vec3<f32>(0.25, 0.60, 1.00);
    let tint = mix(warm, cool, clamp(0.15 + u.treble_w * 0.9, 0.0, 1.0));
    let drift = vec2<f32>(u.time * 0.023, u.time * -0.017);

    var acc = vec3<f32>(0.0);

    // -----------------------------------------------------------------------
    // Ambient tier — always on, tracking the track. Quiet enough to read as the
    // room the mandala sits in rather than as a second subject competing with it,
    // but never nothing: an unremarkable passage should still have a field around
    // it that moves with the music.
    // -----------------------------------------------------------------------

    // The sampling frame is squashed across its short axis, which stretches the noise
    // features along the long one: on a 9:16 panel an isotropic field gives one or two
    // features across the width and leaves the ends as dead space, so the veil is shaped
    // to the frame rather than to a square.
    let vq = rot(u.time * 0.008) * (p * oriented(0.85, 1.7));
    let vn = fbm2(vq * 1.15 + flow(vq * 0.55 + drift) * (0.7 + 0.9 * org) + drift);
    let veil = pow(1.0 - abs(2.0 * vn - 1.0), 2.2);
    // NOTE ON THE COEFFICIENT: this term used to be `0.22 * u.rms`, and `rms` runs at
    // about 0.09 on typical material while `level_norm` runs at about 0.9. Swapping the
    // one for the other without rescaling multiplied this whole term by ten and made the
    // background roughly five times brighter than it had ever been tuned to be — which is
    // what filled the frame with a dominant wash of colour. The coefficient has to shrink
    // by the same factor the quantity grew by.
    acc += tint * veil * (u.bg_ambient + 0.04 * u.level_norm + 0.07 * u.mid_w);

    // Two broad pools anchored just off the two ends of the long axis — the parts of the
    // frame the mandala can never reach, whatever the core is set to. Split by band so the
    // two ends of the screen answer to different parts of the mix.
    // Which bands drive them follows the archetype: bass and brilliance for percussive
    // material, mid and high-mid for sustained material, where the lead lives.
    let low_voice = mix(band(1u), band(3u), u.profile_drift);
    let high_voice = mix(band(6u), band(4u), u.profile_drift);
    let end = long_half();
    acc += warm * falloff(length(p - oriented(end, 0.0)), 0.85) * (0.02 + 0.14 * low_voice);
    acc += cool * falloff(length(p - oriented(-end, 0.0)), 0.85) * (0.02 + 0.14 * high_voice);

    // -----------------------------------------------------------------------
    // Grand tier — the part that only shows up on a real moment. Multiplied by the
    // gate rather than modulated by it, so it is genuinely absent the rest of the time.
    // Calibrated against logged `grandness` on real tracks, where it spends most of its
    // time at 0.00-0.10 and peaks around 0.25-0.30 — not against the nominal 0..1.
    // -----------------------------------------------------------------------
    let gate = smoothstep(u.bg_gate_lo, u.bg_gate_hi, u.grandness);
    if (gate > 0.0) {
        var grand = vec3<f32>(0.0);

        // Curtains. The same ridged construction as the veil but raised to a much higher
        // power, so instead of a soft field it reads as thin bright folds of light.
        let cq = rot(u.time * 0.011) * (p * oriented(0.9, 1.5));
        let cn = fbm2(cq * 1.6 + flow(cq * 0.7 - drift) * (0.8 + 1.0 * org) + drift);
        grand += mix(warm, cool, 0.15 + u.treble_w * 0.8)
               * pow(1.0 - abs(2.0 * cn - 1.0), 5.0)
               * (0.50 + 0.85 * u.mid_w);

        // Swell. Radius is a function of the moment, not of time: it reaches further out
        // the grander the passage, then retreats with the envelope. Nothing periodic.
        let swell_r = 0.45 + 1.10 * u.grandness;
        grand += mix(cool, warm, u.bass_w)
               * falloff(abs(r - swell_r), 0.10 + 0.16 * u.grandness)
               * 0.80;

        // Deep-field blooms — very large, very soft, wandering on the flow field. Spread
        // wide along the long axis and narrow across it, to match the frame they fill.
        for (var i = 0u; i < 3u; i = i + 1u) {
            let fi = f32(i);
            let c = flow(vec2<f32>(fi * 12.7, u.time * 0.02)) * oriented(1.15, 0.55);
            grand += mix(warm, cool, fract(fi * 0.37 + 0.20))
                   * falloff(length(p - c), 0.55 + 0.15 * fi)
                   * (0.18 + band(i * 2u) * 0.45);
        }

        // Filaments. Gated hard on top of the main gate, so these appear only at the peak
        // of a big moment — the thing you see two or three times a track.
        // Placed relative to the archetype's own gate rather than at fixed thresholds,
        // so "the top of a big moment" means the same thing whatever opens the gate.
        let peak = smoothstep(u.bg_gate_hi * 1.15, u.bg_gate_hi * 1.90, u.grandness);
        if (peak > 0.0) {
            let fq = rot(-u.time * 0.013) * (p * oriented(0.9, 1.4));
            let fn_ = fbm2(fq * 2.6 + flow(fq * 1.1 - drift) * 1.4);
            grand += vec3<f32>(0.9, 0.85, 1.0)
                   * pow(1.0 - abs(2.0 * fn_ - 1.0), 22.0)
                   * peak * 1.6;
        }

        acc += grand * gate;
    }

    // Sustained material leans on the background harder — it is where a swell the
    // mandala alone cannot express actually lands. Kept to a moderate lift: at 0.45 this
    // compounded with the ambient term above and the background stopped being the room
    // the mandala sits in and became the subject.
    return acc * outside * (0.55 + 0.25 * u.profile_drift);
}

// Sample the previous frame with the three channels warped by very slightly different
// amounts — red from marginally further out, blue from marginally further in.
//
// A single frame of this is invisible; the point is that it sits inside the feedback loop,
// so the separation compounds over the ~95 frames of trail into iridescent fringing along
// every moving edge. That is the difference between colour as a global property of the
// image and colour as something happening *in* it, and it is why the amount wants to be
// far smaller than it looks like it should — a value that reads correctly for a single
// frame tears the image into three within a second.
//
// Scaling about the frame centre rather than displacing means the separation grows with
// radius, which is what a lens does and what keeps the middle clean.
fn sample_chroma(uvc: vec2<f32>) -> vec3<f32> {
    let c = vec2<f32>(0.5);
    let d = uvc - c;
    let lo = vec2<f32>(0.0);
    let hi = vec2<f32>(1.0);
    let r = textureSample(prev_frame, samp, clamp(c + d * (1.0 + u.chroma), lo, hi)).r;
    let g = textureSample(prev_frame, samp, uvc).g;
    let b = textureSample(prev_frame, samp, clamp(c + d * (1.0 - u.chroma), lo, hi)).b;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / u.resolution.y;
    var p = (uv - vec2<f32>(0.5)) * 2.0;
    p.x *= aspect;

    let w = warp_coord(p);
    let to_uv = vec2<f32>(aspect, 1.0);
    let fold_uv = (w.fold / to_uv) * 0.5 + vec2<f32>(0.5);
    let drift_uv = (w.drift / to_uv) * 0.5 + vec2<f32>(0.5);

    // The fold aims every sample along one direction, and that direction runs off the
    // texture well inside the frame, so a large part of the image samples outside every
    // frame. Three answers have been tried here and only the last
    // one holds up:
    //
    //  - clamping replicates the border pixel, and the loop sets that into hard
    //    rectangular streaks along the edges;
    //  - fading to zero denies the region any history at all, so it shows a single frame
    //    of injection and goes black;
    //  - mirroring is continuous in *value*, but its gradient reverses at the boundary.
    //    That kink is a crease in the sampling, and the loop stands it up into a straight
    //    line across the image wherever the folded coordinate crosses the texture edge —
    //    a horizontal one where it leaves the top or bottom.
    //
    // The drift coordinate is in frame by construction and is a legitimate continuation of
    // the image, so it is simply used instead wherever the folded one escapes. That is a
    // blend between two *sampled colours*, with no coordinate being bent, so there is no
    // crease and nothing for the loop to harden into an edge.
    let over = max(max(-fold_uv.x, fold_uv.x - 1.0),
                   max(-fold_uv.y, fold_uv.y - 1.0));
    let escaped = smoothstep(-0.06, 0.02, over);

    let fold_s = clamp(fold_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let drift_s = clamp(drift_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // How far outside the core this pixel is. It does two jobs: choosing the drift sample
    // over the folded one, and shortening the trail out there. With the outer field
    // accumulating at all, the core's own smeared light otherwise survives long enough to
    // fill the panel with a flat wash of one colour within a couple of seconds. Raising
    // the retention to a power shortens the half-life by that factor and stays frame-rate
    // independent, so the background reads as something passing through rather than as
    // paint building up.
    // Originally 2.5, chosen when the background had no outward motion of its own and
    // being cleared quickly was its only defence against accumulating into a flat wash.
    // The outflow now does part of that job, so this can be lower — but 1.15 was too far:
    // the outflow moves light outward, it does not remove it, and the light piled up at
    // the far end instead. This keeps most of the extra travel without the wash.
    let outer = smoothstep(u.core_radius * 0.85, u.core_radius * 1.30, length(p));
    let decay = pow(u.feedback_decay, 1.0 + 1.80 * outer);

    // Blend the two feedback samples by radius. Doing this in colour rather than in
    // coordinates is what keeps the kaleidoscope inside the core without a seam: the fold
    // builds the mandala, and beyond it the field simply flows, so there is exactly one
    // core on screen and the background is free to carry the feeling on its own terms.
    let folded_prev = sample_chroma(fold_s);
    let drift_prev = sample_chroma(drift_s);
    let decayed = mix(folded_prev, drift_prev, max(outer, escaped)) * decay;

    // The mandala is evaluated in a shrunken frame, so it occupies `core_scale` of the
    // radius it used to and everything beyond it belongs to the background.
    let core = p / max(u.core_radius, 0.05);

    var injected = layer_orb(core)
                 + layer_spectral_ring(core)
                 + layer_point_emitters(core)
                 + layer_constellation(p)
                 + layer_background(p);

    // The other two halves of the vacuum. The warp has already carried the surrounding
    // field outward; this annihilates whatever history is left inside the hole and masks
    // injection so nothing is drawn back into it. Painting black over the top instead
    // would leave the orb intact underneath and still feeding the loop — the point is
    // that for a second or so the middle of the image is not there at all.
    //
    // The rim is a fraction of the radius rather than a fixed width, so a small hole is
    // as sharply defined as a large one.
    // `vacuum_radius` is 0 on almost every frame, and a smoothstep whose two edges are
    // equal is undefined in WGSL — so floor it, exactly as `falloff` does. At the floor
    // the mask is 0 everywhere that matters and `u.vacuum` is 0 regardless.
    let vr = max(u.vacuum_radius, 1e-4);
    let void_mask = 1.0 - smoothstep(vr * 0.72, vr, length(p));
    let keep = 1.0 - void_mask * u.vacuum;
    injected *= keep;

    return vec4<f32>(decayed * keep + injected * u.inject_gain * u.dt, 1.0);
}
