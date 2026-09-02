//! # Feedback Render Pipeline
//!
//! Two GPU passes over a ping-ponged pair of HDR textures:
//!
//! 1. **feedback** — samples the previous frame through a warp (5 selectable variants),
//!    decays it, and adds new audio-driven injection (4 selectable variants).
//! 2. **color** — palette maps, tone maps, and dithers the result to the swapchain.
//!
//! Inject and warp are combined into one pass rather than three, because injection needs
//! the same domain-warped coordinate space as the feedback sample to look coherent — doing
//! them as separate render-to-texture passes would just add a texture round-trip with no
//! benefit. "One pipeline with swappable stages," not three data-dependent pipelines.
//!
//! Structural variety comes from two independent axes (see DESIGN.md):
//! - `warp_mode` / `inject_mode` / `symmetry` — discrete, key-switchable "kind" of look.
//! - spectral balance (bass/mid/treble weights) — continuous, always-on "genre" of look.

use bytemuck::{Pod, Zeroable};
use trippinator_analysis_audio::{AudioFeatures, SongCharacter};
use trippinator_renderer_gpu::{RenderTarget, UniformBuffer};
use wgpu::{Device, Queue, TextureFormat};

const FEEDBACK_SHADER_SRC: &str = concat!(
    include_str!("../../../shaders/common/fullscreen_quad.wgsl"),
    include_str!("../../../shaders/render/feedback.wgsl"),
);
const COLOR_SHADER_SRC: &str = concat!(
    include_str!("../../../shaders/common/fullscreen_quad.wgsl"),
    include_str!("../../../shaders/render/color.wgsl"),
);

const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Hand-switchable structural state (Phase 1: set by keyboard; Phase 2: driven by
/// Poincare-section crossings of the chaotic attractor).
///
/// The warp is always the kaleidoscope fold and all four injection layers render
/// together, so symmetry order is the only structural degree of freedom left here.
#[derive(Debug, Clone, Copy)]
pub struct StructuralState {
    /// Kaleidoscope fold order.
    pub symmetry: u32,
}
impl Default for StructuralState {
    fn default() -> Self {
        Self { symmetry: 9 }
    }
}

/// GPU-side uniform block. Every field here is either a per-frame reading or a
/// hand-tuned safe-range constant — see DESIGN.md's safety contract. No chaotic dynamics
/// yet; those land in Phase 2 as additional fields, not a replacement of this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    dt: f32,

    /// 7 bands used, 8th is padding for 16-byte alignment.
    band_energy: [f32; 8],

    rms: f32,
    energy_velocity: f32,
    attack: f32,
    decay: f32,

    /// Normalized spectral balance — see DESIGN.md "Spectral balance".
    bass_w: f32,
    mid_w: f32,
    treble_w: f32,
    spectral_centroid: f32,

    symmetry: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,

    // Hand-tuned safe-range constants (Phase 1: static; Phase 2: slow drift).
    /// Log-zoom rate in units per second. Positive pulls the image inward (tunnel),
    /// negative lets it bloom outward. See `WarpCharacter`.
    warp_radial: f32,
    /// Per-frame retention, derived from a half-life and `dt` — NOT a fixed constant.
    /// A fixed per-frame value makes brightness depend on frame rate and drives the
    /// loop to `inject / (1 - decay)`, which saturates the tonemapper to white.
    feedback_decay: f32,
    /// The song's hue, 0..1 around the colour circle. Derived from slow spectral
    /// character and then held still — see `palette_target_hue`.
    palette_hue: f32,
    /// 0..1 "this is a big moment" envelope, from energy vs. its own rolling baseline.
    grandness: f32,

    /// Injection is energy-per-second; the shader multiplies by `dt`.
    inject_gain: f32,
    exposure: f32,
    /// Scale on every departure from rigid geometry in the feedback shader — imperfect
    /// fold, breathing mirror order, flow-field warp, ring rotation and shape morphing,
    /// orb churn, blob eccentricity. Arrives already mapped into `ORGANIC_MIN..MAX`.
    organic: f32,
    /// Radius the central mandala occupies, in the shader's aspect-corrected units.
    /// Everything outside it is background. Derived from `core_frac` and the frame's
    /// short axis, so it means the same thing in portrait and landscape.
    core_radius: f32,

    // --- Warp character. Every one is a RATE, in units per second, and the shader
    // multiplies by `dt`. The values these replaced were per-frame constants, which made
    // the speed of every motion in the image a function of the frame rate.
    /// Uniform rotation, radians/second.
    warp_rotate: f32,
    /// Differential rotation, radians/second per unit radius — what turns concentric
    /// trails into spiral arms. Signed.
    warp_spiral: f32,
    /// Anisotropic stretch rate along a slowly turning axis, per second. Signed.
    warp_shear: f32,
    /// Flow-field advection amount, frame-widths per second.
    warp_turb: f32,

    /// Current level as a fraction of this track's own loudness ceiling (0..1).
    /// Everything that used to key off raw `rms` uses this instead, so a quiet master
    /// reaches the same visual range as a loud one.
    level_norm: f32,
    /// Base brightness of the always-on ambient background tier.
    bg_ambient: f32,
    /// Grandness at which the grand background tier starts to open.
    bg_gate_lo: f32,
    /// Grandness at which it is fully open.
    bg_gate_hi: f32,

    /// Blend weights of the two archetypes the shader cares about, 0..1.
    profile_drift: f32,
    profile_pulse: f32,
    /// Strength of the current vacuum event, 0..1, and the radius of the hole it has
    /// opened, in aspect-corrected screen units. Both are 0 almost all the time.
    vacuum: f32,
    vacuum_radius: f32,

    /// Seconds since the current ripple wavefront was launched from the centre, and the
    /// radial displacement rate at its crest. The front's radius is `age * speed`, so
    /// this is one travelling wave rather than a standing pattern.
    ripple_age: f32,
    ripple_amp: f32,
    /// Extra always-outward radial rate applied to the background's sample coordinate
    /// only, so the outer field streams outward and fades instead of pulsing in place.
    bg_outflow: f32,
    /// How far apart the three colour channels are warped, as a fraction of the sample
    /// coordinate. Tiny per frame; the loop compounds it into iridescent fringing.
    chroma: f32,

    /// Distance around the hue circle from the song's primary hue to its secondary one.
    /// Small is an analogous palette, ~0.5 is complementary.
    palette_spread: f32,
    palette_sat: f32,
    /// Radius of the central orb, in the same units as `core_radius`. Computed here
    /// rather than in the shader because the warp needs it too — the kaleidoscope has to
    /// know where the orb ends so it can start outside it.
    orb_radius: f32,
    _pad3: f32,
}

/// Hand-tuned safe range for `organic`, both ends found by sweeping the live keys
/// against real music: below 0.10 the image is rigidly geometric, above 0.70 the fold
/// softens enough that the structure stops holding together. The song-feel envelope
/// moves within this range and never outside it — see DESIGN.md's safety contract.
const ORGANIC_MIN: f32 = 0.10;
const ORGANIC_MAX: f32 = 0.70;

/// How fluid the current track feels, 0..1, from four slow quantities.
///
/// Everything here is chosen to describe the *material* rather than the moment: a steady
/// bass-led drone lands near 0 and stays geometric, a bright track with real dynamics
/// lands high and flows. Deliberately not built from `attack` or per-band energy — those
/// move on every beat, and a looseness that flickers at beat rate would read as the image
/// glitching rather than as the music having a character.
fn song_feel(audio: &AudioFeatures, grandness: f32, bass_w: f32, mid_w: f32, treble_w: f32) -> f32 {
    // Level churn over the last second or two: a drone sits near 0, a track with real
    // dynamic range sits high.
    let churn = audio.long_term.volatility.clamp(0.0, 1.0);

    // Timbral movement — how fast the spectral centre of mass is sliding. It is a raw
    // derivative, so it is scaled down hard and clamped before use.
    let movement = (audio.short_term.spectral_movement.abs() * 0.35).clamp(0.0, 1.0);

    // Spectral balance. Bass-dominant material wants to stay solid, bright material
    // wants to flow; `bass_w` is subtracted rather than merely unweighted so a heavy
    // low end actively pulls the image back toward geometry.
    let brightness = (mid_w * 0.5 + treble_w * 1.5 - bass_w * 0.3 + 0.25).clamp(0.0, 1.0);

    (churn * 0.40 + brightness * 0.35 + movement * 0.15 + grandness * 0.10).clamp(0.0, 1.0)
}

/// The song's own hue, 0..1 around the colour circle.
///
/// Deterministic in the music and nothing else. There is deliberately no entropy seed
/// here, unlike the warp: the same song should *move* differently every run and be
/// *coloured* the same every run. A colour that is randomized per run is not the song's
/// colour, it is the run's.
///
/// Brightness is the primary axis because it is the one mapping between sound and colour
/// that is not arbitrary — a dark, low-centroid mix reads as red and amber, a bright
/// airy one as cyan and violet, and listeners agree about that far more than they agree
/// about anything else in this space. The spectral tilt is a secondary term, there to
/// separate songs that happen to share a centroid.
fn palette_target_hue(c: &SongCharacter) -> f32 {
    let bright = smoothstep(0.03, 0.30, c.brightness);
    // 0.02 (deep red) through amber, green and cyan to 0.78 (violet).
    let base = 0.02 + 0.76 * bright;
    let tilt = (c.treble_w - c.bass_w) * 0.10;
    (base + tilt).rem_euclid(1.0)
}

/// Signed shortest distance from `a` to `b` around a circle of circumference 1.
fn circular_delta(a: f32, b: f32) -> f32 {
    let d = (b - a).rem_euclid(1.0);
    if d > 0.5 { d - 1.0 } else { d }
}

/// Same curve as WGSL/GLSL `smoothstep`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Frame-rate-independent exponential move toward a target.
fn ema(cur: f32, target: f32, tau: f32, dt: f32) -> f32 {
    cur + (1.0 - (-dt / tau.max(1e-4)).exp()) * (target - cur)
}

/// How much of each archetype this track is, summing to 1.
///
/// Not a mode switch. Everything downstream is a linear blend of the three parameter
/// sets by these weights, so a track that is half riff and half held chord gets a
/// configuration halfway between — and a track that changes character mid-song crosses
/// over smoothly rather than snapping.
#[derive(Debug, Clone, Copy)]
pub struct SongProfile {
    /// Transient-led. Drums, riffs, plucked strings — energy arrives in hits.
    pub pulse: f32,
    /// Sustain-led. Held tones, bowed and blown instruments, pads, a guitar lead over a
    /// long chord. Energy arrives as a swell and stays.
    pub drift: f32,
    /// Dense and busy: full spectrum, loud, lots of everything at once.
    pub swarm: f32,
}

impl Default for SongProfile {
    fn default() -> Self {
        Self { pulse: 0.34, drift: 0.33, swarm: 0.33 }
    }
}

impl SongProfile {
    /// Classify from the slow character features. Deliberately built only out of
    /// quantities with time constants in the tens of seconds — this decides which
    /// configuration the visual runs, and a classification that moved on a beat would
    /// read as the whole look glitching rather than as the music having a character.
    fn from_audio(audio: &AudioFeatures) -> Self {
        let c = &audio.character;
        let level = audio.long_term.level_norm;

        let pulse = c.percussive * (1.0 - 0.5 * c.sustained);
        // Bright, sustained material scores highest — a held guitar lead or a pad, as
        // opposed to a sustained bass drone, which should stay solid.
        let drift = c.sustained * (0.35 + 0.65 * (c.mid_w + c.treble_w)) * (1.0 - 0.4 * c.density);
        let swarm = c.density * (0.35 + 0.65 * c.percussive) * smoothstep(0.35, 0.75, level);

        // A floor on each keeps every blend weight differentiable and stops a silent or
        // ambiguous passage from dividing by ~0 and snapping to whichever score is
        // marginally largest.
        let (pulse, drift, swarm) = (pulse + 0.06, drift + 0.06, swarm + 0.06);
        let total = pulse + drift + swarm;
        Self { pulse: pulse / total, drift: drift / total, swarm: swarm / total }
    }

    fn lerp(&mut self, target: &SongProfile, tau: f32, dt: f32) {
        self.pulse = ema(self.pulse, target.pulse, tau, dt);
        self.drift = ema(self.drift, target.drift, tau, dt);
        self.swarm = ema(self.swarm, target.swarm, tau, dt);
    }

    /// Blend the three parameter sets by these weights.
    fn params(&self) -> ProfileParams {
        let mut out = ProfileParams::ZERO;
        for (w, p) in [
            (self.pulse, ProfileParams::PULSE),
            (self.drift, ProfileParams::DRIFT),
            (self.swarm, ProfileParams::SWARM),
        ] {
            out = out.add_scaled(&p, w);
        }
        out
    }
}

/// One archetype's configuration. Three of these are blended per frame.
///
/// This struct is the answer to "different kinds of songs need different settings":
/// rather than one global tuning that has to work for everything and therefore suits the
/// track it was tuned against, each archetype carries its own grandness detector, its own
/// trail length, and its own warp character.
#[derive(Debug, Clone, Copy)]
struct ProfileParams {
    /// How much each grandness path counts for this archetype.
    hit_weight: f32,
    swell_weight: f32,
    /// Hit path: energy against its own ~1.5 s context, as a ratio.
    hit_lo: f32,
    hit_hi: f32,
    /// Swell path: level against this track's own 60 s loudness ceiling.
    swell_lo: f32,
    swell_hi: f32,
    /// Envelope shape of the grandness reading itself.
    grand_attack: f32,
    grand_release: f32,

    /// Multipliers on the hand-tuned globals.
    trail_mul: f32,
    inject_mul: f32,

    /// Background: always-on ambient level, and where the grand tier opens.
    bg_ambient: f32,
    bg_gate_lo: f32,
    bg_gate_hi: f32,

    /// Warp character, all rates per second. See `Uniforms`.
    radial: f32,
    rotate: f32,
    spiral: f32,
    shear: f32,
    turb: f32,
    /// How hard a grand moment pushes the zoom outward, in log-units per second.
    grand_bloom: f32,
    /// How hard a transient pushes it outward.
    pulse_kick: f32,
}

impl ProfileParams {
    const ZERO: Self = Self {
        hit_weight: 0.0, swell_weight: 0.0, hit_lo: 0.0, hit_hi: 0.0,
        swell_lo: 0.0, swell_hi: 0.0, grand_attack: 0.0, grand_release: 0.0,
        trail_mul: 0.0, inject_mul: 0.0, bg_ambient: 0.0, bg_gate_lo: 0.0,
        bg_gate_hi: 0.0, radial: 0.0, rotate: 0.0, spiral: 0.0, shear: 0.0,
        turb: 0.0, grand_bloom: 0.0, pulse_kick: 0.0,
    };

    /// Transient-led music. This one reproduces the look that was already tuned by ear
    /// and liked — the warp rates are the old per-frame constants multiplied by the
    /// ~178 fps they were tuned at, so at that frame rate the image is unchanged and at
    /// any other frame rate it is now correct rather than merely different.
    const PULSE: Self = Self {
        hit_weight: 1.0,
        swell_weight: 0.35,
        hit_lo: 1.5,
        hit_hi: 2.8,
        swell_lo: 0.72,
        swell_hi: 0.95,
        grand_attack: 0.4,
        grand_release: 2.5,
        trail_mul: 1.0,
        inject_mul: 1.0,
        bg_ambient: 0.035,
        bg_gate_lo: 0.04,
        bg_gate_hi: 0.18,
        radial: 1.42,
        rotate: 0.0,
        spiral: 1.78,
        shear: 0.0,
        turb: 1.96,
        // The old warp blew the zoom outward by `1 - 0.045 * grandness` per frame, which
        // at 178 fps is this rate. It was safe only because grandness never got far off
        // the floor; now that a sustained passage can hold it near the top, the clamp on
        // the combined rate is what keeps it safe rather than the detector's timidity.
        grand_bloom: 8.0,
        pulse_kick: 0.9,
    };

    /// Sustain-led music — the case that was failing. Everything here is the opposite of
    /// the percussive tuning, and each difference is answering a specific way the old
    /// single configuration mis-read this material:
    ///
    /// - grandness comes almost entirely from the swell path, because sustained music
    ///   never produces the spike-over-recent-baseline the hit path looks for;
    /// - the envelope is slow in both directions, so a two-minute solo reads as one long
    ///   grand passage instead of flickering;
    /// - the ambient background sits far higher, because this is the material where the
    ///   room around the mandala is doing most of the work;
    /// - the warp blooms outward rather than tunnelling inward, turns slowly, and is
    ///   mostly flow rather than spiral — sustained sound should look like weather, not
    ///   like a drill.
    const DRIFT: Self = Self {
        hit_weight: 0.25,
        swell_weight: 1.0,
        hit_lo: 1.25,
        hit_hi: 2.0,
        swell_lo: 0.58,
        swell_hi: 0.88,
        grand_attack: 1.6,
        grand_release: 6.0,
        trail_mul: 2.1,
        inject_mul: 1.15,
        bg_ambient: 0.090,
        bg_gate_lo: 0.02,
        bg_gate_hi: 0.11,
        // Negative: a slow outward bloom rather than a tunnel inward. Sized so light
        // injected at the centre takes about four seconds to reach the edge — fast
        // enough to read as movement, slow enough for the trail to build behind it.
        radial: -0.30,
        rotate: 0.34,
        spiral: 0.55,
        shear: 0.34,
        turb: 3.2,
        // Small on purpose, and the one number here that is not simply "the opposite of
        // PULSE": this profile can hold grandness near the top for minutes at a time, and
        // a bloom sized for a two-second transient would empty the frame and keep it
        // empty for the whole solo.
        grand_bloom: 0.35,
        pulse_kick: 0.15,
    };

    /// Dense, loud, everything at once. Fast and tight, so the image does not turn to
    /// soup under material that is already saturating every band.
    const SWARM: Self = Self {
        hit_weight: 0.8,
        swell_weight: 0.7,
        hit_lo: 1.35,
        hit_hi: 2.4,
        swell_lo: 0.62,
        swell_hi: 0.92,
        grand_attack: 0.7,
        grand_release: 3.5,
        trail_mul: 0.8,
        inject_mul: 0.88,
        bg_ambient: 0.06,
        bg_gate_lo: 0.05,
        bg_gate_hi: 0.20,
        radial: 1.95,
        rotate: -0.48,
        spiral: 2.6,
        shear: 0.18,
        turb: 2.6,
        grand_bloom: 5.0,
        pulse_kick: 0.6,
    };

    fn add_scaled(mut self, o: &Self, w: f32) -> Self {
        self.hit_weight += o.hit_weight * w;
        self.swell_weight += o.swell_weight * w;
        self.hit_lo += o.hit_lo * w;
        self.hit_hi += o.hit_hi * w;
        self.swell_lo += o.swell_lo * w;
        self.swell_hi += o.swell_hi * w;
        self.grand_attack += o.grand_attack * w;
        self.grand_release += o.grand_release * w;
        self.trail_mul += o.trail_mul * w;
        self.inject_mul += o.inject_mul * w;
        self.bg_ambient += o.bg_ambient * w;
        self.bg_gate_lo += o.bg_gate_lo * w;
        self.bg_gate_hi += o.bg_gate_hi * w;
        self.radial += o.radial * w;
        self.rotate += o.rotate * w;
        self.spiral += o.spiral * w;
        self.shear += o.shear * w;
        self.turb += o.turb * w;
        self.grand_bloom += o.grand_bloom * w;
        self.pulse_kick += o.pulse_kick * w;
        self
    }
}

/// Detects a sudden change in what the music is doing — a drop, a section boundary, a
/// hard change of texture.
///
/// It compares a fast reading of a five-element feature vector against a slow one and
/// looks for the distance between them to spike. That one test covers all three of the
/// cases worth reacting to, because each shows up as the fast vector leaving the slow one:
/// the floor dropping out moves the level element, a bass slam moves level and balance
/// together, and a change of instrumentation moves the balance and centroid with the level
/// barely touched.
///
/// The threshold is adaptive — the track's own recent novelty statistics — which is what
/// keeps this rare in ordinary playing without imposing a quota. There is deliberately
/// **no limit on how often it can fire**: a track built out of drops should get a vacuum
/// at every one of them. The only gate on repetition is hysteresis plus a short debounce,
/// which is edge detection rather than rationing — without it a single event fires on
/// every frame for as long as the condition holds.
struct RuptureDetector {
    fast: [f32; 5],
    slow: [f32; 5],
    novelty_ema: f32,
    novelty_dev: f32,
    /// False between a trigger and the novelty falling back to ordinary levels.
    armed: bool,
    debounce: f32,
}

impl RuptureDetector {
    fn new() -> Self {
        Self {
            fast: [0.0; 5],
            slow: [0.0; 5],
            novelty_ema: 0.0,
            novelty_dev: 0.0,
            armed: true,
            debounce: 0.0,
        }
    }

    /// Returns 0.0 on an ordinary frame, or the event's strength (0..1) on the frame a
    /// rupture is detected.
    fn update(&mut self, audio: &AudioFeatures, bass_w: f32, mid_w: f32, treble_w: f32, dt: f32) -> f32 {
        // Level is taken straight off the instantaneous RMS rather than off an envelope,
        // because the whole point is to catch the moment the floor moves — a smoothed
        // level cannot be sudden.
        let level = (audio.instant.rms / audio.long_term.loudness_ref.max(1e-4)).clamp(0.0, 1.5);
        let target = [level, bass_w, mid_w, treble_w, audio.instant.spectral_centroid];

        // Level is weighted up: a change of loudness is more of an event than a change
        // of timbre at the same loudness.
        let weights = [1.6f32, 1.0, 1.0, 1.0, 1.0];

        let mut novelty = 0.0f32;
        for (((fast, slow), &t), &weight) in self
            .fast
            .iter_mut()
            .zip(self.slow.iter_mut())
            .zip(target.iter())
            .zip(weights.iter())
        {
            *fast = ema(*fast, t, 0.30, dt);
            *slow = ema(*slow, t, 5.0, dt);
            let d = (*fast - *slow) * weight;
            novelty += d * d;
        }
        let novelty = novelty.sqrt();

        let hi = self.novelty_ema + 2.6 * self.novelty_dev + 0.06;
        let lo = self.novelty_ema + 0.8 * self.novelty_dev;

        self.debounce = (self.debounce - dt).max(0.0);
        let mut strength = 0.0;
        if self.armed && self.debounce <= 0.0 && novelty > hi {
            // How far past the bar it went, so a mild section change opens a small brief
            // hole and a real drop opens a large one.
            strength = smoothstep(hi, hi * 2.2 + 0.10, novelty);
            self.armed = false;
            self.debounce = 0.35;
        } else if !self.armed && novelty < lo {
            self.armed = true;
        }

        // Updated after the test so a trigger does not immediately raise its own bar.
        self.novelty_ema = ema(self.novelty_ema, novelty, 6.0, dt);
        self.novelty_dev = ema(self.novelty_dev, (novelty - self.novelty_ema).abs(), 8.0, dt);

        strength
    }
}

/// How long a vacuum event lasts, from the hole opening to it being fully refilled.
const VACUUM_DURATION: f32 = 1.7;

/// Three phase accumulators at golden-ratio-related rates.
///
/// DESIGN.md's "slow layer": the combined signal has an infinite period, so the warp
/// never returns to a state it has been in, while staying perfectly smooth. Seeded from
/// OS entropy at startup, which is what makes two runs of the same track diverge — with
/// identical phases the profile alone would give the same song the same motion forever.
#[derive(Debug, Clone, Copy)]
struct SlowPhases {
    theta: [f32; 3],
}

impl SlowPhases {
    const PHI: f32 = 1.618_034;

    fn seeded() -> Self {
        use std::hash::{BuildHasher, Hasher};
        // RandomState is seeded from OS entropy per process, so this gives a different
        // starting point every run without pulling in an RNG crate.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(0x5eed);
        let a = h.finish();
        let f = |shift: u32| ((a >> shift) & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
        Self { theta: [f(0), f(16), f(32)] }
    }

    /// Advance. `rate` is the base angular rate; the three phases run at rate, rate*phi,
    /// and rate*phi^2, whose ratios are irrational, so the sum never repeats.
    fn advance(&mut self, dt: f32, rate: f32) {
        let w = [rate, rate * Self::PHI, rate * Self::PHI * Self::PHI];
        for (theta, w) in self.theta.iter_mut().zip(w) {
            *theta = (*theta + dt * w) % std::f32::consts::TAU;
        }
    }

    fn sin(&self, i: usize) -> f32 {
        self.theta[i].sin()
    }
}

/// Owns the ping-pong HDR targets and the two render pipelines.
pub struct Renderer {
    width: u32,
    height: u32,
    time: f32,
    /// The song's hue, held still by a deadband until the material actually changes.
    palette_hue: f32,
    palette_spread: f32,
    palette_sat: f32,
    /// Live-tunable chromatic separation between the three channels' warps.
    chroma: f32,
    /// Smoothed "big moment" envelope — fast to rise, slow to fall, so a peak sustains
    /// instead of flickering frame to frame.
    grandness_env: f32,

    /// Live-tunable overall injection brightness, bound to keys so it can be dialed in
    /// against real music. Worth noting the tuned value came out *higher* than the
    /// single-layer default, not lower — predicting it from the equilibrium arithmetic
    /// alone was wrong, because the four layers barely overlap in practice.
    inject_gain: f32,
    /// Trail length, in seconds to fade to half brightness.
    feedback_half_life: f32,
    /// Smoothed 0..1 "how fluid does this track feel" envelope, mapped across
    /// `ORGANIC_MIN..ORGANIC_MAX` to drive `Uniforms::organic`. Slow on purpose — this
    /// is meant to drift over the course of a song, not react to a beat.
    organic_env: f32,
    /// Manual offset on that envelope, on the live keys, so the automatic mapping can
    /// still be nudged by hand against real music without fighting it.
    organic_bias: f32,
    /// Radius of the central mandala as a fraction of the frame's SHORT axis. A
    /// fraction rather than an absolute radius so it means the same thing on a portrait
    /// panel and a landscape one — see `Uniforms::core_radius`. On live keys because how
    /// big it *reads* depends on the panel, and this one is judged by eye.
    core_frac: f32,

    /// Smoothed archetype weights. ~12 s to cross over, so a track that changes
    /// character mid-song moves between configurations over a phrase, not a bar.
    profile: SongProfile,
    /// Never-repeating slow phases that keep the warp from being one fixed motion even
    /// within a single archetype.
    phases: SlowPhases,

    rupture: RuptureDetector,
    /// Seconds since the last vacuum event. Starts past the end of one so nothing fires
    /// on the first frame.
    vac_age: f32,
    vac_strength: f32,
    /// Seconds since the current ripple wavefront left the centre, and the rate at its
    /// crest. One wave at a time — see the relaunch rule in `render`.
    ripple_age: f32,
    ripple_amp: f32,

    ping: RenderTarget,
    pong: RenderTarget,
    /// True when `ping` holds the most recent frame (the one to sample from next).
    ping_is_latest: bool,

    sampler: wgpu::Sampler,
    uniforms: UniformBuffer<Uniforms>,

    feedback_pipeline: wgpu::RenderPipeline,
    feedback_bind_layout: wgpu::BindGroupLayout,
    color_pipeline: wgpu::RenderPipeline,
    color_bind_layout: wgpu::BindGroupLayout,

    quad_vbuf: wgpu::Buffer,
}

/// position (xy) + uv, matching `shaders/common/fullscreen_quad.wgsl`'s `VertexInput`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    QuadVertex { position: [1.0, -1.0], uv: [1.0, 1.0] },
    QuadVertex { position: [-1.0, 1.0], uv: [0.0, 0.0] },
    QuadVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
];

impl Renderer {
    pub fn new(device: &Device, width: u32, height: u32, output_format: TextureFormat) -> Self {
        use wgpu::util::DeviceExt;

        let ping = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-ping");
        let pong = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-pong");
        let sampler = RenderTarget::create_linear_sampler(device);
        let uniforms = UniformBuffer::<Uniforms>::new(device, "visual-uniforms");

        let quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen-quad"),
            contents: bytemuck::bytes_of(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
        };

        let (feedback_pipeline, feedback_bind_layout) = build_pipeline(
            device,
            "feedback",
            FEEDBACK_SHADER_SRC,
            HDR_FORMAT,
            &vertex_layout,
        );
        let (color_pipeline, color_bind_layout) = build_pipeline(
            device,
            "color",
            COLOR_SHADER_SRC,
            output_format,
            &vertex_layout,
        );

        Self {
            width,
            height,
            time: 0.0,
            // Amber, so the first seconds before the slow character features have settled
            // are a plausible colour rather than a lurch away from an arbitrary one.
            palette_hue: 0.08,
            palette_spread: 0.18,
            palette_sat: 0.72,
            // Tuned to be visible as fringing on moving edges without reading as a
            // misconverged projector. The feedback loop multiplies this by the trail
            // depth, so it wants to be far smaller than it looks like it should.
            // Backed off after the first look: the loop compounds this over the whole
            // trail, and at 0.0011 the fringing was strong enough to read as glass rather
            // than as iridescence. Raise it on `p` if it wants more.
            chroma: 0.0005,
            grandness_env: 0.0,
            // Tuned by ear against real music with all four layers running. Notably
            // ~2x higher than the value predicted from equilibrium alone: the layers
            // overlap far less than worst-case, so the loop sits well below saturation
            // and needs more drive than the arithmetic suggested.
            inject_gain: 3.43,
            feedback_half_life: 0.37,
            // Starts mid-range and settles within a few seconds of audio.
            organic_env: 0.5,
            organic_bias: 0.0,
            // 0.8 of the short axis. This is the 0.45 that was tuned by eye on the
            // portrait panel, re-expressed: that panel is +/-0.5625 wide in the shader's
            // aspect-corrected space, and 0.45 / 0.5625 = 0.8. Above ~1.0 the mandala
            // reaches the sides and leaves no background to see.
            core_frac: 0.8,
            profile: SongProfile::default(),
            phases: SlowPhases::seeded(),
            rupture: RuptureDetector::new(),
            vac_age: VACUUM_DURATION * 2.0,
            vac_strength: 0.0,
            ripple_age: 10.0,
            ripple_amp: 0.0,
            ping,
            pong,
            ping_is_latest: true,
            sampler,
            uniforms,
            feedback_pipeline,
            feedback_bind_layout,
            color_pipeline,
            color_bind_layout,
            quad_vbuf,
        }
    }

    /// Current "big moment" envelope (0..1), for logging/tuning.
    pub fn grandness(&self) -> f32 {
        self.grandness_env
    }

    /// Scale overall injected brightness. Clamped to a range that cannot go black or
    /// blow the tonemapper out.
    pub fn adjust_inject_gain(&mut self, factor: f32) {
        // Ceiling raised from 6.0 after tuning ran into it while exploring.
        self.inject_gain = (self.inject_gain * factor).clamp(0.15, 12.0);
        log::info!("inject_gain -> {:.2}", self.inject_gain);
    }

    /// Scale the chromatic separation between the three channels' warps.
    pub fn adjust_chroma(&mut self, factor: f32) {
        self.chroma = (self.chroma * factor).clamp(0.0, 0.012);
        log::info!("chroma -> {:.4}", self.chroma);
    }

    /// The song's current hue, 0..1, for logging/tuning.
    pub fn palette_hue(&self) -> f32 {
        self.palette_hue
    }

    /// Scale trail length (seconds to half brightness).
    pub fn adjust_trail(&mut self, factor: f32) {
        self.feedback_half_life = (self.feedback_half_life * factor).clamp(0.05, 4.0);
        log::info!("trail half-life -> {:.2}s", self.feedback_half_life);
    }

    /// Nudge the song-feel envelope up or down by hand. The automatic mapping still
    /// runs; this only shifts where a given track sits inside the tuned range.
    pub fn adjust_organic(&mut self, delta: f32) {
        self.organic_bias = (self.organic_bias + delta).clamp(-0.5, 0.5);
        log::info!(
            "organic bias -> {:+.2} (now {:.2})",
            self.organic_bias,
            self.organic()
        );
    }

    /// Current `organic` value: the song-feel envelope plus the manual bias, mapped
    /// across the tuned safe range.
    pub fn organic(&self) -> f32 {
        let t = (self.organic_env + self.organic_bias).clamp(0.0, 1.0);
        ORGANIC_MIN + (ORGANIC_MAX - ORGANIC_MIN) * t
    }

    /// Grow or shrink the central mandala, leaving more or less room for the background.
    pub fn adjust_core_scale(&mut self, factor: f32) {
        self.core_frac = (self.core_frac * factor).clamp(0.25, 1.60);
        log::info!("core scale -> {:.2} of short axis", self.core_frac);
    }

    /// Current mandala radius as a fraction of the frame's short axis, for logging.
    pub fn core_scale(&self) -> f32 {
        self.core_frac
    }

    /// Half-extent of the frame's short axis in the shader's aspect-corrected units:
    /// 0.5625 on a 1080x1920 portrait panel, 1.0 on a 1920x1080 landscape one. The
    /// shader normalizes height to 1 either way, which is exactly why an orientation
    /// change is not a 90-degree rotation of the same picture — it changes which axis is
    /// short *and* the scale of everything measured against it.
    fn short_half(&self) -> f32 {
        self.width.min(self.height) as f32 / self.height.max(1) as f32
    }

    /// Which configuration the visual is currently running, for logging/tuning. This is
    /// the first thing to look at when a track does not look the way it should.
    pub fn profile(&self) -> SongProfile {
        self.profile
    }

    /// Strength of the vacuum event currently open, 0..1, for logging/tuning. Zero
    /// almost all the time by design.
    pub fn vacuum(&self) -> f32 {
        let a = (self.vac_age / VACUUM_DURATION).clamp(0.0, 1.0);
        self.vac_strength * smoothstep(0.0, 0.10, a) * (1.0 - smoothstep(0.40, 1.0, a))
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.ping = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-ping");
        self.pong = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-pong");
        self.ping_is_latest = true;
    }

    /// Render one frame: feedback pass into the back buffer, then color pass to `target`.
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        target: &wgpu::TextureView,
        dt: f32,
        audio: &AudioFeatures,
        structural: StructuralState,
    ) {
        self.time += dt;

        let bands = &audio.bands.energy;
        let bass_raw = bands[0] + bands[1];
        let mid_raw = bands[2] + bands[3] + bands[4];
        let treble_raw = bands[5] + bands[6];
        let total = (bass_raw + mid_raw + treble_raw).max(1e-4);

        let mut band_energy = [0.0f32; 8];
        band_energy[..7].copy_from_slice(bands);

        let core_radius = self.core_frac * self.short_half();

        // Which kind of track is this? Everything below reads its constants out of the
        // blended archetype rather than from one global tuning.
        self.profile.lerp(&SongProfile::from_audio(audio), 12.0, dt);
        let cfg = self.profile.params();

        let level = audio.long_term.level_norm;

        // "Grand moment" detection, by two paths that answer the question differently,
        // because two kinds of music make a big moment in incompatible ways.
        //
        // The HIT path is the original: loud relative to this track's own recent context.
        // It is the right question for transient-led music, where a chorus arrives as a
        // step change against the verse.
        //
        // The SWELL path is new, and is what was missing. Sustained music raises its own
        // recent baseline as it swells, so by the time a held passage is at full height
        // the ratio the hit path measures has already collapsed back to ~1 and the whole
        // moment reads as ordinary. Measuring against the track's 60 s loudness ceiling
        // instead asks "are we near the top of what this track ever does, and staying
        // there" — which a two-minute guitar lead answers yes to for its whole length.
        //
        // Combined with `max` rather than a sum: they are two readings of one thing, and
        // adding them double-counts a track that happens to satisfy both.
        let baseline = audio.long_term.avg_energy.max(1e-3);
        let ratio = audio.instant.rms / baseline;

        // Both paths are gated on level *normalized by this track's own ceiling* rather
        // than on an absolute RMS threshold. The absolute gate this replaces was the
        // other half of the failure: it was calibrated against a loud modern master, so a
        // dynamic-range-preserving older mix never cleared it however grand the passage.
        // Calibrated to land near where the absolute gate it replaced actually sat on
        // typical material — that gate averaged ~0.7 rather than saturating, and swapping
        // it for one that reaches 1.0 on anything merely loud is part of why grandness
        // rose across the board.
        let audible = smoothstep(0.35, 0.80, level);

        let hit = smoothstep(cfg.hit_lo, cfg.hit_hi, ratio) * audible * cfg.hit_weight;

        // Being near the ceiling only means something if the track ever leaves it. On its
        // own that test calls a passage grand whenever the level is simply steady and up,
        // which is the normal state of most material and the permanent state of speech —
        // so it is multiplied by how far the level currently sits above this track's own
        // 25-second norm. Kept as a factor between 0.55 and 1 rather than a gate, because
        // a long sustained passage does eventually raise its own 25 s norm and should not
        // then be demoted to ordinary.
        let slow = audio.long_term.slow_energy.max(1e-4);
        let elevated = 0.55 + 0.45 * smoothstep(1.05, 1.35, audio.long_term.avg_energy / slow);

        let swell = smoothstep(cfg.swell_lo, cfg.swell_hi, level)
            * (0.45 + 0.55 * audio.character.sustained)
            * elevated
            * cfg.swell_weight;

        // Nothing without dynamic range gets to be grand. A narration video, a podcast or
        // a stream sits at one level indefinitely: it clears every ceiling-relative test
        // there is, because its ceiling *is* its normal level. Judging the material rather
        // than the moment is the only test that separates that case from music, and it is
        // applied to both paths because continuous speech trips the hit path too — syllable
        // peaks clear a 1.5x-over-context ratio easily.
        //
        // A factor rather than a gate, and floored at 0.12 rather than 0, so a compressed
        // but genuinely musical track is damped instead of being switched off.
        let range_gate =
            0.12 + 0.88 * smoothstep(0.22, 0.50, audio.long_term.dynamic_range);

        let target_grand = (hit.max(swell) * range_gate).clamp(0.0, 1.0);
        let tau = if target_grand > self.grandness_env {
            cfg.grand_attack
        } else {
            cfg.grand_release
        };
        self.grandness_env = ema(self.grandness_env, target_grand, tau, dt);

        // --- Palette ---------------------------------------------------------------
        // The colour is the song's, and it is meant to stay put. A deadband holds it
        // exactly still until the material has moved a noticeable distance around the
        // circle, and only then does it travel — on the short way round, so a song whose
        // hue creeps past the wrap point does not sweep through every other colour to get
        // to a neighbouring one.
        //
        // What this replaces was a continuous ~90 s hue rotation. That is a filter rather
        // than a palette: it moves every pixel by the same angle, so the colour ends up
        // being a property of when you happened to be watching instead of what is playing.
        let target_hue = palette_target_hue(&audio.character);
        let delta = circular_delta(self.palette_hue, target_hue);
        if delta.abs() > 0.045 {
            // Slow even once it is moving: this should read as a new song having a
            // different colour, never as the colour animating.
            self.palette_hue = (self.palette_hue + delta * (1.0 - (-dt / 20.0).exp()))
                .rem_euclid(1.0);
        }

        // A dense mix gets a wider palette because it has the spectral content to justify
        // more than one hue; a sparse one stays analogous and keeps its identity.
        let target_spread = 0.10 + 0.32 * audio.character.density;
        self.palette_spread = ema(self.palette_spread, target_spread, 20.0, dt);
        // Dense mixes desaturate slightly — several saturated hues overlapping in a
        // feedback loop average toward mud, and pulling the saturation back is what stops
        // that without giving up the colour.
        let target_sat = 0.82 - 0.28 * audio.character.density;
        self.palette_sat = ema(self.palette_sat, target_sat, 20.0, dt);

        // Song-feel envelope. Very long time constants — ~8s to settle on a new feel,
        // ~16s to relax back — so this tracks the track, not the bar. Slower to fall so a
        // quiet passage inside a busy song does not snap the image back to hard geometry.
        let target_feel = song_feel(
            audio,
            self.grandness_env,
            bass_raw / total,
            mid_raw / total,
            treble_raw / total,
        );
        let feel_tau = if target_feel > self.organic_env { 8.0 } else { 16.0 };
        self.organic_env += (1.0 - (-dt / feel_tau).exp()) * (target_feel - self.organic_env);

        // Frame-rate-independent trail retention. Over `feedback_half_life` seconds the
        // image fades to half, whether we are running at 60 or 178 fps. The archetype
        // scales it: sustained material wants a long smear, dense material wants the
        // image cleared out before the next bar arrives.
        let half_life = (self.feedback_half_life * cfg.trail_mul).clamp(0.05, 6.0);
        let feedback_decay = (-dt * std::f32::consts::LN_2 / half_life).exp();

        // Warp character. This is the answer to "it is just one motion every time": the
        // rates below used to be six hard-coded constants, so every track on every run
        // got the same inward spiral and only its amplitude changed. Now the archetype
        // sets the *kind* of motion, the slow phases wander it inside a bounded range,
        // and the phases are entropy-seeded, so the same song twice is not the same
        // motion twice either.
        //
        // Everything is clamped, per DESIGN.md's safety contract: the wandering may
        // modulate a rate inside a hand-checked range but never set one.
        self.phases.advance(dt, 0.021);
        let org = self.organic();
        // Scaled by `organic` so the wander is seasoning on the archetype rather than a
        // second, competing source of motion — see the organic range note above.
        let w = |i: usize, amount: f32| self.phases.sin(i) * amount * org;

        let bloom = cfg.grand_bloom * self.grandness_env;
        let kick = cfg.pulse_kick * audio.short_term.attack.min(1.5);
        // Positive radial is a tunnel inward; grandness and transients push outward. The
        // floor is the real safety limit here: an outward rate of 0.95/s empties the frame
        // from centre to edge in about 2.5 s, and anything faster, held, outruns injection
        // and leaves the image blank. The old code never discovered that only because its
        // grandness never rose far enough to ask the question.
        let warp_radial = (cfg.radial * (1.0 + w(0, 0.45)) - bloom - kick).clamp(-0.95, 2.80);
        // Never fully dies with `organic` at its floor — a uniform turn is a character of
        // the motion, not a departure from geometry.
        let warp_rotate = (cfg.rotate * (0.4 + 0.6 * org) + w(1, 0.28)).clamp(-1.20, 1.20);
        // The `organic` and grandness factors here were applied inside the shader before
        // the rates moved to the CPU; they are kept so PULSE reproduces the tuned look.
        let warp_spiral = (cfg.spiral
            * (1.0 + w(2, 0.55))
            * (0.6 + 0.8 * treble_raw / total)
            * org
            * (0.5 + self.grandness_env))
            .clamp(-3.50, 3.50);
        let warp_shear = (cfg.shear * org + w(0, 0.22) * 0.5).clamp(-0.80, 0.80);
        let warp_turb = (cfg.turb * (1.0 + w(1, 0.40)) * (0.6 + 0.8 * mid_raw / total) * org)
            .clamp(0.0, 4.20);

        // --- Rupture: vacuum events and the ripple they launch --------------------
        self.vac_age = (self.vac_age + dt).min(VACUUM_DURATION * 2.0);
        // Capped rather than left to accumulate: the wave is long dead by then, and an
        // age of a few thousand seconds puts `sin(dr * 24.0)` deep into the range where
        // f32 has no precision left.
        self.ripple_age = (self.ripple_age + dt).min(12.0);

        let rupture = self.rupture.update(
            audio,
            bass_raw / total,
            mid_raw / total,
            treble_raw / total,
            dt,
        );
        if rupture > 0.0 {
            log::info!("rupture {:.2} — vacuum", rupture);
            self.vac_age = 0.0;
            self.vac_strength = 0.35 + 0.65 * rupture;
        }

        // The hole opens fast, holds, and refills. Deliberately not a decaying envelope:
        // a void that fades back in reads as a dimmer, not as space closing.
        let a = (self.vac_age / VACUUM_DURATION).clamp(0.0, 1.0);
        let vacuum = self.vac_strength
            * smoothstep(0.0, 0.10, a)
            * (1.0 - smoothstep(0.40, 1.0, a));
        // Sized against the live core so it always reads as "the middle of the mandala is
        // gone" whatever the mandala has been set to, and expanding slightly while open.
        let vacuum_radius = 0.62 * core_radius * vacuum * (0.80 + 0.40 * a);

        // Ripples are launched from the centre by onsets, and much harder by a rupture.
        // Only one wave travels at a time: a new launch takes over only if it would be
        // stronger than what is already out there, so an ordinary beat cannot stomp the
        // wave a drop just sent out.
        let live = self.ripple_amp * (-self.ripple_age * 1.1).exp();
        let mut launch: f32 = 0.0;
        if audio.short_term.onset {
            launch = 0.8 + 2.2 * audio.short_term.transient_strength;
        }
        if rupture > 0.0 {
            launch = launch.max(4.0 + 4.0 * rupture);
        }
        if launch > live {
            self.ripple_age = 0.0;
            self.ripple_amp = launch;
        }

        // Orb radius, moved here from the shader because the warp needs it as well: the
        // kaleidoscope fold has to know where the orb ends so it can begin outside it.
        // Sized off the normalized level rather than raw RMS so it reaches the same size
        // on a quiet master as on a loud one, and off whichever voice this kind of music
        // leads with rather than off the bass unconditionally.
        let voice = (bass_raw / total) * (1.0 - self.profile.drift)
            + (mid_raw / total) * self.profile.drift;
        let orb_radius = (0.10 + 0.30 * voice * (0.4 + level * 0.45)) * core_radius;

        // The outer field always drifts outward, on top of whatever the core is doing —
        // so the background streams away and fades rather than pulsing in place. Sustained
        // material and grand passages push it further before it goes.
        let bg_outflow = 0.40 + 0.50 * self.profile.drift + 0.35 * self.grandness_env;

        let uniform_data = Uniforms {
            resolution: [self.width as f32, self.height as f32],
            time: self.time,
            dt,
            band_energy,
            rms: audio.instant.rms,
            energy_velocity: audio.short_term.energy_velocity,
            attack: audio.short_term.attack,
            decay: audio.short_term.decay,
            bass_w: bass_raw / total,
            mid_w: mid_raw / total,
            treble_w: treble_raw / total,
            spectral_centroid: audio.instant.spectral_centroid,
            symmetry: structural.symmetry.max(2),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            warp_radial,
            feedback_decay,
            palette_hue: self.palette_hue,
            grandness: self.grandness_env,
            inject_gain: self.inject_gain * cfg.inject_mul,
            exposure: 1.0,
            organic: org,
            core_radius,
            warp_rotate,
            warp_spiral,
            warp_shear,
            warp_turb,
            level_norm: level,
            bg_ambient: cfg.bg_ambient,
            bg_gate_lo: cfg.bg_gate_lo,
            bg_gate_hi: cfg.bg_gate_hi,
            profile_drift: self.profile.drift,
            profile_pulse: self.profile.pulse,
            vacuum,
            vacuum_radius,
            ripple_age: self.ripple_age,
            ripple_amp: self.ripple_amp,
            bg_outflow,
            // Widened on a grand moment: the separation is what makes a peak read as
            // luminous rather than merely brighter.
            chroma: self.chroma * (1.0 + 1.2 * self.grandness_env),
            palette_spread: self.palette_spread,
            palette_sat: self.palette_sat,
            orb_radius,
            _pad3: 0.0,
        };
        self.uniforms.update(queue, &uniform_data);

        let (src, dst) = if self.ping_is_latest {
            (&self.ping, &self.pong)
        } else {
            (&self.pong, &self.ping)
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-frame"),
        });

        // Pass 1: feedback (warp + decay + inject) — src -> dst.
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("feedback-bind-group"),
                layout: &self.feedback_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.feedback_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
            pass.draw(0..4, 0..1);
        }

        // Pass 2: color (palette + tonemap + dither) — dst -> swapchain.
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("color-bind-group"),
                layout: &self.color_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("color-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.color_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
            pass.draw(0..4, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        self.ping_is_latest = !self.ping_is_latest;
    }
}

fn build_pipeline(
    device: &Device,
    label: &str,
    shader_src: &str,
    target_format: TextureFormat,
    vertex_layout: &wgpu::VertexBufferLayout,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}-bind-layout")),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-pipeline-layout")),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_layout)
}

#[cfg(test)]
mod tests {
    use super::{COLOR_SHADER_SRC, FEEDBACK_SHADER_SRC};

    /// Parse and validate both shaders exactly as wgpu will at pipeline creation.
    /// Without this a WGSL error only surfaces as a panic on the first rendered frame.
    fn validate(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{label} failed to parse:
{}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label} failed validation: {e:?}"));
    }

    #[test]
    fn feedback_shader_is_valid_wgsl() {
        validate("feedback", FEEDBACK_SHADER_SRC);
    }

    #[test]
    fn color_shader_is_valid_wgsl() {
        validate("color", COLOR_SHADER_SRC);
    }

    /// The `name: type` pairs of the `Uniforms` struct in a shader source, in order,
    /// with comments and blank lines stripped.
    fn uniform_fields(src: &str) -> Vec<(String, String)> {
        let start = src.find("struct Uniforms {").expect("no Uniforms struct");
        let rest = &src[start..];
        let end = rest.find("};").expect("unterminated Uniforms struct");
        rest[..end]
            .lines()
            .map(|l| l.split("//").next().unwrap_or("").trim())
            .filter_map(|l| l.split_once(':'))
            .map(|(n, t)| {
                (
                    n.trim().to_string(),
                    t.trim().trim_end_matches(',').trim().to_string(),
                )
            })
            .collect()
    }

    /// Count 4-byte slots in a list of WGSL fields.
    fn wgsl_slots(fields: &[(String, String)]) -> usize {
        fields
            .iter()
            .map(|(_, ty)| {
                let ty = ty.as_str();
                if ty.starts_with("array<vec4<f32>, 2>") {
                    8
                } else if ty.starts_with("vec4") {
                    4
                } else if ty.starts_with("vec2") {
                    2
                } else {
                    1
                }
            })
            .sum()
    }

    /// The uniform layout is written out three times — once in Rust and once in each
    /// shader — and nothing in the type system relates them. Getting them out of step
    /// does not fail to compile or fail validation; it silently feeds every field after
    /// the divergence the wrong value, which is close to impossible to recognize by
    /// looking at the output. So check it here instead.
    #[test]
    fn uniform_layout_matches_between_rust_and_wgsl() {
        let feedback = uniform_fields(FEEDBACK_SHADER_SRC);
        let color = uniform_fields(COLOR_SHADER_SRC);
        assert_eq!(
            feedback, color,
            "the two shaders declare different Uniforms blocks"
        );

        let rust_slots = std::mem::size_of::<super::Uniforms>() / 4;
        assert_eq!(
            wgsl_slots(&feedback),
            rust_slots,
            "WGSL Uniforms has a different number of 4-byte slots than the Rust struct"
        );
    }
}
