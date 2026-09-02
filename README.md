# trippinator

A real-time audio visualizer for a secondary portrait display. It listens to whatever your
system is playing and renders a feedback-loop organism on a 1080×1920 panel at ~178fps.

The picture is built from two things that do different jobs:

- **The mandala** — a central orb ringed by four wave rings, a spectral ring, orbiting
  emitters, and a constellation of per-band nodes, all folded through a kaleidoscope. It
  runs continuously and visualizes whatever is playing, so it has to work on anything.
- **The background** — everything outside it. A quiet ambient field always tracks the
  track, and a second tier gated on "grandness" is genuinely absent until a passage earns
  it. The mandala says *what* is playing; the background says how it feels, and when
  something matters.

Nothing is drawn as geometry. Every frame samples the previous frame through a warp,
decays it, and adds new audio-driven light on top, so what you see is the accumulated
history of the music rather than a picture of the current moment.

## Requirements

- **Windows** (audio loopback capture is WASAPI via `cpal`)
- A GPU with a working `wgpu` backend — Vulkan, DX12, or Metal
- Rust (edition 2024 — needs a recent toolchain)
- Ideally a **secondary portrait monitor**; it falls back to any secondary, then to primary

## Run

```bash
cargo run --release
```

It picks the audio output device your system defaults to, finds a portrait secondary
display, and opens undecorated and fullscreen-sized on it. `Esc` quits, `F11` toggles
borderless fullscreen.

## Controls

Everything visual is tuned live against real music rather than computed, and each press is
logged so a value you like can be read back out of the terminal.

| Keys | Adjusts | Notes |
|---|---|---|
| `-` / `=` | Injection brightness | Overall drive into the feedback loop |
| `,` / `.` | Trail length | Seconds to fade to half brightness |
| `[` / `]` | Kaleidoscope symmetry | Fold order, 3–12 |
| `k` / `l` | Mandala size | The background is whatever is left over |
| `;` / `'` | Organic bias | Shifts the automatic song-feel mapping |
| `F11` | Borderless fullscreen | |
| `Esc` | Quit | |

Once a second it logs what it is seeing and what it decided:

```
178 fps | rms 0.088 | lvl 0.74 | bass 0.915 | mid 0.208 | treble 0.151 | grand 0.03
        | pulse 0.61 drift 0.18 swarm 0.21 | onsets 4.2/s | organic 0.24 | core 0.45
```

Four derived values are worth watching, and between them they usually say which input is
responsible when the visuals feel wrong:

- **`lvl`** — how loud this is *for this track*, against its own 60-second loudness
  ceiling. This is what every level decision keys off, not raw `rms`, so that a quiet
  master reaches the same visual range as a loud one.
- **`pulse` / `drift` / `swarm`** — which configuration the track is being given.
- **`grand`** — the big-moment envelope, and **`dyn`** — how much dynamic range the
  material has. Nothing without range gets to be grand: a narration video or a stream sits
  at one level indefinitely and so clears every ceiling-relative test there is, because its
  ceiling *is* its normal level. `dyn` near 0 with `grand` near 0 on speech is correct.
- **`vac`** — a vacuum event, 0 almost all the time by design.
- **`organic`** — how far from strict geometry the image is being allowed to go.

## How it reacts

Audio is analysed across three timescales (`crates/analysis/audio_analysis`): instantaneous
RMS/centroid/flux, short-term attack and decay, and long-term energy envelopes at four
different time constants, plus seven frequency bands and onset detection. The derived
signals that drive the look:

- **Song profile** — see below. Decides *which configuration* runs.
- **Spectral balance** (bass/mid/treble weights) tints the palette, so a bassy track and a
  trebly one read as different moods of the same visual.
- **Grandness** — the big-moment envelope, by two paths. The *hit* path is loud relative to
  this track's own recent context, which is how transient-led music makes a big moment. The
  *swell* path is sustained near the top of the track's own 60-second loudness ceiling,
  which is how sustained music makes one — a held passage raises its own recent baseline as
  it builds, so by the time it is at full height the ratio the hit path measures has
  already collapsed back to ~1. The two are combined with `max`, and the archetype decides
  how much each counts. It gates the background's grand tier and blooms the image outward.
- **Song feel** — how fluid the current track is, from level churn, timbral movement and
  spectral balance. It maps to `organic`, which scales every departure from strict
  geometry. Built deliberately from slow quantities, not from `attack`: looseness
  flickering at beat rate reads as the image glitching rather than as the music having a
  character. ~8s to settle, ~16s to relax.

Values that were found by ear, not derived, are marked as such in the code with the range
they were swept over.

## Song profiles

Different kinds of music need different settings, not one tuning that has to work for
everything. Every track is classified continuously across three archetypes, from slow
features — onset rate, crest factor, spectral occupancy, spectral balance — with time
constants in the tens of seconds. The weights sum to 1 and everything downstream is a
linear blend of the three parameter sets, so a track that is half riff and half held chord
gets a configuration halfway between, and one that changes character mid-song crosses over
during a phrase rather than snapping.

| | **pulse** | **drift** | **swarm** |
|---|---|---|---|
| Material | transient-led — drums, riffs | sustain-led — held tones, atmospheric leads | dense and busy — full spectrum, loud |
| Grandness from | hits above local context | sustained level near the track's ceiling | both |
| Envelope | fast (0.4 s / 2.5 s) | slow (1.6 s / 6 s) | medium |
| Trails | as tuned | ×2.1 | ×0.8 |
| Warp | tunnels inward, spirals | blooms outward, turns, flows | fast tunnel, counter-rotates |
| Background | quiet ambient, high gate | strong ambient, low gate, mid-band driven | medium |

The warp is where this is most visible. It used to be one motion — flow bend, one-signed
swirl, always inward — so every track on every run got the same inward spiral and only the
amplitude changed. It is now six independent components (radial, uniform rotation,
differential spiral, shear, turbulence, transient kick) whose signs and balance come from
the archetype, wandered inside a bounded range by phase accumulators at golden-ratio
frequency ratios, seeded from OS entropy per run. Sustained music blooms outward and flows;
transient music tunnels inward and spirals; and the same song twice is not the same motion
twice.

Every component is a **rate per second** multiplied by `dt` in the shader. The constants
these replaced were applied per frame, which made the speed of every motion in the image a
function of how fast the GPU happened to be running.

## Ruptures, ripples and the background

**Ripples** are launched from the centre by onsets and travel outward as a wave packet,
applied as a radial velocity so the space compresses ahead of the front and rarefies
behind. Only one wave is in flight at a time; a new launch takes over only if it would be
stronger than what is already travelling, so an ordinary beat cannot stomp the wave a drop
just sent out.

**Vacuum events** remove the middle of the image from the space rather than from the
display. The warp carries the surrounding field outward, the history inside the hole is
annihilated, and injection is masked — so for about a second there is genuinely no orb
there, as opposed to a black disc drawn over one that is still running and still feeding
the loop.

They fire when a rupture detector sees a fast reading of a five-element feature vector
(level, three spectral weights, centroid) pull away from a slow one. One test covers every
case worth reacting to: the floor dropping out moves the level element, a bass slam moves
level and balance together, a change of instrumentation moves balance and centroid with the
level barely touched. The threshold is the track's own recent novelty statistics.

**There is no limit on how often this can fire.** A track built out of drops gets a vacuum
at every one of them. The only gate on repetition is hysteresis plus a 0.35 s debounce,
which is edge detection — without it a single event fires on every frame for as long as
the condition holds.

**The background** carries an always-outward radial rate of its own, separate from
whatever the core is doing, so the outer field streams outward and fades on the way rather
than sitting where it was injected and pulsing in place. The mandala can tunnel inward
while the field around it flows out.

## Working on the shaders

`shaders/render/feedback.wgsl` is where the image comes from, and it has three invariants
documented at the top of the file. They are not style preferences — each one was written
after the corresponding artifact showed up on screen:

1. **Injection is energy-per-second** and is multiplied by `dt`. A per-frame constant makes
   brightness frame-rate dependent and saturates the loop to white.
2. **The warp applies a per-frame step**, never a function of absolute `time` that
   compounds. The feedback texture already carries the accumulated history. Those steps
   are `rate * dt`, never a bare per-frame constant, for the same reason as invariant 1.
3. **Every function of the polar angle must be periodic with period TAU**, and the warp
   must be continuous in screen space. `atan2` has a branch cut; a one-pixel jump across it
   is invisible in a single frame, but a feedback loop writes that discontinuity into the
   texture and re-reads it every frame until it is a permanent hard seam.

The general lesson behind all three: **in a feedback loop, a defect that is invisible in one
frame is amplified ~95× by the trail.** Anything that bends a *coordinate* leaves a
geometric signature the loop can stand up into an edge — clamping out-of-frame samples
gives border streaks, fading them to zero gives dead black regions, mirroring them gives a
crease where the gradient reverses. Blending sampled *colours* has no geometry in it and is
the tool that has consistently worked.

`cargo test -p trippinator-render` validates both shaders with `naga`, so a WGSL error
fails the test run instead of panicking on the first rendered frame.

## Not wired up yet

- **Screen capture and analysis.** `crates/capture/screen` and
  `crates/analysis/screen_analysis` are built and compile, but the engine is audio-only —
  desktop reactivity is v2 scope.
- **`assets/config.toml`.** It documents the intended settings but nothing reads it yet;
  the audio device is the system default and the monitor is auto-detected. Tuning happens
  on the live keys.
- **Chaotic dynamics.** DESIGN.md describes structural state eventually being driven by
  Poincaré-section crossings of an attractor. Today `symmetry` is set by hand and the rest
  is driven directly from audio.

See [DESIGN.md](DESIGN.md) for the fuller intent.

## License

MIT
