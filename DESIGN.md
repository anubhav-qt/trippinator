# Trippinator — Design

An audio- and desktop-reactive visualizer for a secondary monitor, portrait or landscape.
Runs alongside music *and* games, as ambient immersion — an extension of whatever is on
the main screen, not a thing that competes with it.

> **This document is intent, not specification, and it lags the code.**
>
> It is where the reasoning lives — why a mechanism was chosen, what it is trying to buy,
> what went wrong last time. It is *not* a description of what is currently built, and
> parts of it describe approaches that were tried and replaced. **When the two disagree,
> the code is right and this file is out of date.**
>
> Sections carry a marker so the difference is visible at a glance:
>
> - `[DONE]` — built, and the code matches this description.
> - `[PLANNED]` — not built. Written down so the idea is not lost, not because it is next.
> - `[SUPERSEDED]` — was built or was intended, and has since been replaced. The text is
>   kept because the reasoning is still useful; the note says what replaced it.
>
> Anything unmarked is background reasoning that is not a claim about the code.
> Update the marker in the same change that makes it wrong. If you are about to build
> something because this document describes it, check the code first.

## The governing idea

> The image is produced by a state function that carries its own history and never repeats.
> Input perturbs that system. Input does not draw the image.

Concretely: two nested memories.

1. **Pixel memory** — the frame buffer feeds back into itself through a warp. `frame[n] =
   blend(warp(frame[n-1]), inject(audio))`. This is where trails, tunnels, zoom and
   kaleidoscope come from, and it is literally a state function over previous states.
2. **Parameter memory** — a chaotic ODE integrated every frame supplies the warp's
   parameters. Bounded, aperiodic, and sensitive to initial conditions.

These two want the same architecture, which is why the psychedelic aesthetic and the
"never repeats" requirement are not in tension — they are the same structure at two scales.

## Non-repetition: the actual mechanism

Three properties are needed, and they are different things:

| Property | Mechanism |
|---|---|
| Two runs of the same song differ | Seed the state from OS entropy at startup |
| That difference *grows* instead of washing out | Positive Lyapunov exponent — a chaotic, not merely complex, system |
| It never settles into a loop | Strange attractor (aperiodic by construction) + irrational-ratio phases |

The second row is the one that is easy to get wrong. A stable dynamical system driven by
audio converges to the same trajectory regardless of where it started — the seed washes
out and the same song *does* give the same visual. Chaos is what makes the seed matter
forever.

### Core: Thomas' cyclically symmetric attractor

```
dx/dt = sin(y) - b*x
dy/dt = sin(z) - b*y
dz/dt = sin(x) - b*z
```

Chosen over Lorenz for three reasons:

- **Cyclically symmetric** — x, y, z are statistically interchangeable, so all three can
  drive unrelated visual parameters without one dominating.
- **Bounded** in roughly [-5, 5] with no blow-up risk, so mapping to visual ranges is safe.
- **`b` is a smooth dial from order into chaos.** `b > 0.32` → periodic, then a fixed
  point. `b ≈ 0.32` → onset of chaos. `b → 0.208` → strongly chaotic.

That last point is the design payoff: **drive `b` from audio energy.** Quiet passages pull
`b` up and the dynamics become near-periodic — the visual gets orderly, calm, almost
still. Loud passages push `b` down into chaos and the geometry starts wandering. The
*character of the dynamics itself* tracks the music, not just an amplitude multiplier.
That is an honest coupling, and it is the thing that stops it feeling like a level meter.

Integrated with RK4 at fixed `dt` (accumulator, decoupled from frame rate) so behavior is
identical at 60 and 165 fps.

**Divergence budget.** Thomas has λ₁ ≈ 0.05 per time unit at b=0.208. Scaling time so one
unit ≈ 0.2 s, a 1e-6 seed difference reaches O(1) in roughly 30–60 s — two runs of the same
track visibly part ways inside the first verse, which is what we want.

### Colour comes from the song, not the clock `[DONE]`

An earlier version drifted the hue continuously, ~90 s per rotation, on the reasoning in
"timescale hierarchy" below: a process slower than the viewing session means the session
cannot contain a repeat.

**That was rejected on use.** A slow global hue rotation is a filter, not a palette — it
moves every pixel by the same angle, so it changes what colour the image is without
changing how it is coloured, and it makes the colour a property of *when you are watching*
rather than of what is playing. Two different songs an hour apart look like the same song
under different lighting.

The palette is now derived from the song's own slow spectral character and then **held
still**: a deadband stops it moving at all until the material genuinely changes, and it
moves on the short way round the hue circle when it does. A song has a colour; it keeps
it. This is a deliberate exception to the never-repeat principle — colour identity is worth
more here than colour novelty, and the novelty budget is spent on geometry and motion
instead.

Note the contrast with the warp, which *is* entropy-seeded per run: the same song should
move differently every time and be coloured the same every time.

### Slow layer: quasi-periodic phases

For anything that must stay smooth (palette rotation, warp angle drift), chaos is too
jittery. Use phase accumulators with irrational frequency ratios:

```
ω = ω₀ * [1, φ, φ²]        φ = 1.6180339887...
θᵢ += dt * ωᵢ * (1 + k * energy)
```

The combined signal has infinite period — guaranteed never to repeat, perfectly smooth,
zero risk of an ugly state. Cheap insurance where chaos would be too violent.

### Seeding and reproducibility

Seed `(x, y, z, θ)` from OS entropy on startup and **log the seed**. Add `--seed N` to
replay a run that looked good. Default is never-repeating; determinism is opt-in.

## The safety contract

Chaos that can reach any parameter value will eventually find an ugly one. So:

> The chaotic state may only **modulate** a parameter inside a hand-tuned safe range.
> It may never **set** a parameter directly.

Every mapping has the form:

```
param = base(t) + range(t) * shape(tanh(gain * source))
```

bounded by construction. `base` and `range` are tuned by hand until every value in the
range looks good; the dynamics explore that range.

Note `base` and `range` are **functions of time**, not constants. That is not cosmetic —
see below.

## Staying novel (not merely aperiodic)

Two requirements were conflated in an earlier draft of this document, and they need
different mechanisms:

| Requirement | Solved by |
|---|---|
| The same song twice looks different | Entropy seed + positive Lyapunov exponent |
| It does not get boring over a three-hour session | Everything in this section |

The seed-and-chaos machinery above fully solves the first and **does not solve the
second.** A strange attractor is *ergodic*: the trajectory is unpredictable, but the
long-run distribution over its bounded region is **fixed**. After roughly 10–15 minutes a
viewer has effectively seen the whole invariant measure, and everything afterwards is a
rearrangement of the familiar. Perception adapts to distributions, not trajectories.

So: strictly aperiodic and perceptually repetitive are entirely compatible. Chaos buys
unpredictability. It does not buy novelty. Three mechanisms buy novelty.

### 1. Timescale hierarchy — break stationarity

The safe box does not hold still. `base` and `range` drift on a slow system (minutes)
while chaos wanders inside it (seconds). Implemented as a bounded random walk over the
parameter manifold with reflecting boundaries at the hand-tuned safe limits.

Design target: **the slowest process in the system must have a period longer than a
viewing session.** If it does, the session cannot contain a repeat.

### 2. Discrete structural switching — variety of kind `[SUPERSEDED]`

> **Replaced by song archetypes and the parameterized warp.** The five warp topologies and
> four injection geometries below were built as selectable modes and then collapsed: the
> injection layers all render *together* (they compose better than they alternate), and the
> single kaleidoscope warp was parameterized into six continuous components instead of five
> discrete variants. Structural variety now comes from the archetype blend and the warp
> character, not from switching. `symmetry` is the only discrete axis left, and it is on a
> key rather than driven by an attractor. **There are no `warp_mode` or `inject_mode`
> uniforms and there never will be — do not go looking for them.**

Continuous modulation of one structure always reads as "one visual with a knob wiggling."
Genuine novelty needs changes of *kind*. Poincaré-section crossings of the attractor —
aperiodic and unpredictable by construction — trigger discrete jumps:

- kaleidoscope symmetry order (3–10, 8 values)
- warp topology (5 variants)
- injection geometry (4 variants)

8 × 5 × 4 = 160 structural combinations, walked in a non-repeating order.
Combinatorial novelty dominates continuous novelty; this is the largest single
contributor to "I have not seen this before."

### 3. Novelty repulsion — make it anti-ergodic

The mechanism that dissolves the safety-vs-variety trade instead of trading against it.

Maintain a coarse occupancy grid over a 4-D style vector (8⁴ = 4096 cells, ~16 KB),
incremented at the current state each frame and decayed with a ~10-minute half-life. Add
a weak force down the occupancy gradient.

The state stays inside the hand-tuned beautiful region, but is continuously pushed toward
the parts of it that have not been used recently. This deliberately **breaks ergodicity** —
the system is repelled by its own history. The safe range no longer needs to be wide,
because the system stops re-treading it.

### The real bottleneck

The math *distributes* variety; it does not *create* it. With one warp function, no
amount of clever dynamics helps — the ceiling is set by how many structurally distinct
behaviors actually exist in the shader.

This is **not** a licence to rebuild the six-systems mistake. The distinction:

| Deleted code | This design |
|---|---|
| 6 independent systems | 1 pipeline |
| separate crate, pipeline, shaders each | shared loop, textures, uniforms |
| glued by a compositor | swappable stages |
| ~600 lines per system | `switch` over ~15-line WGSL variants |

Variety at twenty lines apiece is affordable. Variety at a crate apiece is what produced
3,500 lines that rendered nothing.

## Layered reactivity

Two tiers with a strict division of labor — this is what keeps chaos from fighting the music:

- **Chaos governs *how* it looks** — texture, geometry, symmetry order, palette drift, warp
  character. Slow, wandering, never beat-locked.
- **Envelopes govern *when* things happen** — hits, flashes, surges, zoom kicks. Tight,
  deterministic, locked to transients.

Fast tier, per band:

```
attack/release follower:
  τ_attack ≈ 5 ms, τ_release ≈ 150 ms
  e += (x - e) * (1 - exp(-dt / τ))
```

plus onset detection from spectral flux with an adaptive median threshold. Onsets are
discrete events, never chaotic — so every kick lands where it should even while the
underlying geometry is off wandering.

## Screen coupling (music *and* games)

The portrait monitor should feel lit by the main screen.

- **Palette** — dominant hues from the spatial color map become the visual's palette
  anchors, so the second monitor matches the game's lighting. Heavily smoothed (~1 s) so
  scene cuts don't strobe.
- **Exposure** — average brightness sets overall output level. A dark horror game gets a
  dark second monitor.
- **Motion field** — the 128x72 motion grid is downsampled to a coarse vector field and
  used as an *advection force* in the warp pass. An explosion on the left of the game makes
  the visual surge left. This is the piece that makes it read as immersion rather than
  decoration.

Design rule for gaming: the screen coupling should be *slow and low-contrast*. The visual
tracks the game's mood, never its individual frames — anything twitchy in peripheral vision
is actively unpleasant during play.

## Spectral balance (genre differentiation)

Structural switching (above) answers "does it look different over time." It does not
answer "does a bass-heavy track look different from a treble-heavy one" — two songs
with identical RMS envelopes but opposite spectral shape would drive the same amplitude
signal and look the same. That is a real gap, not a nuance: genre character lives in
spectral shape, not loudness.

Fix: group the 7 bands into three weights, normalized to sum to 1 each frame —

```
bass_w   = bands[0] + bands[1]                    (sub-bass, bass)
mid_w    = bands[2] + bands[3] + bands[4]          (low-mid, mid, high-mid)
treble_w = bands[5] + bands[6]                     (treble, brilliance)
```

and let them drive continuous, always-on differentiation, independent of the discrete
warp/injection switches:

| | bass-heavy | mid-heavy | treble-heavy |
|---|---|---|---|
| palette | warm (red/orange) | mid-tone (green/violet) | cool (cyan/white) |
| injection shape | large soft blobs | kaleidoscope-forward | fine sparkle/glints |
| motion character | slow pulses | rotation-forward | fast shimmer |

This is a second, orthogonal source of variety from the structural switches: switching
answers "kind," spectral balance answers "genre." Both are needed — a dubstep track and a
piano ballad should not just be different *frames* of the same look, they should read as
different *moods* of it.

## Song archetypes (configuration, not just modulation) `[DONE]`

Spectral balance turned out to be necessary and not sufficient, and the gap is instructive.
It differentiates *what the palette and the shapes do* while leaving every threshold,
time constant and warp direction global — so the whole pipeline was still tuned against one
kind of music, and material built the other way round fell outside it. Two failures, on
real tracks:

- **A sustained lead over a held chord registered as nothing.** "Grandness" was energy
  against a rolling baseline, and the baseline was a 180-frame ring buffer — one second at
  the frame rate this actually runs at. Sustained music raises its own one-second baseline
  as it swells, so a two-minute guitar solo sat at a ratio of ~1.0 for its entire length and
  the visual treated the biggest passage in the song as ordinary playing.
- **A dynamic-range-preserving master never cleared the absolute gate.** The second half of
  the grandness test was `smoothstep(0.04, 0.12, rms)`, calibrated against a loud modern
  master. An older mix with real headroom is quieter everywhere, so however grand the
  passage, it failed the gate.

Both are the same mistake: a constant that encodes an assumption about the material.

The fix is two-part. **Normalize level against the track**, not against an absolute: a
60-second loudness ceiling (fast attack, slow release) makes every level judgement
master-independent. And **classify the material and blend between parameter sets**, from
slow features — onset rate, crest factor, spectral occupancy — with time constants in the
tens of seconds:

| | pulse | drift | swarm |
|---|---|---|---|
| Material | transient-led | sustain-led | dense, loud |
| Grandness path | hit vs. local context | sustained near the ceiling | both |
| Envelope | 0.4 s / 2.5 s | 1.6 s / 6 s | 0.7 s / 3.5 s |
| Warp | tunnels inward | blooms outward, turns | fast tunnel, counter-rotates |

Weights sum to 1 and every downstream parameter is a linear blend, so this is a continuous
field rather than a mode switch — no snapping, and a track between two archetypes gets a
configuration between them.

Note how this interacts with the safety contract: per-archetype limits are *tighter* than a
global one could be. `drift` can hold grandness near the top for minutes, so its outward
bloom is a sixth of `pulse`'s — a bloom sized for a two-second transient would empty the
frame and keep it empty for the whole solo. A single global constant would have to be safe
for the worst case and would therefore be timid in every other one.

## Render graph

Ping-pong two `RGBA16Float` textures at native portrait resolution.

```
1. INJECT   → draw new energy into HDR target
              audio-driven emitters, spectrum geometry, onset flashes
2. WARP     → sample previous frame through a domain warp:
              rotate + zoom + kaleidoscope (symmetry order from chaos)
              + curl-noise advection + screen motion field
              blend with injection, apply decay
3. COLOR    → palette map (screen anchors + chaotic hue rotation),
              tone map, bloom, dither to kill banding on gradients
4. PRESENT
```

**What is actually built** — inject and warp are *one* pass, not two, because injection
needs the same warped coordinate space as the feedback sample; splitting them adds a
texture round-trip and buys nothing. So the real graph is two passes, not four.

Three items in that sketch are not built and should not be assumed:

- **Curl-noise advection `[PLANNED]`.** What exists is `flow()`, which returns two
  *independent* fbm channels. That is not curl noise: it has nonzero divergence, so it
  creates sources and sinks and the material pools and thins. Curl noise is the
  perpendicular gradient of a single scalar potential and is divergence-free, which is
  what makes it look like fluid rather than like noise pushing things around.
- **Screen motion field `[PLANNED]`** — Phase 3, nothing is wired.
- **Bloom `[PLANNED]`** — the color pass is palette, tonemap, dither.

And the palette is `[SUPERSEDED]`: see "Colour comes from the song, not the clock".

Trivial load for a 5070 at 1080x1920. Budget is nowhere near the constraint; restraint is.

## Crate layout

Keep the existing workspace, add two crates, rewrite one.

```
crates/
  app/                 window, event loop, monitor detect     [keep as-is]
  capture/audio/       REWRITE — real WASAPI loopback
  capture/screen/      keep GDI now; WGC later
  analysis/audio/      trim to: bands, envelopes, onsets, centroid
  analysis/screen/     keep: palette, brightness, motion grid
  dynamics/            NEW, small — Thomas + phases + mapping layer
  render/              NEW — feedback ping-pong graph
```

`dynamics/` should stay under ~300 lines. If it grows past that, something has gone wrong.

## Build phases

**Phase 0 — fix the audio. `[DONE]`** Original diagnosis was wrong: cpal 0.15's WASAPI
backend auto-sets `AUDCLNT_STREAMFLAGS_LOOPBACK` whenever `build_input_stream()` is called
on a render-flow (output) device — confirmed by reading the actual cpal source, not
assumption. `loopback.rs` was never broken. The `Energy: 0.00` in the old log was silence
because nothing was playing, not a capture bug. Verified live: `bass 0.997` while audio
played, mirrored in `rms`. No code change was needed.

**Phase 1 — feedback core. `[DONE, in part]`** Inject + warp + color with hand-tuned
parameters. The `switch`-over-variants part was built and then collapsed — see
"Discrete structural switching" above.

**Shader hot-reload is `[PLANNED]`, not done.** Shaders are `include_str!`d, so they are
baked in at compile time and a tweak costs a rebuild. The paragraph below calling it "not
optional" was right about the cost and wrong about the schedule; a release build is fast
enough that this has not yet been the bottleneck it was predicted to be.

**Phase 2 — dynamics.** Thomas + phases + the mapping layer + slow drift of the safe box,
wired to the Phase 1 parameters, with structural switching on Poincaré crossings. Goal: it
stops repeating.

**Phase 2.5 — novelty repulsion.** The occupancy grid and gradient force. Split out
because it can only be tuned once there is enough structural variety for "unvisited" to
mean something.

**Phase 3 — screen coupling.** Palette, exposure, motion advection. Test against an
actual game.

**Phase 4 — tuning.** Presets, seed control, a debug overlay for live parameter tweaking.

Shader hot-reload was specified to land in Phase 1 and did not — see above. Visual work
is iteration-bound and this is still worth doing, but live-tunable uniforms on keys have
covered most of the need.

## The open gap: authored content vs. emergent content `[PLANNED]`

The governing idea at the top says *"Input perturbs that system. Input does not draw the
image."* That is currently only half true, and it is the largest gap between what this
looks like and what it is trying to be.

The feedback loop is genuinely emergent. The *content* fed into it is not: `layer_orb`,
`layer_spectral_ring`, `layer_point_emitters`, `layer_constellation` and the background
tiers place specific shapes at specific radii in specific tints. That is illustration
parameterized by audio, wrapped in an emergent process. Two consequences, both worth
fixing, in priority order:

### 1. Spatial frequency is too low — reaction–diffusion tier

Every drawn element is a `falloff()`, which is `1 - smoothstep`: a smooth gradient with no
high-frequency content anywhere. `fbm2` is deliberately restricted to low-amplitude
modulation and explicitly never used as per-pixel texture. So the image is soft blobs and
trails, and no amount of good motion makes soft blobs read as the dense, ornamented,
resolves-into-more-detail quality this is aiming at.

The fix is not "add noise texture" — it is to put a **pattern-forming system** in the loop
so the fine structure is generated rather than drawn. Gray–Scott reaction–diffusion is the
obvious candidate: it produces exactly the dense self-organizing organic structure wanted,
it is the canonical instance of beauty out of pure math, and *it is already the
architecture* — a two-species ping-pong over a texture, which is what this renderer is.

Sketch: a third target pair carrying (U, V), advected by the same warp, with feed/kill
rates modulated inside a hand-checked safe range by audio and by the archetype (the
feed/kill plane has well-mapped regions — spots, stripes, worms, mitosis — and moving
between them is a genuine change of *kind*, which is what the abandoned structural
switching was trying to buy). Composite V into the injection rather than replacing it, on
a live key, so it can be A/B'd against the current look.

Care needed: reaction–diffusion is resolution-dependent and frame-rate-sensitive in a way
the rest of this is not. It wants a fixed simulation timestep on an accumulator and its own
fixed-size target, not the swapchain's.

### 2. The actual mathematics of the form constants — log-polar warp

Klüver's form constants — lattices, honeycombs, cobwebs, tunnels, spirals — are what
hallucinated geometry actually consists of, and they have a known mechanism: the
retino-cortical map is approximately log-polar, so a plane wave in cortical coordinates
maps to a spiral, tunnel or ring in the visual field (Bressloff, Cowan, Golubitsky, Thomas
and Wiener, 2001).

Tunnels and spirals appear here *incidentally*, as a side effect of the feedback zoom.
Doing the warp in log-polar space — `(log r, theta)` — would produce them *structurally*,
and would also give the lattice and honeycomb constants, which nothing in the current
pipeline can produce at all. It is a small change to `warp_coord` and it is the highest
ratio of payoff to lines in this document.

Note that the existing invariant 3 (everything periodic in TAU, no discontinuity at the
atan2 branch cut) becomes *more* load-bearing here, not less.

## What went wrong last time

The deleted code had six half-finished visual systems, a mood-attractor state machine, and
a compositor blending them — roughly 3,500 lines that rendered nothing coherent. The
failure was breadth: many systems, none finished.

**This design has one visual system.** Depth over breadth. The variety comes from the
dynamics driving one well-built loop, not from a menu of half-built loops.
