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
178 fps | rms 0.088 | bass 0.915 | mid 0.208 | treble 0.151 | grand 0.03 | organic 0.24 | core 0.45
```

`grand` and `organic` are the two derived values worth watching — if the visuals feel
wrong, that line usually says which input is responsible.

## How it reacts

Audio is analysed across three timescales (`crates/analysis/audio_analysis`): instantaneous
RMS/centroid/flux, short-term attack and decay, and long-term rolling energy and
volatility, plus seven frequency bands. Three derived signals drive most of the look:

- **Spectral balance** (bass/mid/treble weights) tints the palette, so a bassy track and a
  trebly one read as different moods of the same visual.
- **Grandness** — loud relative to *this track's own rolling baseline*, and loud in
  absolute terms. Both gates matter: the ratio alone fires on every note in a quiet
  passage, the absolute level alone fires on all of a loud track. It gates the background's
  grand tier and blooms the whole image outward.
- **Song feel** — how fluid the current track is, from level churn, timbral movement and
  spectral balance. It maps to `organic`, which scales every departure from strict
  geometry. Built deliberately from slow quantities, not from `attack`: looseness
  flickering at beat rate reads as the image glitching rather than as the music having a
  character. ~8s to settle, ~16s to relax.

Values that were found by ear, not derived, are marked as such in the code with the range
they were swept over.

## Working on the shaders

`shaders/render/feedback.wgsl` is where the image comes from, and it has three invariants
documented at the top of the file. They are not style preferences — each one was written
after the corresponding artifact showed up on screen:

1. **Injection is energy-per-second** and is multiplied by `dt`. A per-frame constant makes
   brightness frame-rate dependent and saturates the loop to white.
2. **The warp applies a per-frame step**, never a function of absolute `time` that
   compounds. The feedback texture already carries the accumulated history.
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
