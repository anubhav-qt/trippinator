// Trippinator — Generative Audiovisual Organism
// Real-time wave dynamics, domain-warped fields, harmonic interference, portrait-native composition,
// and dynamic state-integrated (non-timer) acoustic phase & color synthesis.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct AudioVisualUniforms {
    // GlobalState (12 floats = 48 bytes)
    energy: f32,
    tension: f32,
    chaos: f32,
    coherence: f32,
    turbulence: f32,
    density: f32,
    flow: f32,
    compression: f32,
    expansion: f32,
    entropy: f32,
    chroma: f32,
    temporal_memory: f32,

    // Audio frequency bands (8 floats = 32 bytes)
    sub_bass: f32,
    bass: f32,
    low_mid: f32,
    mid: f32,
    high_mid: f32,
    treble: f32,
    brilliance: f32,
    rms: f32,

    // Audio dynamics & spectral character (4 floats = 16 bytes)
    attack: f32,
    decay: f32,
    energy_velocity: f32,
    spectral_centroid: f32,

    // Screen color & palette (4 floats = 16 bytes)
    screen_r: f32,
    screen_g: f32,
    screen_b: f32,
    screen_saturation: f32,

    // Dynamic Integrated Phases & Viewport (4 floats = 16 bytes)
    flow_phase: f32,
    color_phase: f32,
    harmonic_phase: f32,
    dt: f32,
};

@group(0) @binding(0)
var<uniform> u: AudioVisualUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Procedural hash and noise
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn smooth_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u_smooth = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u_smooth.x), mix(c, d, u_smooth.x), u_smooth.y);
}

// Curl noise for fluid-like vector fields
fn curl_noise(p: vec2<f32>) -> vec2<f32> {
    let eps = 0.04;
    let n0 = smooth_noise(p + vec2<f32>(0.0, eps));
    let n1 = smooth_noise(p - vec2<f32>(0.0, eps));
    let n2 = smooth_noise(p + vec2<f32>(eps, 0.0));
    let n3 = smooth_noise(p - vec2<f32>(eps, 0.0));

    let dx = (n2 - n3) / (2.0 * eps);
    let dy = (n0 - n1) / (2.0 * eps);

    return vec2<f32>(dy, -dx);
}

// Cosine color palette generator (Inigo Quilez parameterization)
fn palette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.2831853 * (c * t + d));
}

// OKLCH to Linear RGB conversion for perceptual color gradients
fn oklch_to_rgb(l: f32, c: f32, h: f32) -> vec3<f32> {
    let a = c * cos(h);
    let b = c * sin(h);

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    let r =  4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
    let g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
    let bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;

    return clamp(vec3<f32>(r, g, bl), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Dynamic state-integrated organism color (driven by acoustic metabolism instead of fixed clocks)
fn get_organism_color(t: f32, energy: f32, chroma: f32, chaos: f32, color_phase: f32) -> vec3<f32> {
    // Desktop ambient color influence
    let screen_tint = vec3<f32>(u.screen_r, u.screen_g, u.screen_b);
    let ambient_shift = (screen_tint - 0.5) * (0.3 * u.screen_saturation);

    // Spectral temperature: higher centroid & energy = warmer, hot ionization; lower = deep abyssal blues
    let spectral_temp = (u.spectral_centroid - 0.3) * 0.4 + energy * 0.3;

    let a = vec3<f32>(0.5, 0.5, 0.5) + ambient_shift;
    let b = vec3<f32>(0.5, 0.5, 0.5) * (0.6 + chroma * 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0) + vec3<f32>(spectral_temp * 0.2, -spectral_temp * 0.1, spectral_temp * 0.3);

    // Orbital phase vector rotates with the organism's dynamic integrated color phase
    let orbital_phase = vec3<f32>(
        0.00 + sin(color_phase * 0.5) * 0.3 + color_phase * 0.2,
        0.33 + cos(color_phase * 0.6) * 0.3 + energy * 0.25,
        0.67 + sin(color_phase * 0.4 + 1.57) * 0.3 - chaos * 0.2
    );

    let d = orbital_phase + vec3<f32>(chaos * 0.2, spectral_temp, -chaos * 0.15);

    return palette(t, a, b, c, d);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Normalize coordinates: center at (0, 0), correct for portrait aspect ratio
    let aspect = 1920.0 / 1080.0;
    var uv = (in.uv * 2.0 - 1.0);
    uv.y *= aspect;

    let energy = u.energy;
    let bass = u.bass + u.sub_bass * 1.4;
    let treble = u.treble + u.brilliance * 0.9;
    let mid = u.mid + u.low_mid;

    let flow_p = u.flow_phase;
    let color_p = u.color_phase;
    let harm_p = u.harmonic_phase;

    // 2. Resting breath pulse (organism metabolism floor)
    let resting_breath = sin(flow_p * 0.3) * 0.08 + sin(flow_p * 0.07 + 1.0) * 0.05;
    let effective_energy = max(energy, 0.06) + resting_breath;

    // 3. Portrait-native domain warp: vertical flow currents + audio-reactive curl field
    let flow_coord = uv * (1.6 + u.density * 0.9) + vec2<f32>(0.0, -flow_p * 0.4);

    // Multi-octave curl noise turbulence
    let curl1 = curl_noise(flow_coord * 0.7) * (0.35 + u.turbulence * 0.7 + u.chaos * 0.45);
    let curl2 = curl_noise(flow_coord * 1.5 + curl1 * 0.8) * (0.2 + bass * 0.65);
    let curl3 = curl_noise(flow_coord * 3.2 + curl2) * (0.08 + treble * 0.45);

    var p = uv + curl1 * 0.4 + curl2 * 0.25 + curl3 * 0.15;

    // 4. Central vertical resonance spine (gravitational breathing channel)
    let spine_x = p.x * (2.0 + u.compression * 1.6);
    let spine_y = p.y * 0.75;
    let r_center = length(vec2<f32>(spine_x, spine_y));

    // Vertical standing waves along the spine
    let vertical_wave = sin(p.y * (3.0 + effective_energy * 4.0) - flow_p * 0.8) * (0.25 + bass * 0.65);
    let central_pulse = sin(r_center * (5.0 + bass * 6.5) - flow_p * 1.8 + vertical_wave) * (0.3 + bass * 0.75);

    // 5. Multi-layer Harmonic Wave Interference with per-layer chromatic coupling
    var wave_accum = 0.0;
    var caustic_accum = 0.0;
    var layer_color_accum = vec3<f32>(0.0);

    let num_layers = 5;
    for (var i = 0; i < num_layers; i++) {
        let fi = f32(i);
        let freq_scale = pow(1.618, fi) * (1.4 + u.tension * 0.75);
        let phase = harm_p * (0.5 + fi * 0.25) + fi * 1.256;

        let angle = 1.5708 + sin(harm_p * 0.1 + fi * 1.05) * (0.35 + u.chaos * 0.8);
        let dir = vec2<f32>(cos(angle), sin(angle));

        // Spatial standing interference
        let d = dot(p, dir) * freq_scale;
        let standing_wave = sin(d + phase) * cos(p.x * freq_scale * 0.7 - phase * 0.6 + vertical_wave * 0.5);

        // Micro-caustics on higher frequency harmonics
        let caustic_freq = freq_scale * (2.8 + u.density * 1.2);
        let c_wave = abs(sin(dot(p + curl1, vec2<f32>(sin(d * 0.5), cos(d * 0.5))) * caustic_freq + phase * 1.8));
        let sharp_caustic = pow(clamp(1.0 - c_wave, 0.0, 1.0), 6.5) * (0.25 + treble * 1.5);

        let weight = 1.0 / (1.0 + fi * 0.55);
        wave_accum += standing_wave * weight;
        caustic_accum += sharp_caustic * weight;

        // Layer-specific harmonic hue rotation driven by integrated color phase
        let layer_hue_angle = (color_p * 0.4 + fi * 1.256 + standing_wave * 0.5);
        let layer_lch_color = oklch_to_rgb(0.65 + sharp_caustic * 0.3, 0.22 + u.chroma * 0.14, layer_hue_angle);
        layer_color_accum += layer_lch_color * abs(standing_wave) * weight * 0.35;
    }

    // 6. Opposing behavior integration:
    // Deep bass expands outward; treble creates razor-sharp luminous filaments
    let bloom_mask = smoothstep(1.9 + u.expansion * 1.2, 0.1, r_center) * (0.35 + bass * 0.8);
    let filament_intensity = abs(wave_accum + central_pulse * 0.45);
    let sharp_filaments = pow(clamp(1.0 - filament_intensity * 0.28, 0.0, 1.0), 3.2);

    // 7. Dynamic Organic Color Synthesis
    let spatial_phase = (p.y * 0.35 + wave_accum * 0.25 + central_pulse * 0.12);
    var base_col = get_organism_color(spatial_phase, effective_energy, u.chroma, u.chaos, color_p);

    // Blend base palette with harmonic layer colors
    var col = mix(base_col, layer_color_accum, 0.45);

    // 8. Bioluminescent & Radiative Emission Accents
    // Core Abyssal Cyan / Lavender Glow
    let core_glow = vec3<f32>(0.15, 0.75, 1.0) * caustic_accum * (0.7 + treble * 1.7);

    // Deep Magenta / Rose Resonance
    let deep_accent = vec3<f32>(0.95, 0.18, 0.55) * sharp_filaments * (0.45 + u.tension * 0.9);

    // Molten Solar Gold Bass Pulse
    let gold_bass = vec3<f32>(1.0, 0.65, 0.25) * pow(clamp(central_pulse, 0.0, 1.0), 1.8) * (0.35 + bass * 0.95);

    // Celestial Emerald Filament Shimmer
    let emerald_filament = vec3<f32>(0.1, 0.95, 0.65) * pow(clamp(vertical_wave, 0.0, 1.0), 2.2) * (0.3 + mid * 0.75);

    col = col * (0.2 + sharp_filaments * 0.8) * (0.5 + bloom_mask * 0.85);
    col += core_glow + deep_accent + gold_bass + emerald_filament;

    // 9. Spectral Dispersion (Chromatic Aberration on Transients)
    let dispersion = (u.attack * 0.035 + abs(u.energy_velocity) * 0.025 + u.chaos * 0.02);
    let col_r = get_organism_color(spatial_phase + dispersion, effective_energy, u.chroma, u.chaos, color_p).r * 0.25;
    let col_b = get_organism_color(spatial_phase - dispersion, effective_energy, u.chroma, u.chaos, color_p).b * 0.25;
    col.r += col_r;
    col.b += col_b;

    // 10. Vertical Portrait Lighting & Vignette
    let vertical_gradient = smoothstep(-1.2, 1.2, uv.y);
    col *= mix(vec3<f32>(0.85, 0.9, 1.1), vec3<f32>(1.1, 0.95, 0.85), vertical_gradient);

    let vig_dist = length(in.uv - vec2<f32>(0.5, 0.5));
    let vignette = smoothstep(0.88, 0.22, vig_dist);
    col *= vignette;

    // 11. Filmic Tonemapping (ACES-curve shoulder + gamma correction)
    let tonemapped = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    let final_color = pow(clamp(tonemapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_color, 1.0);
}
