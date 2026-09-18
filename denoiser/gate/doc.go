// Package gate is a pure-Go, streaming spectral soft-gate waveform denoiser: a
// second, lighter denoise method that lives beside the flagship spectral
// denoiser under denoiser/gate.
//
// It is built for one job: suppress continuous outdoor background noise (wind,
// rain, traffic hum, insect or cicada drone) in field recordings while
// preserving bird song, which is tonal, harmonic and frequency-modulated. It is
// POST-PROCESSING ONLY: waveform in, mono float32 waveform out, for the clip,
// export and playback path. It is a normal dsp.Processor that produces audio,
// never a feature stage, and must never run before analysis or in a detector's
// feature path.
//
// How it works. Each Hann-windowed frame is transformed with a real FFT; a
// per-bin soft mask is derived from how far the bin's power sits above an
// estimated per-bin noise floor (a smooth sigmoid knee around ThresholdDB, of
// width TransitionDB, mapped into a residual gain bounded below by the
// MaxAttenuationDB floor); the mask is
// smoothed across frequency and, optionally, time; and the frame is
// resynthesized by overlap-add. The whole per-bin apply path is elementwise, so
// it takes github.com/tphakala/simd's AVX or NEON tiers where available and a
// portable Go fallback elsewhere, with no scalar transcendental in the hot loop.
// This is the concrete difference from the flagship, whose per-bin gain is a
// scalar float64 loop through the exponential integral: the gate is, by
// construction, lighter and faster on low-power hosts and gentle on tonal song.
// It does not replace the flagship.
//
// Noise floor. Two estimators feed the mask. The primary path is a learned floor
// measured from a noise-only excerpt (LearnNoise, satisfying the
// denoiser.NoiseLearner capability) or set directly (SetNoiseFloor); this is the
// path the method is tuned for. When no floor is learned, a blind fallback
// tracks a per-bin rolling median of log-power over a short window, which is the
// right estimator for gapless continuous noise (a minimum tracker would never
// open on steady rain or cicada drone). The blind estimator keeps a small
// sliding histogram per bin (about 170 KB of state at the 1024/256 default), and
// passes audio through unchanged during its brief warm-up.
//
// A Gate streams: call Process (or ProcessInto) with consecutive chunks of any
// size and Flush (or FlushInto) at the end. Denoise, DenoiseWithNoise and
// DenoiseWithFloor wrap that for a clip already in memory. One Gate serves one
// stream at a time and is not safe for concurrent use; process stereo as two
// Gates, one per channel. Strength (Light, Medium, Heavy) selects a tuned knob
// set shared with the flagship's vocabulary; Params overrides every knob.
package gate
