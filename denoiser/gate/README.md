# denoiser/gate

Pure-Go, streaming spectral **soft-gate** waveform denoiser for mono float32
audio. No cgo, so it cross-compiles cleanly to every platform BirdNET-Go targets.
Optional SIMD acceleration via [`github.com/tphakala/simd`](https://github.com/tphakala/simd).

`gate` is a second, lighter denoise method beside the flagship spectral
[`denoiser`](../README.md), built for one job: suppress continuous outdoor
background noise (wind, rain, traffic hum, insect or cicada drone) in field
recordings while preserving bird song. It is **post-processing only**: waveform
in, waveform out, for the clip, export and playback path. It is a normal
`dsp.Processor` that produces audio, never a feature stage, and must not run
before analysis or in a detector's feature path.

It is **beta**: the streaming core, both noise-floor estimators, and the strength
knob sets are stable and tested, but the knob values are reasoned starting points
still to be validated against a real recording corpus, so they may change before
v1.

## Why a gate, not the flagship

The flagship's per-bin gain is a scalar float64 loop through the exponential
integral, and its noise tracker needs quiet gaps to open. This method is:

- **Lighter and faster on low-power hosts**: the whole per-bin apply path is
  elementwise SIMD (divide, log, affine, sigmoid, affine), with no scalar
  transcendental in the hot loop.
- **Gentle on tonal song by construction**: a soft sigmoid knee opens on any bin
  whose power sits above the noise floor, so harmonics and frequency-modulated
  calls pass largely untouched.
- **Correct for gapless noise**: its blind floor is a per-bin rolling **median**,
  the right estimator for steady rain or cicada drone, where a minimum tracker
  never opens.
  `Denoise` caps it at the clip's quietest-window level plus 3 dB, so continuous
  song is not mistaken for noise when the clip has a usable quiet window; noise
  elsewhere in the clip that sits more than a few dB above that window is reduced
  less.

It does not replace the flagship, which reduces broadband noise more
aggressively.

## Install

```sh
go get github.com/tphakala/go-audio-dsp
```

## Usage

The primary path learns a noise floor from a noise-only excerpt (a clip's quiet
lead-in, or a region the user marked), then gates the clip:

```go
import "github.com/tphakala/go-audio-dsp/denoiser/gate"

out, err := gate.DenoiseWithNoise(clip, noiseOnly, gate.Config{SampleRate: 48000})
if err != nil {
    return err
}
// out has len(clip) samples, aligned with clip.
```

With no excerpt, `Denoise` uses the blind rolling-median floor, bounded by a
quietest-window pre-scan: it measures the clip's quietest 0.5 s window with the
flagship's `EstimateNoiseProfile` and caps each bin's floor at that power plus
3 dB, so continuous song that fills a bin cannot become the floor, while noise
elsewhere in the clip that sits more than a few dB above that window is reduced
less. The plain median is kept when the clip has about 1 s or more of non-silent
audio and no distinctly quiet window, when the measured stretch holds digital
silence (no 0.5 s window free of it exists, or a `FrameSize` longer than 0.5 s
reaches into it), or when the clip is shorter than one frame; otherwise a clip
with less non-silent audio takes its quietest window as is. Audio passes through
unchanged during a brief warm-up at the start of the clip:

```go
out, err := gate.Denoise(clip, gate.Config{SampleRate: 48000})
```

A floor measured once (in the same `|RFFT(hann*x)|^2` scale the flagship's
`NoiseProfile.Spectrum` returns) can be applied to every clip from the same
station with `DenoiseWithFloor`, or read back from a gate with `NoiseFloor`.

### Streaming

Drive a `Gate` directly for live or chunked audio. Feed consecutive chunks of any
size with `Process` and call `Flush` at the end; the output is aligned with the
input and lags by `Latency()` samples during the stream:

```go
g, err := gate.New(gate.Config{SampleRate: 48000})
if err != nil {
    return err
}
if err := g.LearnNoise(noiseOnly); err != nil { // optional; else blind tracking
    return err
}
for _, chunk := range chunks {
    out, err := g.Process(chunk)
    if err != nil {
        return err
    }
    sink(out)
}
tail, err := g.Flush()
if err != nil {
    return err
}
sink(tail)
```

`ProcessInto` and `FlushInto` are the allocation-free variants: they write into a
caller-supplied buffer and return the sample count, returning `ErrBufferTooSmall`
(and consuming nothing) if it is too short. A buffer of `MaxOutputLen(len(in))`
always fits `ProcessInto`, and one of `Latency()+HopSize()` always fits
`FlushInto`.

`Gate` implements `denoiser.NoiseLearner`, so a consumer holding a `dsp.Processor`
can learn a floor without depending on the concrete type. One `Gate` serves one
stream at a time and is not safe for concurrent use; process stereo as two
`Gate`s, one per channel.

## Strengths and tuning

`Config.Strength` selects a tuned knob set; the zero value is `Medium`. It is the
coarse aggressiveness dial shared with the flagship.

| Strength | Max reduction | Threshold | Notes                                  |
| -------- | ------------- | --------- | -------------------------------------- |
| `Light`  | 6 dB          | 3 dB      | Preserves the most detail.             |
| `Medium` | 12 dB         | 5 dB      | Default, balanced.                     |
| `Heavy`  | 20 dB         | 8 dB      | Most aggressive; most time smoothing.  |

`Config.Params` overrides every knob: the residual gain floor (`MaxAttenuationDB`;
0 disables the gate, +Inf allows full gating), the gate threshold above the floor
(`ThresholdDB`), the soft-knee width (`TransitionDB`), smoothing widths across
frequency (`FreqSmoothBins`, applied to the per-bin SNR) and time (`TimeSmoothFrames`, which adds lookahead to
`Latency()`), and the blind window (`FloorWindowSec`). `FrameSize` and `HopSize`
default to the power of two nearest ~21.3 ms of audio and a quarter of that (75%
overlap), matching the flagship.

## How it works

1. **Analysis**: a periodic Hann window and a real FFT per frame, at 75% overlap
   by default.
2. **Noise floor**: a learned per-bin floor (`LearnNoise` / `SetNoiseFloor`), or
   the blind per-bin rolling median otherwise (capped by a quiet-window
   pre-scan in `Denoise`).
3. **Soft mask**: per bin, `power / noise-floor`, smoothed across frequency
   (fewer isolated musical-noise bins; done on the SNR so a strong narrowband
   call is not averaged down by its closed neighbours, while one near
   `ThresholdDB` still is) -> log10 -> a sigmoid knee at
   `ThresholdDB` of width `TransitionDB` -> a gain in the residual-gain range
   `[gFloor, 1]`, where `gFloor` is set by `MaxAttenuationDB`. Optionally
   smoothed across time (the memoryless sigmoid would otherwise flutter).
4. **Synthesis**: inverse FFT, synthesis window, and weighted overlap-add,
   normalized for exact reconstruction.

The blind tracker and the learned floor are guarded against non-finite values: a
NaN or Inf in the input cannot corrupt the sliding-window histogram or the stored
floor, so it cannot poison later output. Output samples overlapping a non-finite
input sample are not themselves sanitized (garbage in, garbage out, localized).

## Correctness

- **Framing**: unity-floor identity, latency accounting, exact flush drain, and
  bit-exact chunk invariance across arbitrary chunk splits.
- **Mask**: a float64 scalar reference cross-checks the SIMD chain (including
  the SNR smoothing) on both the SIMD and pure-Go tiers; the gate attenuates
  noise while preserving a strong tone, including one centered exactly on a bin.
- **Blind floor**: matches the learned mean-power floor on continuous noise
  (median-to-mean corrected) and ignores a sparse loud tone.
  `Denoise` keeps continuous on-bin song that fills most of the median window
  (the cap), caps nothing and equals the plain blind stream when a clip long
  enough to be judged has no distinct quiet window, when its quiet window holds
  digital silence, or when the clip is under one frame, and the capped path is
  chunk-invariant and allocation-free.
- **Zero-allocation** steady state and a gating benchmark.

```sh
go test ./denoiser/gate/            # unit, framing, and floor tests
SIMD_DISABLE=all go test ./denoiser/gate/   # pure-Go tier (scalar parity)
go test -bench=. ./denoiser/gate/   # throughput and per-call allocations
```

`ab_test.go` measures the gate against ffmpeg `afftdn` and the flagship denoiser
on synthetic clips, a real-clip corpus and a recorded noise bed, through the
shared `internal/audiotest` helpers.

## License

Released under the **Apache License 2.0**. See [LICENSE](../../LICENSE).
