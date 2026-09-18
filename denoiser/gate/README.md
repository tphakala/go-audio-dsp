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

With no excerpt, `Denoise` uses the blind rolling-median floor. Unlike the
flagship it does not run a quietest-window pre-scan (its blind path is designed
for gapless continuous noise), and audio passes through unchanged during a brief
warm-up:

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
frequency (`FreqSmoothBins`) and time (`TimeSmoothFrames`, which adds lookahead to
`Latency()`), and the blind window (`FloorWindowSec`). `FrameSize` and `HopSize`
default to the power of two nearest ~21.3 ms of audio and a quarter of that (75%
overlap), matching the flagship.

## How it works

1. **Analysis**: a periodic Hann window and a real FFT per frame, at 75% overlap
   by default.
2. **Noise floor**: a learned per-bin floor (`LearnNoise` / `SetNoiseFloor`), or
   the blind per-bin rolling median otherwise.
3. **Soft mask**: per bin, `power/floor` -> log10 -> a sigmoid knee at
   `ThresholdDB` of width `TransitionDB` -> a gain in `[floor, 1]`. Optionally
   smoothed across frequency (fewer isolated musical-noise bins) and time (the
   memoryless sigmoid would otherwise flutter).
4. **Synthesis**: inverse FFT, synthesis window, and weighted overlap-add,
   normalized for exact reconstruction.

Non-finite input and floor values are guarded throughout: a NaN or Inf cannot
poison the stream or corrupt the blind tracker's sliding window.

## Correctness

- **Framing**: unity-floor identity, latency accounting, exact flush drain, and
  bit-exact chunk invariance across arbitrary chunk splits.
- **Mask**: a float64 scalar reference cross-checks the SIMD chain on both the
  SIMD and pure-Go tiers; the gate attenuates noise while preserving a strong
  tone.
- **Blind floor**: matches the learned mean-power floor on continuous noise
  (median-to-mean corrected) and ignores a sparse loud tone.
- **Zero-allocation** steady state and a gating benchmark.

```sh
go test ./denoiser/gate/            # unit, framing, and floor tests
SIMD_DISABLE=all go test ./denoiser/gate/   # pure-Go tier (scalar parity)
go test -bench=. ./denoiser/gate/   # throughput and per-call allocations
```

A shared `afftdn` reference oracle and real-corpus A/B comparison across both
denoise methods are planned once the flagship's test harness is extracted to an
internal package.

## License

Released under the **Apache License 2.0**. See [LICENSE](../../LICENSE).
