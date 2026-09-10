# denoiser

Pure-Go, streaming spectral denoiser for mono float32 audio. No cgo, so it
cross-compiles cleanly to every platform BirdNET-Go targets. Optional SIMD
acceleration via [`github.com/tphakala/simd`](https://github.com/tphakala/simd).

It removes stationary and slowly varying background noise (wind, rain, traffic
hum, recorder hiss) with short-time spectral gain: each frame is transformed
with a Hann-windowed real FFT, a per-bin gain derived from the estimated noise
power is applied, and the frame is resynthesized by weighted overlap-add. Output
is aligned sample-for-sample with the input.

`denoiser` is the noise-reduction package of
[`go-audio-dsp`](../README.md). It is **beta**: the streaming core, noise
estimation, and strength presets are stable and tested, but their tuning is
still being validated against a real recording corpus, so gain values may change
before v1.

## Install

```sh
go get github.com/tphakala/go-audio-dsp
```

## Usage

The simplest path denoises a clip already in memory. `Denoise` measures the
noise profile from the clip's quietest window and falls back to an adaptive
tracker when there is no distinct quiet region, so it needs nothing but the
audio and a sample rate:

```go
import "github.com/tphakala/go-audio-dsp/denoiser"

out, err := denoiser.Denoise(x, denoiser.Config{SampleRate: 48000})
if err != nil {
    return err
}
// out has len(x) samples, aligned with x.
```

When a noise-only excerpt is available (for example a region the user marked on
a spectrogram), measure the profile from it directly:

```go
out, err := denoiser.DenoiseWithNoise(clip, noiseOnly, denoiser.Config{SampleRate: 48000})
```

A profile can be measured once and reused across clips or runs.
`NoiseProfile.Spectrum` returns the per-bin noise power; `NewNoiseProfile`
rebuilds an equivalent profile from those floats:

```go
d, _ := denoiser.New(denoiser.Config{SampleRate: 48000})
p, _ := d.NoiseProfileFromSamples(noiseOnly)
spectrum := p.Spectrum() // persist these, then later:
p2, _ := denoiser.NewNoiseProfile(spectrum)
out, err := denoiser.DenoiseWithProfile(clip, p2, denoiser.Config{SampleRate: 48000})
```

### Streaming

For live or chunked audio, drive a `Denoiser` directly. Feed consecutive chunks
of any size with `Process` and call `Flush` at the end; the output is aligned
with the input and lags by `Latency()` (FrameSize - HopSize) samples during the
stream, and `Flush` returns the tail so total output equals total input:

```go
d, err := denoiser.New(denoiser.Config{SampleRate: 48000})
if err != nil {
    return err
}
for _, chunk := range chunks {
    out, err := d.Process(chunk)
    if err != nil {
        return err
    }
    sink(out)
}
tail, err := d.Flush()
if err != nil {
    return err
}
sink(tail)
```

`ProcessInto` and `FlushInto` are the allocation-free variants: they write into
a caller-supplied buffer and return the sample count, returning
`ErrBufferTooSmall` (and consuming nothing) if it is too short.

One `Denoiser` serves one stream at a time and is not safe for concurrent use.
Process stereo as two `Denoiser`s, one per channel.

## Strengths and tuning

`Config.Strength` selects a tuned knob set; the zero value is `Medium`. It is
the coarse aggressiveness dial shared across denoise methods (see the
pluggable-method note below), and `ParamsFor` returns the `Params` a strength
maps to so you can start from one and override single knobs.

| Strength | Max reduction | Notes                                              |
| -------- | ------------- | -------------------------------------------------- |
| `Light`  | 6 dB          | Preserves the most detail.                         |
| `Medium` | 12 dB         | Default, balanced.                                 |
| `Heavy`  | 20 dB         | Aggressive over-subtraction; best on steady noise. |

`Config.Params` overrides the strength's knobs entirely for full control: the gain
estimator (`MMSELSA` by default, `Wiener` and `Subtraction` as alternatives),
the residual gain floor, decision-directed SNR smoothing, the a priori SNR
floor, optional smoothing across frequency bins, and the adaptive tracker
window. `FrameSize` and `HopSize` default to the power of two nearest ~21.3 ms
of audio and a quarter of that (75% overlap); both can be set explicitly.

## Noise estimation

The per-bin gain is driven by an estimate of the noise power, from one of three
sources:

1. A profile measured from a caller-selected noise-only excerpt
   (`NoiseProfileFromSamples`).
2. A profile measured automatically from the quietest part of a clip
   (`EstimateNoiseProfile`), which `Denoise` uses.
3. An adaptive minima-controlled recursive averaging (MCRA) tracker that runs
   when no profile is set, so the estimate follows slowly changing noise.

## Methods

This package is the default spectral (STFT) method. It is built to grow:
additional denoise families (an RNNoise-style ML method, a noise gate, a wavelet
method) land as sibling `denoiser/<method>` sub-packages, each satisfying only
the shared `dsp.Processor` (plus `dsp.Flusher`) streaming contract. Method-specific
concepts stay in their own package and never enter a shared interface;
cross-cutting optional behaviour is a capability interface such as `NoiseLearner`,
which a consumer finds with a type assertion; and `Strength` is the coarse
aggressiveness dial shared across methods. No method-selection API ships yet: a
consumer constructs the method it wants directly.

## How it works

1. **Analysis**: a periodic Hann window and a real FFT per frame, at 75% overlap
   by default.
2. **Noise power**: a fixed profile, or the MCRA tracker updated per frame.
3. **Per-bin gain**: MMSE log-spectral amplitude with a decision-directed a
   priori SNR by default, floored so no bin is attenuated past the strength's
   residual floor (nothing gates hard to silence).
4. **Synthesis**: inverse FFT, synthesis window, and weighted overlap-add,
   normalized for exact reconstruction.

Non-finite input samples and noise estimates are guarded throughout, so a NaN or
Inf in the input cannot poison the stream.

## Correctness

- **Synthetic quality harness**: tones and pink/white noise at known SNRs,
  asserting noise reduction and signal preservation across strengths.
- **FFmpeg `afftdn` as a reference oracle**: tests compare against ffmpeg's
  spectral denoiser and skip automatically when ffmpeg is not on `PATH`.

```sh
go test ./...          # unit + synthetic quality + ffmpeg cross-validation
go test -bench=. ./... # throughput benchmarks
```

A real-corpus A/B comparison against `afftdn` is planned via the `ab` Taskfile
target, to be enabled once a `testdata/corpus/*.wav` set and a pinned ffmpeg are
in place.

## License

Released under the **Apache License 2.0**. See [LICENSE](../LICENSE).
