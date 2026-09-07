# loudnorm

Pure-Go, two-pass loudness normalization of in-memory PCM to the
**EBU R 128 / ITU-R BS.1770-4** standard. No cgo, so it cross-compiles cleanly
to every platform BirdNET-Go targets. Optional SIMD acceleration via
[`github.com/tphakala/simd`](https://github.com/tphakala/simd).

It measures gated integrated loudness (LUFS) and true peak (dBTP), then applies
a single linear gain so audio reaches a target loudness without its true peak
exceeding a ceiling. Linear gain preserves dynamics and never pumps, which suits
field and nature recordings.

`loudnorm` is the loudness-normalization package of
[`go-audio-dsp`](../README.md).

## Why

A survey of the Go ecosystem found no library that is pure-Go, works at
arbitrary sample rates, and normalizes an in-memory buffer: `exaring/ebur128` is
pure-Go but measure-only and 48 kHz-locked; the other two wrap C `libebur128`
via cgo. So this implements BS.1770-4 from scratch. See [DESIGN.md](DESIGN.md).

## Install

```sh
go get github.com/tphakala/go-audio-dsp
```

## Usage

The primary BirdNET-Go path is 48 kHz, 16-bit, mono, which is what
`DefaultOptions` assumes:

```go
import "github.com/tphakala/go-audio-dsp/loudnorm"

// pcm is interleaved 16-bit PCM, modified in place.
opts := loudnorm.DefaultOptions() // 48 kHz mono, -23 LUFS, -1.0 dBTP ceiling
res, err := loudnorm.NormalizeInt16(pcm, opts)
if err != nil {
    return err
}
// res.GainDB is the gain applied; res.OutputLUFS the resulting loudness;
// res.PeakLimited reports whether the true-peak ceiling capped the gain.
```

Other rates, channel counts, and float32 buffers are supported too:

```go
opts := loudnorm.DefaultOptions()
opts.SampleRate = 44100
opts.Channels = 2
opts.TargetLUFS = -16 // louder target for streaming/playback
_, err := loudnorm.NormalizeFloat32(pcmFloat32, opts)
```

Measurement only (no modification):

```go
m, err := loudnorm.MeasureInt16(pcm, 48000, 1)
// m.IntegratedLUFS, m.TruePeakDBTP
```

For streaming or advanced use, drive the `Meter` directly. It accepts
`AddFloat64`, `AddFloat32`, or `AddInt16` blocks (converted inline, no
full-length scratch buffer):

```go
meter := loudnorm.NewMeter(48000, 1)
meter.AddInt16(block1)
meter.AddInt16(block2)
lufs := meter.IntegratedLoudness()
tp := meter.TruePeakDBTP()
```

## Performance

Measurement and normalization are allocation-free in steady state. The
`Measure*`/`Normalize*` helpers pool meters internally, so repeated calls at a
fixed sample rate and channel count do not allocate after warm-up. For an
explicit zero-allocation loop, reuse one meter with `Reset`:

```go
m := loudnorm.NewMeter(48000, 1)
for _, clip := range clips {
    m.Reset()
    m.AddInt16(clip)
    lufs, tp := m.IntegratedLoudness(), m.TruePeakDBTP()
    _ = lufs
    _ = tp
}
```

The true-peak oversampler runs as batched SIMD convolutions via
`github.com/tphakala/simd`, and true-peak scratch is bounded regardless of clip
length, so memory does not scale with input size.

## How it works

1. **K-weighting** pre-filter (two biquads), coefficients computed for the
   actual sample rate via the bilinear transform.
2. **Gated integrated loudness**: 400 ms blocks at 75% overlap, absolute gate at
   -70 LUFS and relative gate at -10 LU below the gated mean.
3. **True peak** via 4x Kaiser-windowed polyphase oversampling, in dBTP.
4. **Linear gain** `target - measured`, reduced if needed to keep the true peak
   at or below the ceiling.

## Correctness

- **Known-answer tests** from EBU Tech 3341 (e.g. a stereo 1 kHz sine at
  -23 dBFS reads -23.0 LUFS; absolute-gate silence rejection).
- **FFmpeg as a reference oracle**: tests synthesize PCM, run
  `ffmpeg -af ebur128`, and assert our integrated loudness and true peak agree
  within tolerance, across tones, white noise, two-segment gating, mono/stereo,
  and 44.1 kHz. They skip automatically when ffmpeg is not on `PATH`. An
  end-to-end test normalizes with this library and has ffmpeg confirm the result
  hits target and respects the ceiling.

```sh
go test ./...          # unit + ffmpeg cross-validation (if ffmpeg present)
go test -bench=. ./... # throughput benchmarks
```

## License

Released under the **Apache License 2.0**. See [LICENSE](../LICENSE).
