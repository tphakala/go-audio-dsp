# go-audio-dsp

Pure-Go, zero-cgo audio DSP building blocks for offline and streaming audio
processing. Part of the `tphakala` go-audio family (go-flac, go-opus, go-aac,
go-mp3, go-wav, go-audio-stream, go-audio-resampler). Everything cross-compiles
cleanly to every platform BirdNET-Go targets, with optional SIMD acceleration
via [`github.com/tphakala/simd`](https://github.com/tphakala/simd).

## Packages

| Package | Status | What it does |
| ------- | ------ | ------------ |
| [`loudnorm`](loudnorm/) | stable | EBU R 128 / ITU-R BS.1770-4 loudness normalization of in-memory PCM (LUFS measurement, true-peak limiting, linear gain). |
| [`denoiser`](denoiser/) | beta | Spectral audio denoiser (STFT, measured or adaptive noise profile, Wiener / MMSE-LSA gain, overlap-add reconstruction). Replaces ffmpeg `afftdn` on the BirdNET-Go clip path. |

More processors may follow (high-pass, gain, and related building blocks) as the
need arises.

## Install

```sh
go get github.com/tphakala/go-audio-dsp
```

Import the package you need:

```go
import "github.com/tphakala/go-audio-dsp/loudnorm"
```

See each package's README for usage: [`loudnorm`](loudnorm/README.md) and [`denoiser`](denoiser/README.md).

## Design goals

- **Pure Go, no cgo.** Clean cross-compilation to every target; no C toolchain.
- **Arbitrary sample rates.** Coefficients and windows computed for the actual
  rate, not locked to 48 kHz.
- **Low allocation.** Allocation-free steady state where it matters; buffers are
  reused and scratch is bounded regardless of input length.
- **Optional SIMD.** Hot paths accelerate through `github.com/tphakala/simd`
  with a scalar fallback that produces the same result.
- **Verified against a reference.** ffmpeg is used as a cross-validation oracle
  in tests, not as a runtime dependency.

## License

Released under the **Apache License 2.0**. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
