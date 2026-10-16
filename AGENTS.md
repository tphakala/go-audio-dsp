# AGENTS.md

Orientation for coding agents and automated reviewers working on this repository. README.md and the per-package READMEs are the user-facing reference (usage, package status); this file is the map for changing and reviewing the code.

## What this is

`github.com/tphakala/go-audio-dsp` (root package `dsp`) is a set of pure-Go audio DSP building blocks for offline and streaming processing: denoising, equalization, gain, loudness normalization, STFT, spectrograms and mel features. It is part of the tphakala go-audio family (go-flac, go-opus, go-aac, go-mp3, go-m4a, go-audio-capture, go-audio-stream) and BirdNET-Go is its main consumer, where it replaces ffmpeg and sox subprocesses on the clip and export paths. It aims to:

1. **Build anywhere Go builds.** No cgo, no external binaries at runtime. Every package cross-compiles to every target BirdNET-Go ships on, including 32-bit ARM and little-endian MIPS.
2. **Be fast without being fragile.** Hot loops go through `github.com/tphakala/simd` (AVX, NEON) with a portable Go fallback that gives the same result. Streaming blocks allocate nothing in steady state.
3. **Be measurably correct.** Output is checked against reference implementations (ffmpeg `ebur128` and `afftdn`, librosa mel filterbanks, scipy Dolph-Chebyshev windows), not just against itself. ffmpeg is a test oracle, never a runtime dependency.

Packages:

| Package | Status | Role |
|---|---|---|
| `dsp` (root) | | `Processor` and `Flusher` streaming contract, shared sentinels `ErrBufferTooSmall` and `ErrInvalidConfig`, `FactorFromDB` |
| `loudnorm` | stable | EBU R 128 / BS.1770-4 two-pass loudness normalization of in-memory PCM (whole clip, not a `Processor`) |
| `pcm` | stable | int16 <-> float32 conversion at the chain edges, saturating `ScaleInt16` |
| `denoiser` | beta | Spectral denoiser (measured or adaptive noise profile, Wiener / MMSE-LSA gain, overlap-add), `NoiseProfileEstimator` |
| `denoiser/gate` | beta | Spectral soft-gate waveform denoiser, the lighter second denoise method |
| `equalizer` | beta | RBJ biquad filter chain as a streaming block, plus frequency response for a UI curve; Butterworth band-pass and caller-supplied biquad sections as a `Filter` |
| `gain` | beta | Fixed decibel gain as a streaming block, saturating int16 entry points |
| `stft` | beta | Shared STFT layer: whole-clip `Plan`, streaming `Analyzer`, overlap-add `Synthesizer`, windows |
| `spectrogram` | beta | Magnitude, power or dB spectrogram matrix, whole clip or streaming columns |
| `mel` | beta | Per-frame (log-)mel columns; normalization, stacking and tensor layout are the consumer's job |
| `octave` | beta | Streaming 1/3-octave band level meter (a meter, not a `Processor`): ISO 266 / IEC 61260-1 bands, mean square per period, built on `equalizer` filters (RBJ BandPass by default, an optional Butterworth band-pass profile) |
| `internal/dspshared` | | Helpers shared by `denoiser` and `denoiser/gate` (floor guard, geometry defaults, emission arithmetic, the overlap-add stream driver) so a fix lands once |
| `internal/audiotest` | | Test-only: synthetic signals, metrics, ffmpeg `afftdn` A/B harness, noise-bed mixing |
| `rules` | | gocritic ruleguard matchers (build tag `ruleguard`, lint-only) |
| `examples/streaming` | | Runnable decode, denoise, equalize, gain, encode chain |

Non-goals: playback or capture, resampling (go-audio-resampler), codecs, multi-channel processing inside a block (run one block per channel), per-model ML tensor preparation.

### API stability

This library is pre-v1.0.0 and its public API is not stable. Exported types, functions, struct fields, config options and error values may change in any v0.x release when a cleaner design calls for it, and such a change ships as a minor version bump with a note in the release. The README's "stable" label means the package's behaviour and output are settled, not that its signatures are frozen.

Prefer the better API over compatibility: do not add deprecated aliases, adapters, compatibility shims or wrapper functions to keep an old signature working, and do not bend a design to avoid a breaking change. A changed signature still needs every caller in this repo updated (other packages, `examples/`, example tests), plus the package README and godoc. Mark a breaking commit with `!` in its type prefix (`feat(denoiser)!: ...`).

The design rules below are not part of this: they are the library's contract and hold across every release.

## Design rules you must not break

A change that violates one of these is wrong even if tests pass.

1. **No cgo, no runtime subprocess.** Everything builds with `CGO_ENABLED=0`. ffmpeg, sox and Python tooling appear only in tests. `go-ruleguard/dsl` is lint-only, behind a build tag.
2. **The `Processor` contract** (`processor.go`) is exact. Input and output are mono float32 in arbitrary-size chunks. Output aligns sample for sample with input and lags it by `Latency()`. `MaxOutputLen(n)` is an upper bound a caller can size `out` with and never see `ErrBufferTooSmall`. When `out` is too small, `ProcessInto` and `FlushInto` return `ErrBufferTooSmall`, consume nothing and leave the block unchanged. `FlushInto` makes total output length equal total input length, then resets. `Reset` keeps configuration. A block is single-stream and not safe for concurrent use.
3. **Zero allocations in steady state.** Streaming blocks and per-frame paths reuse their scratch; scratch sized to input grows to the largest chunk seen and never shrinks. Scratch is bounded regardless of total input length. The `alloc_test.go` files pin this.
4. **Arbitrary sample rates.** Coefficients, windows and frame geometry are computed for the actual rate, never hard-coded for 48 kHz. Ultrasonic and high-rate audio (192 kHz and up) is a real use case.
5. **SIMD and scalar agree.** Every `simd` call has a pure-Go fallback chosen by the `simd` package at runtime; a kernel change must keep both paths producing the same result within the documented tolerance. CI runs the suite a second time with `SIMD_DISABLE=all`.
6. **Shared errors are shared.** A block reports an undersized buffer with `dsp.ErrBufferTooSmall` and a bad constructor config by wrapping `dsp.ErrInvalidConfig` with the offending field, so `errors.Is` works uniformly across a chain. Package-specific sentinels (`loudnorm`, `denoiser`, `pcm`, `stft`) carry the package-name prefix (`denoiser: ...`) and wrap with `%w`.
7. **Invalid numbers do not escape.** A caller config that would produce NaN or Inf output (a non-finite dB value, frequency or offset) is rejected at construction with `ErrInvalidConfig`. Floors and guards (for example the noise-power floor in `internal/dspshared`) keep divisions and logs finite on silent or all-zero input.
8. **One implementation per concern.** STFT framing, windows and the padding convention live in `stft`; dB-to-linear is `dsp.FactorFromDB`; helpers shared by the two denoise methods live in `internal/dspshared`. Do not grow a parallel copy in a block.
9. **Portable memory access.** `pcm` reinterprets bytes zero-copy only where unaligned 16-bit loads are safe and the host is little-endian (`align_ok.go`); MIPS takes the byte-wise path (`align_trap.go`). Keep that split when touching conversion code.

## Layout

```
doc.go, processor.go     Package dsp: Processor and Flusher contract
errors.go                ErrBufferTooSmall, ErrInvalidConfig
db.go                    FactorFromDB, shared by gain and loudnorm
dsp_test.go              Buffer-contract test on a pass-through block
chain_test.go            End-to-end chain benchmark (pcm, denoiser, equalizer, gain)
chain_alloc_test.go      Steady-state chain allocation test
<pkg>/doc.go             Package documentation; the godoc is the API reference
<pkg>/contract_test.go   Processor conformance (denoiser, equalizer, gain)
<pkg>/alloc_test.go      Allocation guards (build tag !race)
<pkg>/ffmpeg_test.go     ffmpeg oracle tests (loudnorm, denoiser)
denoiser/corpus_test.go  A/B against afftdn on a local real-clip corpus
mel/testdata/            librosa mel filterbank golden values
stft/testdata/           scipy chebwin golden values
internal/audiotest/      Test-only signal, metric and A/B helpers
rules/                   ruleguard matchers
examples/streaming/      Runnable chain example
```

## Testing approach

Unit tests run without external tools; the oracle tests skip when ffmpeg is not on PATH.

- **Regression tests.** New behaviour gets a test. A bug fix gets a regression test that fails before the fix. A test must be able to fail: break the line it pins and confirm it fails, and avoid fixtures where every path gives the same result (silence, DC, a single sine for a frequency-dependent change).
- **Contract tests.** Every streaming block is tested against the `Processor` contract: `MaxOutputLen` holds for every chunk size, `ErrBufferTooSmall` leaves state untouched, output length equals input length after flush, chunking does not change the result, and `Reset` starts a clean stream. Some of this lives in `contract_test.go`, the rest in chunk-invariance and reset tests next to the block; a new block covers all of it.
- **Allocation tests.** `testing.AllocsPerRun` guards steady-state paths; they carry `//go:build !race` because the race detector allocates. A change on a hot path keeps them at zero.
- **Reference oracles.** `loudnorm` is checked against ffmpeg `ebur128`, `denoiser` against `afftdn`, `mel` against librosa golden data, `stft` Dolph-Chebyshev windows against scipy `chebwin` golden data, `octave`'s Butterworth profile against the IEC 61260:1995 class 1 and 2 relative attenuation template (IS 6964:2001 Table 1, Annex B), `equalizer`'s Butterworth band-pass against the analytic Butterworth magnitude. Tolerances are stated in the test; tighten them when the implementation improves, do not loosen them to make a change pass without explaining why in the PR.
- **Corpus A/B.** `task ab` runs `denoiser` and `denoiser/gate` against `afftdn` on real clips in `testdata/corpus/` and recorded noise beds in `testdata/noisebed/`. Both directories are local only and gitignored; the tests skip without them.
- **Loudness corpus.** `task ab:loudnorm` runs `loudnorm.PlanBytes` against ffmpeg `loudnorm` on real clips in `loudnorm/testdata/corpus/` (or `LOUDNORM_CORPUS_DIR`), gitignored and local only; the test skips without the corpus or ffmpeg.
- **Scalar path.** `task test:noasm` (`SIMD_DISABLE=all`, `-count=1` because the variable is read at init) runs the pure-Go kernels.
- **Benchmarks.** `bench_test.go` per package and `BenchmarkChain` in the root. Back a performance claim with benchstat output from `-count` runs, and check arm64 on real hardware rather than under emulation.

## Commands

Go version is read from `go.mod` (currently 1.27); golangci-lint is pinned in `.github/workflows/ci.yml` (currently v2.14.0) and must match the local version. The project tracks the latest stable Go, so no shims or build tags for older releases.

```
task check        # full local gate: build, vet (amd64, arm64), lint, race tests, scalar-path tests
task build        # go build ./...
task test:short   # go test ./...
task test:race    # go test -race ./...
task test:noasm   # SIMD_DISABLE=all go test -count=1 ./...
task vet          # go vet for amd64 and arm64
task lint         # golangci-lint run ./...
task bench        # all benchmarks with allocation counts
task ab           # corpus A/B against ffmpeg afftdn (needs local corpus and ffmpeg)
task tidy         # go mod tidy and verify
```

Run `task check` before pushing. CI runs tests with `-race` on Linux amd64 and arm64, macOS and Windows, the `SIMD_DISABLE=all` leg on both Linux architectures, coverage to Codecov, golangci-lint, vet, cross builds (linux amd64, arm64, mipsle, mips64le; darwin amd64, arm64; windows amd64, arm64) and a weekly govulncheck.

## Conventions

- **Lint.** Config is `.golangci.yaml`, the shared tphakala standard adapted from go-flac. Notable: gocritic with the style, diagnostic and performance tags plus the ruleguard matchers in `rules/`, gocognit 50, errorlint, exhaustive (`default` counts as exhaustive), errname, prealloc, testifylint, thelper. A linter or check is disabled only with a written reason in the config.
- **No `//nolint` without a linter name and a reason.** Fix new findings after a tooling bump in the same change.
- **Modernize.** The `modernize` analyzer is not a golangci linter here; the ruleguard matchers in `rules/` carry the modernization patterns. After editing a matcher, run `golangci-lint cache clean`, because the analysis cache is not keyed on the rule files.
- **Errors.** Error strings carry the package prefix (`dsp:`, `loudnorm:`, `denoiser:`). Wrap with `%w`; use `errors.New` for constant messages.
- **Comments.** Explain why: the reference (paper, standard clause, ffmpeg behaviour) behind a formula, the failure a guard avoids, the measurement behind a constant or a tuned default. Match the existing density.
- **Package docs.** Each package's `doc.go` and README describe its contract, latency and defaults. Update them with any behaviour change, and the root README's package table when a package is added or changes status.
- **Commits** use an imperative subject with a type prefix (`feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `ci:`, `build:`, `chore:`), scoped by package (`perf(denoiser/gate): ...`), with `!` for a breaking change.

## Review guidance

For CodeRabbit, Copilot and human reviewers. Rate findings on what a caller of the library would hear or see.

Flag these, they are real defects here:

- Any break of the design rules above: cgo, a runtime subprocess, a hard-coded 48 kHz assumption, a parallel copy of STFT or shared denoiser logic.
- A `Processor` that consumes input or changes state before returning `ErrBufferTooSmall`, a `MaxOutputLen` that can be exceeded, output whose length after flush differs from input, or output that depends on how the input was chunked.
- An allocation on a steady-state streaming or per-frame path.
- A SIMD path and scalar path that disagree, or a `simd` call whose slice lengths can mismatch.
- NaN, Inf or a division by zero reachable from silent input, an all-zero noise profile or a caller config.
- Integer overflow in sample-count or frame arithmetic on 32-bit targets (`int` is 32 bits on 386 and arm).
- A constructor error that does not wrap `dsp.ErrInvalidConfig`, or a new per-package sentinel for a condition the shared ones already cover.
- An oracle tolerance loosened without justification, or a test that cannot fail on the behaviour it names.

Intentional, do not flag:

- `loudnorm` not implementing `Processor`: two-pass normalization needs the whole clip.
- Scratch that grows to the largest chunk seen and never shrinks; that is the documented allocation model.
- Alloc tests behind `//go:build !race`.
- Oracle and corpus tests skipping without ffmpeg or without the local corpus.
- The `pcm` zero-copy reinterpret on little-endian hosts with safe unaligned access, and the byte-wise MIPS fallback.
- gocritic `hugeParam` disabled; the shared lint standard keeps it off.
- Breaking exported-API changes before v1.0.0 (see API stability), and the absence of deprecated aliases, adapters or shims for the old form.

## Things that do not belong in the repo

- Planning documents: `docs/` is gitignored and stays private, as do specs, plans and research notes under any name.
- Real audio: `testdata/corpus/` and `testdata/noisebed/` are local only. Committed test data is limited to small generated or golden fixtures such as `mel/testdata/`.
- Editor and agent state (`.serena/`, `.claude/`, `.codegraph/`, `.sentrux/`) and build or profile outputs (`*.test`, `*.out`, `*.prof`, `*.pprof`).
