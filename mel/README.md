# mel

Per-frame mel spectrogram columns over the shared `stft` core: STFT framing and
window, power, magnitude or signed real-part spectrum, a mel filterbank
projection, and an optional floored log or power-law compression. It is the
third analysis block on `stft`, after `spectrogram`, and follows the same shape:
a whole-clip `Extractor` and a streaming `ColumnSource` over one projection
stage, so a streamed column equals the whole-clip column at the same frame
offset bit for bit under `NoPad`.

The package is pure audio DSP. It has no model presets and no per-model
normalization, pitch features, frame stacking, or tensor reshaping; those belong
to the consumer.

## Usage

```go
ex, err := mel.New(mel.Config{
	SampleRate: 16000,
	FrameSize:  512,
	HopSize:    128,
	NumMels:    40,
	MinHz:      0,   // DC
	MaxHz:      0,   // 0 means Nyquist
	Input:      mel.InputPower, // |X|^2 (librosa power=2.0), the zero value
	Log:        mel.Log10,      // log10(x + LogOffset)
	LogOffset:  1e-6,
})
if err != nil {
	// a field is out of range, or a custom Filterbank's bin count is wrong
}

m := ex.Compute(signal, stft.NoPad) // m.Data is frame-contiguous [Frames][Mels]
col := m.Column(0)                   // the first column's NumMels values
```

Reuse one `Extractor` across clips and call `ComputeInto` with a retained
`Matrix` to stay allocation-free. Stream a live source with a `ColumnSource`:

```go
cs, _ := mel.NewColumnSource(cfg)
cs.Feed(chunk, func(col []float32, centerSample int64) {
	// col has NumMels values, valid until this call returns; copy to keep it
})
```

## Configuration

| Group | Fields |
| ----- | ------ |
| Transform | `SampleRate`, `FrameSize`, `HopSize`, `Window`, `WindowLength`, `WindowAlign`, `CustomWindow` |
| Filterbank | `NumMels`, `MinHz`, `MaxHz`, `Scale` (`Slaney`/`HTK`), `Norm` (`NormSlaney`/`NormNone`) |
| Precomputed bank | `Filterbank` (overrides the filterbank group) |
| Value | `Input` (`InputPower`/`InputMagnitude`/`InputReal`), `Log` (`LogNone`/`Log10`/`LogNatural`/`PowerLaw`), `LogOffset`, `LogFloor`, `PowerExponent` |

The zero values of `Scale`, `Norm` and `Input` reproduce librosa's
`melspectrogram` defaults (Slaney scale, `norm="slaney"`, `power=2.0`). `Log` is
`LogNone` by default, keeping the log stage opt-in; when set, at least one of
`LogOffset` and `LogFloor` must be `> 0` for `Log10` and `LogNatural`, so silence
maps to a finite value (`y = log(max(x + LogOffset, LogFloor))`) rather than
`-Inf`. `PowerLaw` needs neither (see below).

`WindowLength` shorter than `FrameSize` supports a `win_length < n_fft` front end
(the window is zero-padded into the frame per `WindowAlign`). `MaxHz` above
Nyquist is rejected, not silently clamped, because a model front end shifted to a
different band would not match training.

### Real-part projection and power-law compression

`InputReal` projects the signed real part `Re(X)` of the half spectrum instead of
`|X|^2` or `|X|`, so mel values are signed under `LogNone`. Because the log of a
signed value is undefined, `InputReal` is rejected with `Log10` and `LogNatural`.

`PowerLaw` compresses each mel value as
`y = max(|x| + LogOffset, LogFloor)^PowerExponent`, with the absolute value taken
only for `InputReal` (power and magnitude are already non-negative).
`PowerExponent` must be in `(0, 1]` for `PowerLaw` and `0` for every other `Log`.
`0^q = 0`, so no offset or floor is required. A front end that projects the real
part, squares it, and raises the result to a power `p` is `InputReal` with
`PowerLaw` and `PowerExponent = 2p`, `LogOffset = LogFloor = 0`; the consumer
supplies the bank and the exponent.

`Pow` runs a fast kernel while every value in a column is positive and finite; a
column holding an exact zero (an empty mel row, or a projection that cancels to
zero) takes the slower exact scalar path, which agrees with the fast kernel within the
relative error simd documents for `f32.Pow` (about 1.4e-5).

### Filterbanks

`NewFilterbank` generates a Slaney or HTK bank; `HzToMel`/`MelToHz` expose the
scale mappings. A model whose bank comes from a table supplies it directly:

```go
fb, _ := mel.FilterbankFromRows(rows) // dense NumMels x NumBins, NumBins = FrameSize/2 + 1
cfg.Filterbank = fb                   // ignores NumMels/MinHz/MaxHz/Scale/Norm
```

The bank is stored sparsely (each row's first nonzero bin plus its contiguous
weights) and is immutable, so one `*Filterbank` is safe to share between
instances and goroutines. Low bands narrower than a bin come back empty and
project to 0, matching librosa's warn-and-continue behavior.

## Framing

`NoPad` frames `signal[f*HopSize : f*HopSize+FrameSize]` (librosa
`center=False`). `PadZero` and `PadReflect` center each frame on `f*HopSize` with
`FrameSize/2` of zero or reflected padding per side (librosa `center=True`,
`pad_mode="constant"` or `"reflect"`), realized by feeding the padding through the
`NoPad` analyzer so the result matches `stft.Plan` under the same mode without an
allocation. The streaming path is `NoPad`; a consumer that wants a centered start
feeds its own leading zeros. `NumFrames` matches `stft.Plan.NumFrames` for every
mode.

## Frame-parallel extraction

Under `NoPad`, frame `f` depends only on `signal[f*HopSize : f*HopSize+FrameSize]`
and `ComputeInto` resets its analyzer first, so
`ComputeInto` on `signal[f0*HopSize : (f0+c-1)*HopSize+FrameSize]` writes exactly
columns `f0` to `f0+c-1` of the whole-clip matrix, bit for bit. To use several
cores, give each goroutine its own `Extractor` (they are not safe for concurrent
use) built from the same `Config`, which shares one immutable `*Filterbank`, and
have it write a three-index sub-slice of one `Matrix.Data`
(`Data[f0*Mels : (f0+c)*Mels : (f0+c)*Mels]`). `ExampleExtractor_parallel` shows
the pattern and checks it against the serial result.

## Notes

- Filterbank tables are computed in float64 and cast to float32 once; the hot
  path is float32 in, float32 out. The projection is sparse per row: a scalar
  multiply-accumulate loop for rows narrower than 16 bins and
  `simd.f32.DotProduct` for wider ones; the input (`Sqrt`, or `Deinterleave2` for
  the real part), offset, floor and log or power (`Abs`, `Pow`) stages are `simd`
  kernels, so the package builds and runs under `SIMD_DISABLE=all` (pure-Go
  fallback) with no CGo and no assembly of its own.
- Within one build, a streamed column equals the whole-clip `NoPad` column bit
  for bit. Across build targets and SIMD tiers the values are tolerance-stable,
  not bit-stable, like `stft` and `spectrogram`.
- Construction allocates; `ComputeInto` (into a caller-sized `Matrix`) and
  `Feed` are allocation-free in the steady state. No type is safe for concurrent
  use; build one per stream. A `*Filterbank` is immutable and shareable.

Every arithmetic pass routes through
[`github.com/tphakala/simd`](https://github.com/tphakala/simd).
