# equalizer

A cascade of RBJ biquad filters that runs as a streaming `dsp.Processor` over
mono float32 audio, plus a frequency-response helper for drawing the curve.

Implements the eight filter types of the Robert Bristow-Johnson audio EQ
cookbook: `LowPass`, `HighPass`, `AllPass`, `BandPass` (constant 0 dB peak),
`BandReject` (notch), `LowShelf`, `HighShelf`, `Peaking`.

## Usage

```go
e, err := equalizer.New(equalizer.Config{
	SampleRate: 48000,
	Bands: []equalizer.Band{
		{Type: equalizer.HighPass, Frequency: 80, Q: 0.7071},
		{Type: equalizer.Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6},
		{Type: equalizer.HighShelf, Frequency: 8000, Q: 0.7071, GainDB: -3},
	},
})
if err != nil {
	// a band is out of range, or its coefficients would be unstable
}

out := make([]float32, len(in))
n, err := e.ProcessInto(in, out) // n == len(in); out may be exactly in
```

## Parameters

| Field | Used by |
| ----- | ------- |
| `Frequency` (Hz) | every type |
| `Q` | LowPass, HighPass, AllPass, LowShelf, HighShelf |
| `WidthHz` (Hz) | BandPass, BandReject, Peaking |
| `GainDB` | Peaking, LowShelf, HighShelf |
| `Passes` | any (cascades that many identical sections; 0 means 1) |

A field a type does not use is ignored. `Frequency` must be in `(0, SampleRate/2)`;
a `WidthHz` so wide that the lower band edge reaches 0 Hz is rejected rather than
clamped. Parameters that would produce non-finite or unstable coefficients (for
example a filter placed too close to Nyquist) are rejected at construction.

`FilterType.UsesQ`, `UsesWidth` and `UsesGain` report which fields a type reads, so a caller mapping its own settings onto `Band` need not keep a copy of this table.

## Frequency response

`Response` returns the magnitude (dB) and phase (rad) at each requested
frequency, and `LogSweep` builds a log-spaced frequency axis for a plot:

```go
freqs, _ := equalizer.LogSweep(20, 20000, 256)
for _, p := range e.Response(freqs) {
	// p.Hz, p.GainDB, p.PhaseRad
}
```

`Response` allocates its result and is a visualization helper, not an audio-path
method; it does not touch the filter state, so it is safe to call while
streaming. A deep notch reports a large finite attenuation, not `-Inf`, unless
the magnitude evaluates to exactly zero. A `Filter` has the same method.

## Filters beyond one RBJ band

`NewFilter` builds one band as a streaming `Filter` that composes into a
`FilterChain`. Two more constructors build a `Filter` from sections that are not
an RBJ band:

- `NewButterworthBandPass(lower, upper, order, sampleRate)` designs a Butterworth
  band-pass with edges `lower` and `upper` (half-power points of the prototype)
  at the actual sample rate (the band edges are prewarped for the bilinear
  transform), with 0 dB at the geometric centre; the gain at an edge is -3.01 dB
  to within 0.002 dB for order 4 and a one-third-octave band up to 0.95 of
  Nyquist, and further off for low orders and wide bands near Nyquist. `order`
  is the low-pass prototype order, so the band-pass has `2*order` poles in
  `order` biquad sections. The upper edge must be below Nyquist.
- `NewBiquadFilter(sections, sampleRate)` runs caller-designed second-order
  sections. A `Biquad` holds `B0, B1, B2, A1, A2` normalized so `a0` is 1, in the
  recurrence `y[n] = B0*x[n] + B1*x[n-1] + B2*x[n-2] - A1*y[n-1] - A2*y[n-2]`.
  Non-finite, out-of-range or unstable sections (poles on or outside the unit
  circle) are rejected at construction.

```go
f, err := equalizer.NewButterworthBandPass(891.25, 1122.02, 4, 48000)
if err != nil {
	// an edge is out of range, or the coefficients would be unstable
}
f.ApplyFloat64(buf) // float64 samples, in place
```

## Notes

- Coefficients and the per-sample recurrence are float64; input and output are
  float32. The double precision keeps low-frequency filters stable, where the
  poles sit close to the unit circle.
- The biquad recurrence is serial, so it has no SIMD form and runs in scalar
  float64. Its cost is small (a handful of nanoseconds per sample per section).
  There is no CGo.
- `Latency()` is 0 and the block has no tail, so it does not implement
  `dsp.Flusher`. It holds filter state between calls; `Reset` clears it.
