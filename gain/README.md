# gain

Fixed decibel gain for mono float32 audio, as a streaming `dsp.Processor`, with
saturating entry points for the int16 PCM transport form.

A `Gain` multiplies every sample by a linear factor derived from a decibel value
(`10^(dB/20)`). It is memoryless: output aligns sample-for-sample with input,
`Latency()` is 0, and there is no tail, so it does not implement `dsp.Flusher`
and `Reset` is a no-op. One instance can process any number of streams in
sequence; a single instance is not safe for concurrent use.

## Usage

```go
g, err := gain.New(-3.0) // -3 dB
if err != nil {
	// gain is non-finite or overflows float32
}

// Float32 chain path (in place or into a separate buffer):
out := make([]float32, len(in))
n, err := g.ProcessInto(in, out) // n == len(in); out may be exactly in

// int16 transport path, saturating (a boost never wraps):
g.ApplyInt16(samples)     // []int16, in place
err = g.ApplyBytes(buf)   // little-endian int16 in []byte, in place
```

## Notes

- `ProcessInto` does not clamp the float32 output: a boost can push samples past
  [-1, 1], which the true-peak stage or the int16 edge bounds. `ApplyInt16` and
  `ApplyBytes` saturate to the int16 range.
- int16 rounding is round-to-nearest, ties to even, matching the `pcm` package.
  Because the factor and product are held in float32, an unsaturated result can
  differ from a float64 round-half-away implementation by at most one LSB on a
  small fraction of samples.
- The float32 scale and the int16 conversions run through
  [`github.com/tphakala/simd`](https://github.com/tphakala/simd) with a scalar
  fallback. No CGo.
