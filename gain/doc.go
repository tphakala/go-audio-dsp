// Package gain applies a fixed decibel gain to mono float32 audio as a streaming
// dsp.Processor, and offers saturating entry points for the int16 PCM transport
// form.
//
// A Gain multiplies every sample by a linear factor derived from a decibel
// value (10^(dB/20)). It is memoryless: output aligns sample-for-sample with
// input, latency is zero, and there is no tail, so it does not implement
// dsp.Flusher and Reset is a no-op. Because it holds no stream state, one
// instance can process any number of streams; it is still not safe for
// concurrent use on the same instance.
//
// ProcessInto is the float32 chain path and does not clamp its output: a boost
// can push samples outside [-1, 1], which the true-peak stage or the int16 edge
// is expected to bound. ApplyInt16 and ApplyBytes are the transport-form paths;
// they round to nearest (ties to even) and saturate to the int16 range so a
// boost never wraps. Rounding matches the pcm package; because the factor and
// the product are held in float32, an unsaturated result may differ by at most
// one least-significant bit from a float64 round-half-away implementation on a
// small fraction of samples.
//
// The float32 scale and the int16 conversions run through
// github.com/tphakala/simd, taking its AVX/NEON paths where available and
// falling back to portable Go elsewhere. There is no CGo.
package gain
