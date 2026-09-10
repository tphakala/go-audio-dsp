package gain

import (
	"fmt"
	"math"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/pcm"

	simdf32 "github.com/tphakala/simd/f32"
)

// Gain is a memoryless streaming dsp.Processor that scales every sample by a
// fixed linear factor derived from a decibel gain. It has zero latency and no
// tail, so it needs no Flusher and Reset is a no-op. A Gain is not safe for
// concurrent use on the same instance, but holding no stream state it can serve
// any number of streams in sequence.
type Gain struct {
	gainDB float64
	factor float32
}

// New returns a Gain applying gainDB decibels. gainDB must be finite and small
// enough that its linear factor is representable in float32; a NaN, an infinity,
// or a gain so large its factor overflows float32 (above roughly +770 dB)
// returns ErrInvalidConfig, since a non-finite factor would emit non-finite
// audio silently. A very negative gain whose factor underflows to zero is
// allowed (it produces silence).
func New(gainDB float64) (*Gain, error) {
	if math.IsNaN(gainDB) || math.IsInf(gainDB, 0) {
		return nil, fmt.Errorf("%w: gainDB must be finite, got %g", ErrInvalidConfig, gainDB)
	}
	factor := float32(dsp.FactorFromDB(gainDB))
	if math.IsInf(float64(factor), 0) {
		return nil, fmt.Errorf("%w: gainDB %g is too large (linear factor overflows float32)", ErrInvalidConfig, gainDB)
	}
	return &Gain{gainDB: gainDB, factor: factor}, nil
}

// GainDB returns the decibel gain the block applies.
func (g *Gain) GainDB() float64 { return g.gainDB }

// ProcessInto writes in scaled by the gain factor into out and returns the
// number of samples written (len(in)). out must have room for len(in) samples
// (see MaxOutputLen); a shorter out returns ErrBufferTooSmall and writes
// nothing. out may be exactly in for in-place scaling; a shifted overlap of out
// and in is not supported. The float32 output is not clamped.
func (g *Gain) ProcessInto(in, out []float32) (int, error) {
	if len(out) < len(in) {
		return 0, ErrBufferTooSmall
	}
	n := len(in)
	if n == 0 {
		return 0, nil
	}
	if g.factor == 1 {
		copy(out[:n], in) // a no-op when out aliases in
		return n, nil
	}
	simdf32.Scale(out[:n], in, g.factor)
	return n, nil
}

// MaxOutputLen returns inputLen: gain writes one output sample per input sample.
func (g *Gain) MaxOutputLen(inputLen int) int { return inputLen }

// Latency returns 0: gain is memoryless, so output aligns with input.
func (g *Gain) Latency() int { return 0 }

// Reset is a no-op: a Gain holds no stream state.
func (g *Gain) Reset() {}

// ApplyInt16 scales interleaved int16 PCM in place by the gain factor, rounding
// to nearest (ties to even) and saturating to the int16 range so a boost never
// wraps. Rounding matches the pcm package; because the factor and the product
// are held in float32, an unsaturated result may differ by at most one LSB from
// a float64 round-half-away implementation on a small fraction of samples.
// Allocation-free.
func (g *Gain) ApplyInt16(s []int16) {
	pcm.ScaleInt16(s, g.factor)
}

// ApplyBytes scales interleaved little-endian int16 PCM carried as bytes in
// place. len(b) must be even (two bytes per sample); otherwise ErrOddByteLength
// is returned and b is left untouched. On little-endian hosts the scale is
// zero-copy. Rounding and saturation are as in ApplyInt16.
func (g *Gain) ApplyBytes(b []byte) error {
	if len(b)%2 != 0 {
		return ErrOddByteLength
	}
	pcm.InPlaceInt16(b, func(s []int16) { pcm.ScaleInt16(s, g.factor) })
	return nil
}
