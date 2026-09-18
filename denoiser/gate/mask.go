package gate

import "github.com/tphakala/simd/f32"

// computeMask writes frame f's per-bin gain into g.mask from its power spectrum
// and the active noise floor. The pipeline is fully elementwise: divide by the
// floor, take log10 (finite-floored), map to the sigmoid knee argument
// 4*(snrDB - ThresholdDB)/TransitionDB with one affine, apply the sigmoid to get a
// soft mask in (0,1), then a second affine maps it into the residual-gain range
// [gFloor, 1]. With MaxAttenuationDB == 0 the residual-gain affine has slope
// 1-gFloor == 0, so every bin is forced to unity and the block is an exact
// identity.
func (g *Gate) computeMask(power []float32) {
	f32.Div(g.ratio, power, g.noise)                 // power / noise
	f32.Log10Floored(g.ratio, g.ratio, 1e-20)        // log10, finite -200 dB floor
	f32.Affine(g.ratio, g.ratio, g.slope, g.offset)  // -> 4*(snrDB - ThresholdDB)/TransitionDB
	f32.Sigmoid(g.mask, g.ratio)                     // raw soft mask in (0, 1)
	f32.Affine(g.mask, g.mask, 1-g.gFloor, g.gFloor) // -> gain in [gFloor, 1]
}

// smoothInto writes the smoothed per-bin gain for output frame f-L into dst. It
// averages the 2L+1 mask rows for frames f-2L..f (centered on the output frame),
// recomputed from the ring each call so the result depends only on the stream
// position, never on how input was chunked, then applies frequency smoothing.
// When L is 0 the average is the single current mask row.
func (g *Gate) smoothInto(dst []float32, f int64) {
	l := int64(g.lookahead)
	copy(g.acc, g.maskRow(f-2*l))
	for j := f - 2*l + 1; j <= f; j++ {
		f32.Add(g.acc, g.acc, g.maskRow(j))
	}
	if g.lookahead > 0 {
		f32.Scale(g.acc, g.acc, 1/float32(2*g.lookahead+1))
	}
	g.freqSmooth(dst, g.acc)
}

// freqSmooth writes the edge-aware centered moving average of src (width 2R+1)
// across bins into dst. It accumulates the box sum in g.tmp by adding R shifted
// copies of src to a copy of the center, then multiplies by invCount[k], the
// reciprocal of the number of bins the box actually spans at k, so edge bins
// average only over the part of the window that exists (matching the flagship's
// smoothGain). dst, src and g.tmp must be three distinct buffers; the only
// aliasing is dst==a inside each Add, which simd permits. With R == 0 the loop is
// empty and invCount is all ones, so dst == src.
func (g *Gate) freqSmooth(dst, src []float32) {
	copy(g.tmp, src)
	for r := 1; r <= g.freqR; r++ {
		f32.Add(g.tmp[r:], g.tmp[r:], src[:g.bins-r])        // add the bin r positions to the left
		f32.Add(g.tmp[:g.bins-r], g.tmp[:g.bins-r], src[r:]) // add the bin r positions to the right
	}
	f32.Mul(dst, g.tmp, g.invCount)
}
