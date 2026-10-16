package gate

import "github.com/tphakala/simd/c64"

// Framing, the leading-zero preroll and the overlap-add live in
// dspshared.Stream. The gate holds each frame back L frames (the time-smoothing
// lookahead), so the output for frame f-L is produced when frame f completes;
// the Stream is initialized with delay L for that.

// maskRow returns the ring row holding frame f's per-bin mask. The ring holds the
// last 2L+1 frames; a pre-stream frame index (negative, only at stream start)
// maps via the positive modulo to a slot Reset primed to unity gain, so the
// start-edge average eases gating in over the first L frames rather than
// over-attenuating them.
func (g *Gate) maskRow(f int64) []float32 {
	m := int64(2*g.lookahead + 1)
	s := int(((f%m)+m)%m) * g.bins
	return g.maskRing[s : s+g.bins]
}

// specRow returns the ring row holding frame f's spectrum. The ring holds the
// last L+1 frames, exactly the frames from the current output frame through the
// newest completed frame.
func (g *Gate) specRow(f int64) []complex64 {
	m := int64(g.lookahead + 1)
	s := int(((f%m)+m)%m) * g.bins
	return g.specRing[s : s+g.bins]
}

// frame is the gate's per-frame step. It updates the blind floor (unless
// learned, flushing, or in the leading warm-up frames), computes that frame's
// soft-gate mask, and stores the frame's mask and spectrum in the smoothing
// rings. Once L frames of lookahead are buffered it returns the L-delayed
// frame's spectrum with the time-smoothed gain applied, for the stream to
// overlap-add; until then it returns nil. f is the frame index in the stream;
// flushing freezes the blind floor.
func (g *Gate) frame(f int64, spec []complex64, power []float32, flushing bool) []complex64 {
	// Blind floor learning: only when no floor is learned, not flushing, and
	// past the leading all-zero warm-up blocks (which would otherwise feed WOLA
	// padding into the estimate). A learned floor freezes step 1 entirely.
	if !g.learned && !flushing && f >= int64(g.ovl-1) {
		g.tracker.push(power)
	}
	g.computeMask(power)       // per-bin gain in [gFloor, 1] for frame f
	copy(g.maskRow(f), g.mask) // store for time smoothing
	copy(g.specRow(f), spec)   // spec is valid only inside this callback
	if f < int64(g.lookahead) {
		return nil // still priming the lookahead ring; no output frame ready yet
	}
	g.smoothInto(g.gain, f) // time smoothing over the ring
	row := g.specRow(f - int64(g.lookahead))
	c64.MulReal(row, row, g.gain) // per-bin real gain; alias-safe (dst == a), scalar in simd (simd #259)
	return row
}
