package gate

import (
	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// The stream is modelled as the padded sequence p = (n-hop zeros) ++ input ++
// (flush zeros). Frame f is p[f*hop : f*hop+n]; after it is processed the hop
// block p[f*hop : f*hop+hop] has received every overlapping frame. Input sample t
// is p[t+n-hop], so output is aligned with input. Two things delay emission: the
// first ovl-1 frames are all leading zeros (discarded), and time smoothing looks
// L frames ahead, so the output for frame g is produced when frame g+L completes.
// The steady-state lag is therefore (n-hop) + L*hop. The framing, windowing, RFFT
// and |X|^2 live in the stft.Analyzer; this file drives it and does the
// L-delayed overlap-add.

// noEmit is the no-op frame callback used to feed the analyzer's leading-zero
// preroll, during which no frame completes and nothing is synthesized.
func noEmit(spec []complex64, power []float32) {}

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

// pendingOutput reports how many output samples feeding extra more input samples
// will emit, given the current stream state. Emission for frame g is delayed to
// when frame g+L completes and the first warm = ovl-1+L frames are discarded, so
// the emitted-frame count after F frames is max(0, F-warm).
func (g *Gate) pendingOutput(extra int) int {
	total := g.an.InFill() + extra
	if total < g.n {
		return 0
	}
	completing := int64((total-g.n)/g.hop) + 1 // frames that will complete
	warm := int64(g.warm)
	emitFrom := max(g.frames, warm)
	cnt := g.frames + completing - emitFrom
	if cnt <= 0 {
		return 0
	}
	return int(cnt) * g.hop
}

// feed pushes in through the analyzer and, for each frame that completes, updates
// the blind floor (unless learned or flushing), computes that frame's soft-gate
// mask, and stores the frame's mask and spectrum in the smoothing rings. Once L
// frames of lookahead are buffered it produces the output for the L-delayed frame:
// time- and frequency-smoothed gain, applied to that frame's spectrum, inverse
// transformed and overlap-added, with the finalized hop block written into out
// (clipped to len(out); ProcessInto sizes out exactly, FlushInto clips the last
// block). flushing freezes the blind floor. Returns samples written.
func (g *Gate) feed(in, out []float32, flushing bool) int {
	written := 0
	g.an.Feed(in, func(spec []complex64, power []float32) {
		f := g.frames
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
			g.frames++ // still priming the lookahead ring; no output frame ready yet
			return
		}
		gOut := f - int64(g.lookahead)
		g.smoothInto(g.gain, f) // time smoothing over the ring, then frequency smoothing
		row := g.specRow(gOut)
		c64.MulReal(row, row, g.gain) // per-bin real gain; alias-safe (dst == a), scalar in simd (simd #259)
		g.an.Inverse(g.synth, row)
		// Fuse the synthesis window and overlap-add: ola += synth * window. synth is
		// only the IRFFT output and is not read after this; MulAdd uses a hardware
		// FMA where available, so it is tolerance-stable across CPU tiers.
		f32.MulAdd(g.ola, g.synth, g.window)
		if gOut >= int64(g.ovl-1) {
			m := min(g.hop, len(out)-written)
			g.finishBlock(out[written : written+m])
			written += m
		} else {
			g.finishBlock(nil)
		}
		g.frames++
	})
	return written
}

// finishBlock emits the oldest hop samples of the overlap-add accumulator, divided
// by the WOLA normalization, into dst (at most hop samples; empty when the block
// is discarded), then advances the accumulator by one hop.
func (g *Gate) finishBlock(dst []float32) {
	f32.Mul(dst, g.ola[:len(dst)], g.invNorm[:len(dst)])
	copy(g.ola, g.ola[g.hop:])
	clear(g.ola[g.n-g.hop:])
}
