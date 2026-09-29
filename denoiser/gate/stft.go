package gate

import (
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
	"github.com/tphakala/simd/c64"
)

// The stream is modelled as the padded sequence p = (n-hop zeros) ++ input ++
// (flush zeros). Frame f is p[f*hop : f*hop+n]; after it is processed the hop
// block p[f*hop : f*hop+hop] has received every overlapping frame. Input sample t
// is p[t+n-hop], so output is aligned with input. Two things delay emission: the
// first ovl-1 frames are all leading zeros (discarded), and time smoothing looks
// L frames ahead, so the output for frame g is produced when frame g+L completes.
// The steady-state lag is therefore (n-hop) + L*hop. The framing, windowing, RFFT
// and |X|^2 live in the stft.Analyzer; this file drives it and the
// stft.Synthesizer does the L-delayed overlap-add.

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
	return dspshared.PendingOutput(g.an.InFill(), extra, g.n, g.hop, g.frames, int64(g.warm))
}

// feed pushes in through the analyzer and, for each frame that completes, updates
// the blind floor (unless learned, flushing, or in the leading warm-up frames),
// computes that frame's soft-gate
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
		g.syn.Add(row)
		if gOut >= int64(g.ovl-1) {
			written += g.syn.Finish(out[written:])
		} else {
			g.syn.Discard()
		}
		g.frames++
	})
	return written
}
