package denoiser

import (
	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// The stream is modelled as the padded sequence p = (n-hop zeros) ++ input ++
// (flush zeros). Frame f is p[f*hop : f*hop+n]; after it is processed the hop
// block p[f*hop : f*hop+hop] has received every overlapping frame and is
// final. Input sample t is p[t+n-hop], so output is aligned with input, the
// first ovl-1 blocks (all leading zeros) are discarded, and the steady-state
// lag is n-hop samples. The leading n-hop zeros are the analyzer preroll Reset
// feeds; framing, windowing, the RFFT and |X|^2 live in the stft.Analyzer.

// noEmit is the no-op frame callback used to feed the analyzer's leading-zero
// preroll, during which no frame completes and nothing is synthesized.
func noEmit(spec []complex64, power []float32) {}

// pendingOutput reports how many output samples feeding extra more input
// samples will emit, given the current stream state.
func (d *Denoiser) pendingOutput(extra int) int {
	total := d.an.InFill() + extra
	if total < d.n {
		return 0
	}
	completing := int64((total-d.n)/d.hop) + 1 // frames that will complete
	warm := int64(d.ovl - 1)                   // frames whose block is leading zeros
	emitFrom := max(d.frames, warm)
	cnt := d.frames + completing - emitFrom
	if cnt <= 0 {
		return 0
	}
	return int(cnt) * d.hop
}

// feed pushes in through the analyzer and, for each frame that completes,
// applies the per-bin gain and overlap-adds the synthesized frame, writing the
// finalized output blocks into out, clipped to len(out) (ProcessInto sizes out
// exactly; FlushInto clips the final block). flushing freezes the adaptive
// noise estimate. Returns samples written.
func (d *Denoiser) feed(in, out []float32, flushing bool) int {
	written := 0
	d.an.Feed(in, func(spec []complex64, power []float32) {
		d.updateNoise(power, flushing)
		d.gains.compute(d.gain, power, d.noise)
		c64.MulReal(spec, spec, d.gain) // per-bin real gain; alias-safe (dst == a), scalar in simd v1.10.0
		d.an.Inverse(d.synth, spec)
		f32.Mul(d.synth, d.synth, d.window)
		f32.Add(d.ola, d.ola, d.synth)
		if d.frames >= int64(d.ovl-1) {
			m := min(d.hop, len(out)-written)
			d.finishBlock(out[written : written+m])
			written += m
		} else {
			d.finishBlock(nil)
		}
		d.frames++
	})
	return written
}

// updateNoise advances the adaptive noise estimate for the current frame's
// power when the tracker is the active source: there is no fixed profile, the
// stream is not flushing (the tail freezes the estimate), and the frame is past
// the leading all-zero warm-up blocks that would otherwise feed WOLA padding
// into the estimate. With a fixed profile the noise slot is constant.
func (d *Denoiser) updateNoise(power []float32, flushing bool) {
	if d.tracker == nil || d.profile != nil || flushing || d.frames < int64(d.ovl-1) {
		return
	}
	d.tracker.update(power)
}

// finishBlock emits the oldest hop samples of the overlap-add accumulator,
// divided by the WOLA normalization, into dst (at most hop samples; empty when
// the block is discarded), then advances the accumulator by one hop.
func (d *Denoiser) finishBlock(dst []float32) {
	f32.Mul(dst, d.ola[:len(dst)], d.invNorm[:len(dst)])
	copy(d.ola, d.ola[d.hop:])
	clear(d.ola[d.n-d.hop:])
}
