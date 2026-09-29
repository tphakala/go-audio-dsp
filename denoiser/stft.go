package denoiser

import (
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
	"github.com/tphakala/simd/c64"
)

// The stream is modelled as the padded sequence p = (n-hop zeros) ++ input ++
// (flush zeros). Frame f is p[f*hop : f*hop+n]; after it is processed the hop
// block p[f*hop : f*hop+hop] has received every overlapping frame and is
// final. Input sample t is p[t+n-hop], so output is aligned with input, the
// first ovl-1 blocks (all leading zeros) are discarded, and the steady-state
// lag is n-hop samples. The leading n-hop zeros are the analyzer preroll Reset
// feeds; framing, windowing, the RFFT and |X|^2 live in the stft.Analyzer and
// the overlap-add in the stft.Synthesizer.

// pendingOutput reports how many output samples feeding extra more input
// samples will emit, given the current stream state. The first ovl-1 frames'
// blocks are leading zeros and discarded.
func (d *Denoiser) pendingOutput(extra int) int {
	return dspshared.PendingOutput(d.an.InFill(), extra, d.n, d.hop, d.frames, int64(d.ovl-1))
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
		c64.MulReal(spec, spec, d.gain) // per-bin real gain; alias-safe (dst == a), scalar in simd (no SIMD kernel yet, simd #259)
		d.syn.Add(spec)
		if d.frames >= int64(d.ovl-1) {
			written += d.syn.Finish(out[written:])
		} else {
			d.syn.Discard()
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
