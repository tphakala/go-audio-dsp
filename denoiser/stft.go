package denoiser

import "github.com/tphakala/simd/c64"

// The stream framing, preroll and overlap-add live in dspshared.Stream; the
// denoiser emits each frame as it completes (delay 0).

// frame is the denoiser's per-frame step: advance the noise estimate, compute
// the per-bin gain and apply it to spec in place. f is the frame index in the
// stream; flushing freezes the adaptive noise estimate.
func (d *Denoiser) frame(f int64, spec []complex64, power []float32, flushing bool) []complex64 {
	d.updateNoise(f, power, flushing)
	d.gains.compute(d.gain, power, d.noise)
	c64.MulReal(spec, spec, d.gain) // per-bin real gain; alias-safe (dst == a), scalar in simd (no SIMD kernel yet, simd #259)
	return spec
}

// updateNoise advances the adaptive noise estimate for the current frame's
// power when the tracker is the active source: there is no fixed profile, the
// stream is not flushing (the tail freezes the estimate), and the frame is past
// the leading all-zero warm-up blocks that would otherwise feed WOLA padding
// into the estimate. With a fixed profile the noise slot is constant.
func (d *Denoiser) updateNoise(f int64, power []float32, flushing bool) {
	if d.tracker == nil || d.profile != nil || flushing || f < int64(d.ovl-1) {
		return
	}
	d.tracker.update(power)
}
