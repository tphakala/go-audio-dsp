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
// lag is n-hop samples.

// pendingOutput reports how many output samples feeding extra more input
// samples will emit, given the current stream state.
func (d *Denoiser) pendingOutput(extra int) int {
	total := d.inFill + extra
	if total < d.n {
		return 0
	}
	completing := int64((total-d.n)/d.hop) + 1 // frames that will complete
	warm := int64(d.ovl - 1)                    // frames whose block is leading zeros
	emitFrom := max(d.frames, warm)
	cnt := d.frames + completing - emitFrom
	if cnt <= 0 {
		return 0
	}
	return int(cnt) * d.hop
}

// feed appends in to the analysis buffer, runs every frame that becomes
// complete, and writes the finalized output blocks into out, clipped to
// len(out) (ProcessInto sizes out exactly; FlushInto clips the final block).
// flushing freezes the adaptive noise estimate. Returns samples written.
func (d *Denoiser) feed(in, out []float32, flushing bool) int {
	written := 0
	for len(in) > 0 {
		take := min(d.n-d.inFill, len(in))
		copy(d.inBuf[d.inFill:], in[:take])
		d.inFill += take
		in = in[take:]
		if d.inFill < d.n {
			break
		}
		d.processFrame(flushing)
		if d.frames >= int64(d.ovl-1) {
			m := min(d.hop, len(out)-written)
			d.finishBlock(out[written : written+m])
			written += m
		} else {
			d.finishBlock(nil)
		}
		d.frames++
		copy(d.inBuf, d.inBuf[d.hop:])
		d.inFill -= d.hop
	}
	return written
}

// processFrame analyses the frame in inBuf, applies the per-bin gain, and
// overlap-adds the synthesized frame into ola.
func (d *Denoiser) processFrame(flushing bool) {
	d.plan.RFFT(d.spec, d.inBuf, d.window)
	c64.AbsSq(d.power, d.spec)
	d.updateNoise(flushing)
	d.gains.compute(d.gain, d.power, d.noise)
	for k, g := range d.gain {
		d.spec[k] = complex(real(d.spec[k])*g, imag(d.spec[k])*g)
	}
	d.plan.IRFFT(d.synth, d.spec)
	f32.Mul(d.synth, d.synth, d.window)
	f32.Add(d.ola, d.ola, d.synth)
}

// updateNoise advances the adaptive noise estimate for the current frame when a
// tracker is the active source. No tracker is wired in yet, so the noise slot
// is currently constant.
func (d *Denoiser) updateNoise(flushing bool) {
	if d.tracker == nil || d.profile != nil || flushing || d.frames < int64(d.ovl-1) {
		return
	}
	d.tracker.update(d.power)
}

// finishBlock emits the oldest hop samples of the overlap-add accumulator,
// divided by the WOLA normalization, into dst (at most hop samples; empty when
// the block is discarded), then advances the accumulator by one hop.
func (d *Denoiser) finishBlock(dst []float32) {
	for i := range dst {
		dst[i] = d.ola[i] * d.invNorm[i]
	}
	copy(d.ola, d.ola[d.hop:])
	clear(d.ola[d.n-d.hop:])
}
