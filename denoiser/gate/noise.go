package gate

import (
	"math"
	"slices"
)

// LearnNoise measures a fixed per-bin noise floor from a noise-only excerpt and
// makes it the active floor, disabling the blind tracker. It is the primary path:
// hand it a clip's quiet lead-in or a user-selected noise region. samples must be
// at least FrameSize long (ErrNoiseTooShort); half a second or more gives a stable
// floor. It satisfies the denoiser.NoiseLearner capability, so a consumer holding
// a dsp.Processor can learn noise without depending on the concrete type. It
// allocates setup scratch (via the whole-clip Plan) and must not run concurrently
// with streaming.
func (g *Gate) LearnNoise(samples []float32) error {
	if len(samples) < g.n {
		return ErrNoiseTooShort
	}
	g.plan.MeanPowerInto(g.noiseBuf, samples)
	copyFloor(g.noiseBuf, g.noiseBuf) // apply the epsPower guard the transform leaves to the caller
	g.noise = g.noiseBuf
	g.learned = true
	return nil
}

// SetNoiseFloor sets the fixed per-bin noise power floor directly, disabling the
// blind tracker; it takes effect from the next frame and survives Flush and Reset.
// power must have FrameSize/2+1 values in the same scale NoiseFloor returns
// (|RFFT(hann*x)|^2), else ErrFloorMismatch. Because that scale matches the parent
// denoiser's NoiseProfile.Spectrum(), a caller can share one measurement across
// both methods. Passing nil returns to blind tracking.
func (g *Gate) SetNoiseFloor(power []float32) error {
	if power == nil {
		g.initNoiseSource() // back to blind tracking: reset the tracker, repoint noise
		return nil
	}
	if len(power) != g.bins {
		return ErrFloorMismatch
	}
	copyFloor(g.noiseBuf, power)
	g.noise = g.noiseBuf
	g.learned = true
	return nil
}

// NoiseFloor returns a copy of the learned per-bin noise power floor, or nil when
// the floor is estimated blind.
func (g *Gate) NoiseFloor() []float32 {
	if !g.learned {
		return nil
	}
	return slices.Clone(g.noiseBuf)
}

// copyFloor copies src into dst, replacing each NaN, +Inf, or value at or below
// epsPower with epsPower (the guard NewNoiseProfile applies in the parent
// package), so the floor is always finite and strictly positive and a finite
// power divided by it stays finite. dst may alias src for an in-place floor.
func copyFloor(dst, src []float32) {
	for k, v := range src {
		if !(v > epsPower) || !(v < math.MaxFloat32) { // NaN, <= floor, or +Inf
			v = epsPower
		}
		dst[k] = v
	}
}
