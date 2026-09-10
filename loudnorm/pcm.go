package loudnorm

import (
	"github.com/tphakala/go-audio-dsp/pcm"

	simdf32 "github.com/tphakala/simd/f32"
)

// applyGainFloat32 multiplies every sample by a linear gain in place. Float
// samples are not clamped: the caller's true-peak ceiling is what bounds them.
func applyGainFloat32(samples []float32, gain float64) {
	if gain == 1.0 || len(samples) == 0 {
		return
	}
	simdf32.Scale(samples, samples, float32(gain))
}

// applyGainInt16 multiplies every sample by a linear gain in place, rounding to
// the nearest integer (ties to even) and saturating to the int16 range so a
// boost can never wrap around. It delegates to pcm.ScaleInt16, the shared
// saturating int16 scale used across the library.
func applyGainInt16(samples []int16, gain float64) {
	pcm.ScaleInt16(samples, float32(gain))
}
