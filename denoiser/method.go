package denoiser

import "fmt"

// This file holds the cross-method vocabulary and the layout rule for the
// pluggable set of denoise methods; the package doc carries the overview. The
// layout rule this file establishes:
//
//   - A method-specific concept stays in its own package and never enters a
//     shared interface. The spectral method's Estimator, NoiseProfile and
//     FrameSize / HopSize are examples: they are knobs of this method, not of
//     "a denoiser".
//   - Cross-cutting optional behaviour is a capability interface (see
//     NoiseLearner) discovered by type assertion, so the base contract stays
//     minimal.
//   - Cross-method vocabulary every family understands (see Strength) is shared
//     here, in the parent package the method sub-packages already import.

// Strength is the coarse aggressiveness dial shared across denoise methods.
// Each method maps it to its own knobs: the spectral method maps it to a tuned
// Params set (see ParamsFor); another family might map it to a gate threshold
// or an ML mixing factor. The zero value is Medium.
type Strength int

const (
	// Medium is the default: a balanced amount of reduction.
	Medium Strength = iota
	// Light is the least aggressive and preserves the most detail.
	Light
	// Heavy is the most aggressive, best on steady broadband noise.
	Heavy
)

// String returns the strength name.
func (s Strength) String() string {
	switch s {
	case Medium:
		return "medium"
	case Light:
		return "light"
	case Heavy:
		return "heavy"
	}
	return fmt.Sprintf("Strength(%d)", int(s))
}

func (s Strength) valid() bool { return s >= Medium && s <= Heavy }

// NoiseLearner is a capability interface implemented by denoise methods that
// can learn their noise estimate from a noise-only excerpt. A consumer
// discovers it with a type assertion on the dsp.Processor it holds:
//
//	if nl, ok := proc.(denoiser.NoiseLearner); ok {
//		if err := nl.LearnNoise(noiseOnly); err != nil {
//			// handle
//		}
//	}
//
// The spectral method implements it; a method with no noise-power concept (a
// fixed gate, some ML methods) simply does not, and the assertion fails
// cleanly. It is deliberately not part of the base dsp.Processor contract.
type NoiseLearner interface {
	// LearnNoise measures a noise estimate from a noise-only excerpt and makes
	// it the method's active noise model for subsequent processing.
	LearnNoise(samples []float32) error
}
