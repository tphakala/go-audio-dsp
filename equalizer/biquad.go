package equalizer

import "fmt"

// Biquad is one second-order section with caller-supplied coefficients,
// normalized so a0 is 1. The section runs the Direct Form I recurrence
//
//	y[n] = B0*x[n] + B1*x[n-1] + B2*x[n-2] - A1*y[n-1] - A2*y[n-2]
//
// so A1 and A2 carry the sign of a denominator 1 + A1*z^-1 + A2*z^-2, the
// convention of a second-order-sections row with a0 = 1 (and of the
// coefficients NewFilter derives from the RBJ cookbook).
type Biquad struct {
	B0, B1, B2, A1, A2 float64
}

// NewBiquadFilter builds a Filter from a cascade of caller-designed biquad
// sections at sampleRate, run in the order given. It copies sections, so the
// caller may reuse the slice. sampleRate is only recorded, so that a FilterChain
// can enforce one rate and Filter.Response can map Hz to radians; the
// coefficients are used as they are.
//
// It returns an error wrapping ErrInvalidConfig for a sampleRate <= 0, for no
// sections or more than 16, and for the first section with a non-finite
// coefficient, one beyond the float32 range, or poles on or outside the unit
// circle (the condition NewFilter applies to its own bands). The error names the
// offending section by index.
func NewBiquadFilter(sections []Biquad, sampleRate int) (*Filter, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, sampleRate)
	}
	if len(sections) < 1 || len(sections) > maxPasses {
		return nil, fmt.Errorf("%w: sections must number in [1, %d], got %d", ErrInvalidConfig, maxPasses, len(sections))
	}
	out := make([]section, len(sections))
	for i, b := range sections {
		c := coeffs{b0: b.B0, b1: b.B1, b2: b.B2, a1: b.A1, a2: b.A2}
		if !c.inRange() {
			return nil, fmt.Errorf("%w: section %d: coefficients must be finite and within float32 range", ErrInvalidConfig, i)
		}
		if !c.stable() {
			return nil, fmt.Errorf("%w: section %d: unstable (need |A2| < 1 and |A1| < 1 + A2), A1 %g, A2 %g", ErrInvalidConfig, i, b.A1, b.A2)
		}
		out[i].c = c
	}
	return &Filter{sampleRate: sampleRate, sections: out}, nil
}
