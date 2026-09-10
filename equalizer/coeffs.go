package equalizer

import (
	"fmt"
	"math"
)

// coeffs holds a biquad's transfer-function coefficients, already normalized so
// a0 is 1 (each raw coefficient divided by a0). The recurrence is
//
//	y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//
// Coefficients are held in float64 and the per-sample recurrence runs in
// float64 (see section), which a general-purpose EQ needs: at low corner
// frequencies relative to the sample rate the poles sit close to the unit
// circle, where float32 state loses the precision to stay stable.
type coeffs struct {
	b0, b1, b2, a1, a2 float64
}

// newCoeffs computes the normalized biquad coefficients for a band at the given
// sample rate using the Robert Bristow-Johnson audio EQ cookbook formulas. It
// validates the band and rejects parameters that produce non-finite or unstable
// coefficients, wrapping the reason in ErrInvalidConfig.
func newCoeffs(b Band, sampleRate int) (coeffs, error) {
	if !b.Type.valid() {
		return coeffs{}, fmt.Errorf("%w: unknown FilterType %d", ErrInvalidConfig, int(b.Type))
	}
	if b.Passes < 0 || b.Passes > maxPasses {
		return coeffs{}, fmt.Errorf("%w: Passes must be in [0, %d], got %d", ErrInvalidConfig, maxPasses, b.Passes)
	}
	fs := float64(sampleRate)
	nyquist := fs / 2
	// Negated comparison so a NaN frequency (every comparison false) is rejected.
	if !(b.Frequency > 0 && b.Frequency < nyquist) {
		return coeffs{}, fmt.Errorf("%w: Frequency must be in (0, %g), got %g", ErrInvalidConfig, nyquist, b.Frequency)
	}
	if b.Type.usesGain() && (math.IsNaN(b.GainDB) || math.IsInf(b.GainDB, 0)) {
		return coeffs{}, fmt.Errorf("%w: GainDB must be finite, got %g", ErrInvalidConfig, b.GainDB)
	}

	w0 := 2 * math.Pi * b.Frequency / fs
	cw := math.Cos(w0)
	sw := math.Sin(w0)

	alpha, err := b.alpha(w0, sw)
	if err != nil {
		return coeffs{}, err
	}

	a := 1.0
	if b.Type.usesGain() {
		a = math.Pow(10, b.GainDB/40)
	}

	var b0, b1, b2, a0, a1, a2 float64
	switch b.Type {
	case LowPass:
		b0, b1, b2 = (1-cw)/2, 1-cw, (1-cw)/2
		a0, a1, a2 = 1+alpha, -2*cw, 1-alpha
	case HighPass:
		b0, b1, b2 = (1+cw)/2, -(1+cw), (1+cw)/2
		a0, a1, a2 = 1+alpha, -2*cw, 1-alpha
	case AllPass:
		b0, b1, b2 = 1-alpha, -2*cw, 1+alpha
		a0, a1, a2 = 1+alpha, -2*cw, 1-alpha
	case BandPass: // constant 0 dB peak gain
		b0, b1, b2 = alpha, 0, -alpha
		a0, a1, a2 = 1+alpha, -2*cw, 1-alpha
	case BandReject:
		b0, b1, b2 = 1, -2*cw, 1
		a0, a1, a2 = 1+alpha, -2*cw, 1-alpha
	case Peaking:
		b0, b1, b2 = 1+alpha*a, -2*cw, 1-alpha*a
		a0, a1, a2 = 1+alpha/a, -2*cw, 1-alpha/a
	case LowShelf:
		beta := math.Sqrt(a) / b.Q
		b0 = a * ((a + 1) - (a-1)*cw + beta*sw)
		b1 = 2 * a * ((a - 1) - (a+1)*cw)
		b2 = a * ((a + 1) - (a-1)*cw - beta*sw)
		a0 = (a + 1) + (a-1)*cw + beta*sw
		a1 = -2 * ((a - 1) + (a+1)*cw)
		a2 = (a + 1) + (a-1)*cw - beta*sw
	case HighShelf:
		beta := math.Sqrt(a) / b.Q
		b0 = a * ((a + 1) + (a-1)*cw + beta*sw)
		b1 = -2 * a * ((a - 1) + (a+1)*cw)
		b2 = a * ((a + 1) + (a-1)*cw - beta*sw)
		a0 = (a + 1) - (a-1)*cw + beta*sw
		a1 = 2 * ((a - 1) - (a+1)*cw)
		a2 = (a + 1) - (a-1)*cw - beta*sw
	}
	// No default: b.Type was checked with valid() above, and the switch lists
	// every FilterType, so the exhaustive linter guards against an unhandled one.

	c := coeffs{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0}
	if err := c.check(b); err != nil {
		return coeffs{}, err
	}
	return c, nil
}

// alpha returns the RBJ intermediate alpha, from Q for the Q-parameterized types
// and from WidthHz (converted to octaves) for the width-parameterized types.
func (b Band) alpha(w0, sw float64) (float64, error) {
	if b.Type.usesWidth() {
		if !(b.WidthHz > 0) {
			return 0, fmt.Errorf("%w: WidthHz must be > 0 for a %s filter, got %g", ErrInvalidConfig, b.Type, b.WidthHz)
		}
		lower := b.Frequency - b.WidthHz/2
		if !(lower > 0) {
			return 0, fmt.Errorf("%w: WidthHz %g is too wide at Frequency %g (lower band edge <= 0)", ErrInvalidConfig, b.WidthHz, b.Frequency)
		}
		bwOctaves := math.Log2((b.Frequency + b.WidthHz/2) / lower)
		return sw * math.Sinh(math.Ln2/2*bwOctaves*w0/sw), nil
	}
	if !(b.Q > 0) {
		return 0, fmt.Errorf("%w: Q must be > 0 for a %s filter, got %g", ErrInvalidConfig, b.Type, b.Q)
	}
	return sw / (2 * b.Q), nil
}

// check rejects normalized coefficients that a general EQ cannot use: a
// non-finite value or one too large for the float32 output, and a pole pair
// outside the unit circle (an unstable filter, which extreme parameters near
// Nyquist can produce even from finite inputs). The stability test is the
// standard second-order condition |a2| < 1 and |a1| < 1 + a2.
func (c coeffs) check(b Band) error {
	for _, v := range [...]float64{c.b0, c.b1, c.b2, c.a1, c.a2} {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > math.MaxFloat32 {
			return fmt.Errorf("%w: filter coefficients out of range for %s at Frequency %g, GainDB %g", ErrInvalidConfig, b.Type, b.Frequency, b.GainDB)
		}
	}
	if !(math.Abs(c.a2) < 1 && math.Abs(c.a1) < 1+c.a2) {
		return fmt.Errorf("%w: %s filter is unstable at Frequency %g (too close to 0 or Nyquist for these parameters)", ErrInvalidConfig, b.Type, b.Frequency)
	}
	return nil
}
