package equalizer

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
)

// NewButterworthBandPass builds a Butterworth band-pass Filter whose half-power
// edges are lower and upper (Hz) at sampleRate, with 0 dB gain at the
// geometric centre sqrt(lower*upper). The peak lies at the prewarped centre, a
// little above the geometric one, so the gain at an edge is -3.01 dB plus that
// excess: within 0.002 dB for order 4 and a one-third-octave band up to 0.95 of
// Nyquist, further off for low orders and wide bands near Nyquist (order 1 reads
// -2.79 dB at the upper edge of a one-third-octave band ending at 21 kHz at
// 48 kHz). order is the prototype
// (low-pass) order N: the band-pass has 2N poles, realised as N biquad sections,
// and falls off at 6N dB per octave far from the band. Order 1 is the
// second-order band-pass, order 4 an eight-pole one.
//
// The design is the classical one: the Butterworth low-pass prototype poles
// exp(j*pi*(2k+N+1)/(2N)) are mapped to a band-pass by the low-pass to
// band-pass substitution, then to the z plane with the bilinear transform whose
// band edges are prewarped, tan(pi*f/sampleRate), so the prototype's half-power
// points land on lower and upper at any sample rate, including near Nyquist,
// where an unwarped design would be pulled in. Each section is normalized to unit gain at
// the geometric centre, so the cascade reads 0 dB there. The sections run the
// same float64 recurrence as the other Filters. Filter.Response gives the
// realised response.
//
// It returns an error wrapping ErrInvalidConfig for a sampleRate <= 0, an order
// outside [1, 16], lower not > 0, upper not above lower or not below
// the Nyquist frequency, and for edges so narrow relative to sampleRate that the
// float64 coefficients are no longer stable.
func NewButterworthBandPass(lower, upper float64, order, sampleRate int) (*Filter, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, sampleRate)
	}
	if order < 1 || order > MaxPasses {
		return nil, fmt.Errorf("%w: order must be in [1, %d], got %d", ErrInvalidConfig, MaxPasses, order)
	}
	fs := float64(sampleRate)
	// Negated comparisons so NaN (every comparison false) is rejected; an infinite
	// edge fails the ordering or Nyquist test below.
	if !(lower > 0) {
		return nil, fmt.Errorf("%w: lower edge must be > 0, got %g", ErrInvalidConfig, lower)
	}
	if !(upper > lower) {
		return nil, fmt.Errorf("%w: upper edge must be > lower edge (%g), got %g", ErrInvalidConfig, lower, upper)
	}
	if !(upper < fs/2) {
		return nil, fmt.Errorf("%w: upper edge must be below the Nyquist frequency (%g), got %g", ErrInvalidConfig, fs/2, upper)
	}
	f, err := NewBiquadFilter(butterworthBandPass(lower, upper, fs, order), sampleRate)
	if err != nil {
		return nil, fmt.Errorf("butterworth band-pass %g to %g Hz at %d Hz: %w", lower, upper, sampleRate, err)
	}
	return f, nil
}

// butterworthBandPass returns the order sections of the design, in float64,
// ordered by ascending analog pole frequency. The caller has validated the
// arguments: 0 < lower < upper < fs/2 and 1 <= order <= MaxPasses.
func butterworthBandPass(lower, upper, fs float64, order int) []Biquad {
	// Prewarped analog edges for the bilinear map s = (1 - z^-1)/(1 + z^-1).
	o1, o2 := math.Tan(math.Pi*lower/fs), math.Tan(math.Pi*upper/fs)
	w0sq, bw := o1*o2, o2-o1

	// Each analog factor is s^2 + alpha*s + beta, with numerator s.
	type factor struct{ alpha, beta float64 }
	factors := make([]factor, 0, order)
	for k := range order {
		// The real prototype pole (odd order only) is p = -1. Find it by the index
		// at which the angle is pi, not by imag(p) == 0: cmplx.Rect(1, math.Pi) has
		// an imaginary part of about 1.2e-16.
		if 2*k+order+1 == 2*order {
			factors = append(factors, factor{alpha: bw, beta: w0sq})
			continue
		}
		p := cmplx.Rect(1, math.Pi*float64(2*k+order+1)/float64(2*order))
		if imag(p) < 0 {
			continue // the conjugate of a pole handled below
		}
		// The substitution s -> (s^2 + w0sq)/(bw*s) turns the prototype pole p into
		// the roots of s^2 - p*bw*s + w0sq = 0. Both roots are in the open left
		// half plane, one with Im > 0 and one with Im < 0. The conjugate pole's
		// quadratic supplies their conjugates, so each root here stands for a
		// conjugate pair and gives one real factor.
		h := p * complex(bw/2, 0)
		d := cmplx.Sqrt(h*h - complex(w0sq, 0))
		for _, s := range [...]complex128{h + d, h - d} {
			factors = append(factors, factor{alpha: -2 * real(s), beta: real(s)*real(s) + imag(s)*imag(s)})
		}
	}
	slices.SortStableFunc(factors, func(a, b factor) int { return cmp.Compare(a.beta, b.beta) })

	wc := 2 * math.Pi * math.Sqrt(lower*upper) / fs
	secs := make([]Biquad, len(factors))
	for i, fc := range factors {
		// Bilinear transform of s/(s^2 + alpha*s + beta).
		a0 := 1 + fc.alpha + fc.beta
		c := coeffs{b0: 1 / a0, b1: 0, b2: -1 / a0, a1: 2 * (fc.beta - 1) / a0, a2: (1 - fc.alpha + fc.beta) / a0}
		g := 1 / cmplx.Abs(c.at(wc))
		secs[i] = Biquad{B0: c.b0 * g, B1: 0, B2: c.b2 * g, A1: c.a1, A2: c.a2}
	}
	return secs
}
