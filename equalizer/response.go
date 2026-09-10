package equalizer

import (
	"fmt"
	"math"
	"math/cmplx"
)

// ResponsePoint is the equalizer's steady-state response at one frequency: the
// magnitude in dB and the phase in radians. GainDB is negative infinity only
// when the magnitude evaluates to exactly zero; a deep notch normally reports a
// large finite attenuation (a few hundred dB down), not -Inf. PhaseRad is the
// principal value in (-pi, pi] and is unspecified where the magnitude is zero.
type ResponsePoint struct {
	Hz       float64
	GainDB   float64
	PhaseRad float64
}

// at returns the complex frequency response H(e^jw) of one section at angular
// frequency w in radians per sample.
func (c coeffs) at(w float64) complex128 {
	z1 := cmplx.Rect(1, -w)   // e^{-jw}
	z2 := cmplx.Rect(1, -2*w) // e^{-2jw}
	num := complex(c.b0, 0) + complex(c.b1, 0)*z1 + complex(c.b2, 0)*z2
	den := complex(1, 0) + complex(c.a1, 0)*z1 + complex(c.a2, 0)*z2
	return num / den
}

// Response evaluates the equalizer's magnitude and phase at each frequency in
// freqs (Hz), returning one ResponsePoint per input frequency in order. It is a
// visualization helper (for example to draw a response curve), not an
// audio-path method: it allocates the returned slice and does not touch the
// filter state, so it is safe to call while streaming. Only frequencies in
// [0, SampleRate/2] are meaningful; the response is periodic, so a frequency
// outside that range is evaluated as given rather than rejected.
func (e *Equalizer) Response(freqs []float64) []ResponsePoint {
	pts := make([]ResponsePoint, len(freqs))
	for i, f := range freqs {
		w := 2 * math.Pi * f / float64(e.sampleRate)
		h := complex(1, 0)
		for j := range e.sections {
			h *= e.sections[j].c.at(w)
		}
		mag := cmplx.Abs(h)
		gainDB := math.Inf(-1)
		if mag > 0 {
			gainDB = 20 * math.Log10(mag)
		}
		pts[i] = ResponsePoint{Hz: f, GainDB: gainDB, PhaseRad: cmplx.Phase(h)}
	}
	return pts
}

// LogSweep returns n frequencies spaced logarithmically from fLow to fHigh
// inclusive, the usual x-axis for a response plot. It returns an error wrapping
// ErrInvalidConfig if fLow <= 0, fHigh <= fLow, or n < 2. The first and last
// values are exactly fLow and fHigh.
func LogSweep(fLow, fHigh float64, n int) ([]float64, error) {
	if !(fLow > 0) {
		return nil, fmt.Errorf("%w: fLow must be > 0, got %g", ErrInvalidConfig, fLow)
	}
	if !(fHigh > fLow) {
		return nil, fmt.Errorf("%w: fHigh must be > fLow (%g), got %g", ErrInvalidConfig, fLow, fHigh)
	}
	if n < 2 {
		return nil, fmt.Errorf("%w: n must be >= 2, got %d", ErrInvalidConfig, n)
	}
	freqs := make([]float64, n)
	logLow := math.Log(fLow)
	step := (math.Log(fHigh) - logLow) / float64(n-1)
	for i := range freqs {
		freqs[i] = math.Exp(logLow + step*float64(i))
	}
	freqs[0] = fLow // pin the endpoints exactly, free of exp/log round-off
	freqs[n-1] = fHigh
	return freqs, nil
}
