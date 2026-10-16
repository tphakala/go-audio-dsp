package stft

import (
	"fmt"
	"math"
	"slices"
)

// Domain of the Dolph-Chebyshev generators. The float64 evaluation below loses
// precision at both extremes (samples at or below zero for very long windows or
// very high attenuation), so the length cap and the attenuation range keep inputs
// inside the region where it stays strictly positive and finite; the corners are
// pinned by TestDolphChebyshevDomainCorners.
const (
	dolphMaxLength        = 1 << 15
	dolphMinAttenuationDB = 20
	dolphMaxAttenuationDB = 200
)

// DolphChebyshev returns the periodic ("DFT-even") Dolph-Chebyshev window of
// length n as float32 samples, computed in float64 and cast down, peak
// normalized to 1. It is the symmetric window of n+1 points with its last sample
// dropped, the convention Hann uses, and matches scipy's
// scipy.signal.windows.chebwin(n, attenuationDB, sym=False) (scipy 1.18.1,
// scipy/signal/windows/_windows.py) to within float32 rounding. Pass the result
// as Config.CustomWindow; spectrogram normalizes by the window energy, which
// makes the peak scale irrelevant.
//
// attenuationDB is the sidelobe attenuation of the underlying symmetric window.
// Dropping the last sample breaks its equiripple shape, so the periodic window's
// highest sidelobe sits a few dB above -attenuationDB (about 3.5 to 4 dB from
// n = 512 up, far more for short high-attenuation windows); use
// DolphChebyshevSymmetric when the bound itself matters. Below about 45 dB the
// window's noise bandwidth stops shrinking with attenuation, and long windows
// peak at their endpoints rather than the center.
//
// n must be in [1, 32768] and attenuationDB in [20, 200]; anything else, NaN
// included, returns an error wrapping ErrInvalidConfig. Above about 150 dB the
// float32 cast, not the design, limits how deep the realized sidelobes go.
// Construction costs O(n^2) arithmetic, so build a window once per (n,
// attenuationDB) and reuse it.
func DolphChebyshev(n int, attenuationDB float64) ([]float32, error) {
	if err := validateDolph(n, attenuationDB); err != nil {
		return nil, err
	}
	w, err := dolph64(n+1, attenuationDB)
	if err != nil {
		return nil, err
	}
	return toFloat32(w[:n]), nil
}

// DolphChebyshevSymmetric returns the symmetric Dolph-Chebyshev window of length
// n, w[i] == w[n-1-i], peak normalized to 1: the equiripple window whose highest
// sidelobe is exactly -attenuationDB, matching
// scipy.signal.windows.chebwin(n, attenuationDB) (scipy 1.18.1). Preferred for
// pure analysis where the sidelobe bound matters; DolphChebyshev is the periodic
// form. The domain, errors and cost are those of DolphChebyshev.
func DolphChebyshevSymmetric(n int, attenuationDB float64) ([]float32, error) {
	if err := validateDolph(n, attenuationDB); err != nil {
		return nil, err
	}
	w, err := dolph64(n, attenuationDB)
	if err != nil {
		return nil, err
	}
	return toFloat32(w), nil
}

func validateDolph(n int, attenuationDB float64) error {
	if n < 1 || n > dolphMaxLength {
		return fmt.Errorf("%w: Dolph-Chebyshev length must be in [1, %d], got %d", ErrInvalidConfig, dolphMaxLength, n)
	}
	// Written as a negated conjunction so NaN, which fails both comparisons, is
	// rejected along with both infinities.
	if !(attenuationDB >= dolphMinAttenuationDB && attenuationDB <= dolphMaxAttenuationDB) {
		return fmt.Errorf("%w: Dolph-Chebyshev attenuationDB must be in [%d, %d], got %g",
			ErrInvalidConfig, dolphMinAttenuationDB, dolphMaxAttenuationDB, attenuationDB)
	}
	return nil
}

func toFloat32(w []float64) []float32 {
	out := make([]float32, len(w))
	for i, v := range w {
		out[i] = float32(v)
	}
	return out
}

// dolph64 returns the symmetric m-point Dolph-Chebyshev window in float64, peak
// normalized to 1, by frequency sampling: the window is the inverse DFT of the
// Chebyshev polynomial T_(m-1) evaluated at x0*cos(pi*k/m), where x0 is set so
// the polynomial's sidelobe plateau sits attenuationDB below its peak (scipy
// 1.18.1 scipy/signal/windows/_windows.py, chebwin).
//
// The spectrum samples are real and mirror-symmetric, so the inverse DFT of a
// symmetric window reduces to w[i] = p[0] + 2*sum_{k=1..K} p[k]*cos(pi*k*d/m)
// with d = (m-1) - 2*i and K = (m-1)/2, evaluated for the first half of the
// window and mirrored. Callers have validated m and attenuationDB.
func dolph64(m int, attenuationDB float64) ([]float64, error) {
	w := make([]float64, m)
	if m == 1 {
		w[0] = 1
		return w, nil
	}
	order := float64(m - 1)
	x0 := math.Cosh(math.Acosh(math.Pow(10, attenuationDB/20)) / order)
	half := (m - 1) / 2

	// For k <= half < m/2 the cosine is positive, so x > 0 and only the
	// hyperbolic and the trigonometric branch of T_order are reachable.
	p := make([]float64, half+1)
	for k := range p {
		x := x0 * math.Cos(math.Pi*float64(k)/float64(m))
		if x > 1 {
			p[k] = math.Cosh(order * math.Acosh(x))
		} else {
			p[k] = math.Cos(order * math.Acos(x))
		}
	}

	// cos(pi*j/m) for j in [0, 2m): indexing the table by (k*d) mod 2m avoids
	// evaluating the cosine of large arguments. Indices stay below 2m, which is
	// below 2^17 at the length cap, so they are safe for a 32-bit int.
	period := 2 * m
	tab := make([]float64, period)
	for j := range tab {
		tab[j] = math.Cos(math.Pi * float64(j) / float64(m))
	}
	for i := range (m + 1) / 2 {
		d := (m - 1) - 2*i // in [0, m-1], below period
		var s float64
		idx := 0
		for k := 1; k <= half; k++ {
			idx += d
			if idx >= period {
				idx -= period
			}
			s += p[k] * tab[idx]
		}
		v := p[0] + 2*s
		w[i] = v
		w[m-1-i] = v
	}

	// The domain keeps the evaluation positive and finite, so this guard does not
	// fire for validated input; it keeps a NaN or Inf from reaching the caller if
	// that measurement is ever wrong on some platform.
	peak := slices.Max(w)
	if !(peak > 0) || math.IsInf(peak, 1) {
		return nil, fmt.Errorf("%w: Dolph-Chebyshev evaluation lost precision (length %d, attenuationDB %g)", ErrInvalidConfig, m, attenuationDB)
	}
	for i := range w {
		w[i] /= peak
	}
	return w, nil
}
