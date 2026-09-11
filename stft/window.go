package stft

import (
	"fmt"
	"math"
)

// Window selects a built-in analysis window shape. The zero value is Hann.
type Window int

const (
	// Hann is the periodic ("DFT-even") Hann window, w[i] = 0.5 - 0.5*cos(2*pi*i/n).
	// The periodic form is the one whose shifted squared copies sum to a constant at
	// hop n/4 and n/8, which is what overlap-add reconstruction relies on; it is the
	// right default for a spectral pipeline that resynthesizes.
	Hann Window = iota
	// HannSymmetric is the symmetric Hann window, w[i] = 0.5 - 0.5*cos(2*pi*i/(n-1)).
	// Preferred for pure analysis (filter design, display) where the exact endpoints
	// matter and no overlap-add reconstruction follows.
	HannSymmetric
	// Hamming is the periodic Hamming window, w[i] = 0.54 - 0.46*cos(2*pi*i/n).
	Hamming
	// Rectangular is the all-ones window (no tapering).
	Rectangular
)

// valid reports whether w is a defined built-in window.
func (w Window) valid() bool { return w >= Hann && w <= Rectangular }

// String returns the window name.
func (w Window) String() string {
	switch w {
	case Hann:
		return "hann"
	case HannSymmetric:
		return "hann-symmetric"
	case Hamming:
		return "hamming"
	case Rectangular:
		return "rectangular"
	}
	return fmt.Sprintf("Window(%d)", int(w))
}

// GenerateWindow returns the built-in window w of length n as float32 samples,
// computed in float64 and cast down. n <= 0 returns an empty slice and n == 1 a
// single unit sample (a length below 2 carries no meaningful taper). An
// unrecognized Window value returns the default (Hann) window. The Hann formula is
// load-bearing: changing it moves the denoiser's output bits (pinned by
// TestGenerateWindowHannFormula).
func GenerateWindow(w Window, n int) []float32 {
	if n <= 0 {
		return []float32{}
	}
	out := make([]float32, n)
	if n == 1 {
		out[0] = 1
		return out
	}
	switch w {
	case Rectangular:
		for i := range out {
			out[i] = 1
		}
	case HannSymmetric:
		for i := range out {
			out[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		}
	case Hamming:
		for i := range out {
			out[i] = float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n)))
		}
	default: // Hann (periodic) and any unexpected value fall back to the default.
		for i := range out {
			out[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
		}
	}
	return out
}

// WOLANorm returns the weighted overlap-add normalization of analysis window wa
// and synthesis window ws (normally the same length; if they differ, the shorter
// length is used) at the given hop:
// norm[i] = sum_m wa[i+m*hop]*ws[i+m*hop] for i in [0, hop), accumulated in
// float64 and cast to float32. In steady-state overlap-add every output sample at
// position i within its hop block is the overlap-add sum divided by norm[i]. A
// hop that does not divide n still returns a length-hop slice; only a hop that
// divides n gives a constant-overlap-add reconstruction. hop must be >= 1.
func WOLANorm(wa, ws []float32, hop int) []float32 {
	if hop < 1 {
		return []float32{}
	}
	if len(ws) < len(wa) {
		wa = wa[:len(ws)]
	}
	norm := make([]float32, hop)
	for i := range norm {
		var s float64
		for j := i; j < len(wa); j += hop {
			s += float64(wa[j]) * float64(ws[j])
		}
		norm[i] = float32(s)
	}
	return norm
}
