package stft

import (
	"fmt"
	"math"
)

// Window selects a built-in analysis window shape. The zero value is Hann.
// Parameterized windows (DolphChebyshev, DolphChebyshevSymmetric) are not Window
// values: a caller generates one and passes it as Config.CustomWindow.
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

// WindowAlign places an analysis window shorter than the frame inside the frame.
// It is ignored when the window length equals FrameSize. The zero value is
// AlignCenter.
type WindowAlign int

const (
	// AlignCenter pads (FrameSize-WindowLength)/2 zeros before the window (floor
	// division, matching librosa util.pad_center) and the remaining zeros after,
	// centering a short window in the frame.
	AlignCenter WindowAlign = iota
	// AlignLeft places the window at the frame start and pads the remaining zeros
	// after it (the Kaldi / torchaudio "round up to a power of two" convention).
	AlignLeft
)

// valid reports whether a is a defined alignment.
func (a WindowAlign) valid() bool { return a >= AlignCenter && a <= AlignLeft }

// String returns the alignment name.
func (a WindowAlign) String() string {
	switch a {
	case AlignCenter:
		return "center"
	case AlignLeft:
		return "left"
	}
	return fmt.Sprintf("WindowAlign(%d)", int(a))
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
			out[i] = float32(hannSymmetricAt(i, n))
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

// hannSymmetricAt returns sample i of the symmetric m-point Hann window in
// float64, 0.5 - 0.5*cos(2*pi*i/(m-1)). It is the one expression behind both
// GenerateWindow(HannSymmetric, m) and EdgeAnalyzer's Hann shape, so the two
// cannot drift apart. m must be >= 2; a one-point window has no taper and the
// callers handle it before calling.
func hannSymmetricAt(i, m int) float64 {
	return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(m-1))
}

// hannSymmetric64 fills dst with the symmetric len(dst)-point Hann window in
// float64. A single point is the unit sample, as in GenerateWindow.
func hannSymmetric64(dst []float64) {
	if len(dst) == 1 {
		dst[0] = 1
		return
	}
	for i := range dst {
		dst[i] = hannSymmetricAt(i, len(dst))
	}
}

// ShapeFamily selects a symmetric window family that EdgeAnalyzer evaluates at
// many lengths, one per frame near the signal edges. It is separate from Window,
// which names a window of one fixed length. The zero value is ShapeHann.
type ShapeFamily int

const (
	// ShapeHann is the symmetric Hann window, 0.5 - 0.5*cos(2*pi*i/(m-1)).
	ShapeHann ShapeFamily = iota
	// ShapeDolphChebyshev is the symmetric Dolph-Chebyshev window with its centre
	// sample normalized to 1 (DolphChebyshevSymmetric normalizes the peak
	// instead; the two differ below about 86 dB attenuation).
	ShapeDolphChebyshev
)

// String returns the family name.
func (f ShapeFamily) String() string {
	switch f {
	case ShapeHann:
		return "hann"
	case ShapeDolphChebyshev:
		return "dolph-chebyshev"
	}
	return fmt.Sprintf("ShapeFamily(%d)", int(f))
}

// WindowShape is a window family plus its parameter. AttenuationDB is the
// Dolph-Chebyshev sidelobe attenuation, in [20, 200] dB; it must be 0 for
// ShapeHann, which catches a Dolph attenuation set alongside the zero-value
// family.
type WindowShape struct {
	Family        ShapeFamily
	AttenuationDB float64
}

// validate reports whether s is usable at frame size n, with an error wrapping
// ErrInvalidConfig otherwise.
func (s WindowShape) validate(n int) error {
	switch s.Family {
	case ShapeHann:
		if s.AttenuationDB != 0 {
			return fmt.Errorf("%w: Hann shape takes no attenuation, got AttenuationDB %g", ErrInvalidConfig, s.AttenuationDB)
		}
		return nil
	case ShapeDolphChebyshev:
		return validateDolph(n, s.AttenuationDB)
	}
	return fmt.Errorf("%w: unknown ShapeFamily %d", ErrInvalidConfig, int(s.Family))
}
