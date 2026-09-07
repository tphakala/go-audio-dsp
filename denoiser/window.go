package denoiser

import "math"

// hannPeriodic returns the periodic ("DFT-even") Hann window of length n,
// w[i] = 0.5 - 0.5*cos(2*pi*i/n). The periodic form is the one whose shifted
// copies (and their squares) sum to a constant at hop n/4 and n/8, which is
// what overlap-add reconstruction relies on.
func hannPeriodic(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// wolaNorm returns the weighted overlap-add normalization for analysis window
// wa and synthesis window ws (same length n) at the given hop, which must
// divide n: norm[i] = sum_m wa[i+m*hop]*ws[i+m*hop] for i in [0, hop). In
// steady state every output sample at position i within its hop block is the
// overlap-add sum divided by norm[i]. Accumulated in float64.
func wolaNorm(wa, ws []float32, hop int) []float32 {
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
