package denoiser

import (
	"math"
	"testing"
)

const eulerGamma = 0.57721566490153286060

// expint1Ref is an independent float64 reference for E1(x) = int_x^inf e^-t / t dt:
// the alternating series for x <= 5 and a backward-evaluated continued fraction
// above.
func expint1Ref(x float64) float64 {
	if x <= 5 {
		sum, term := 0.0, 1.0
		for n := 1; n <= 80; n++ {
			term *= -x / float64(n) // (-x)^n / n!
			sum += -term / float64(n)
		}
		return -eulerGamma - math.Log(x) + sum
	}
	cf := 0.0
	for n := 100; n >= 1; n-- {
		cf = float64(n*n) / (x + float64(2*n+1) - cf)
	}
	return math.Exp(-x) / (x + 1 - cf)
}

// lsaGainRef is the unclamped MMSE-LSA gain w*exp(0.5*E1(w*gamma)) in float64,
// with the integrand argument limited to [lsaVMin, 100] as the estimator
// definition requires.
func lsaGainRef(xi, gamma float64) float64 {
	w := xi / (1 + xi)
	v := min(max(w*gamma, lsaVMin), 100)
	return w * math.Exp(0.5*expint1Ref(v))
}

// TestLSAGainMatchesReference compares the vectorized gain pass with the float64
// reference over a grid of a priori and a posteriori SNRs. prevAmp2 is driven
// directly so the decision-directed xi lands on the grid.
func TestLSAGainMatchesReference(t *testing.T) {
	p := ParamsFor(Medium)
	p.Estimator = MMSELSA
	p.FreqSmoothBins = 0
	var xis, gammas []float64
	for xi := 1e-3; xi < 1e5; xi *= 1.7 {
		for gamma := 0.05; gamma < 1e4; gamma *= 1.9 {
			xis, gammas = append(xis, xi), append(gammas, gamma)
		}
	}
	n := len(xis)
	g := newGainState(n, p)
	power, noise, gain := make([]float32, n), make([]float32, n), make([]float32, n)
	for k := range n {
		// Invert snr(): pick noise so nz = beta*noise = 1, power = gamma, and
		// prevAmp2 so that xi = alpha*prevAmp2 + (1-alpha)*max(gamma-1,0) hits xis[k].
		noise[k] = 1 / g.beta
		power[k] = float32(gammas[k])
		g.prevAmp2[k] = float32((xis[k] - float64(1-g.alpha)*math.Max(gammas[k]-1, 0)) / float64(g.alpha))
	}
	g.lsaGain(gain, power, noise)
	checked := 0
	for k := range n {
		xi, gam := g.snr(power[k], noise[k], g.prevAmp2[k])
		want := math.Min(math.Max(lsaGainRef(float64(xi), float64(gam)), float64(g.gFloor)), 1)
		// float32 arithmetic throughout, plus the 4.7e-7 relative error of the
		// vectorized E1: a few 1e-6 at worst.
		if rel := math.Abs(float64(gain[k])-want) / want; rel > 1e-5 {
			t.Fatalf("xi=%g gamma=%g: gain %.9g, reference %.9g (rel %.2g)", xi, gam, gain[k], want, rel)
		}
		checked++
	}
	if checked < 500 {
		t.Fatalf("grid too small: %d points", checked)
	}
}
