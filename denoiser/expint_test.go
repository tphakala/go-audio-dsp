package denoiser

import (
	"math"
	"testing"
)

const eulerGamma = 0.57721566490153286060

// expint1Ref is an independent float64 reference: the alternating series for
// x <= 5 and a backward-evaluated continued fraction above.
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

func TestExpint1TabulatedValues(t *testing.T) {
	// Abramowitz & Stegun table 5.1.
	cases := map[float64]float64{
		0.1: 1.8229239584, 0.5: 0.5597735948, 1: 0.2193839344,
		2: 0.0489005107, 5: 0.0011482955, 10: 4.156968929e-6,
	}
	for x, want := range cases {
		got := expint1(x)
		if rel := math.Abs(got-want) / want; rel > 2e-6 {
			t.Errorf("E1(%g) = %.10g, want %.10g (rel %.2g)", x, got, want, rel)
		}
	}
}

func TestExpint1AgainstReference(t *testing.T) {
	for e := -6.0; e <= 1.7; e += 0.05 { // x from 1e-6 to 50
		x := math.Pow(10, e)
		got, want := expint1(x), expint1Ref(x)
		if rel := math.Abs(got-want) / want; rel > 2e-6 {
			t.Errorf("E1(%g) = %.10g, ref %.10g (rel %.2g)", x, got, want, rel)
		}
	}
	if !math.IsInf(expint1(0), 1) || !math.IsInf(expint1(-1), 1) {
		t.Error("E1 of a non-positive argument must be +Inf")
	}
	if v := expint1(100); v < 0 || v > 1e-40 {
		t.Errorf("E1(100) = %g, want a non-negative value near 0", v)
	}
}
