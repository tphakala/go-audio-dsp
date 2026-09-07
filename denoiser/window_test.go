package denoiser

import (
	"math"
	"testing"
)

func TestHannPeriodic(t *testing.T) {
	const n = 16
	w := hannPeriodic(n)
	if len(w) != n {
		t.Fatalf("len %d, want %d", len(w), n)
	}
	if w[0] != 0 {
		t.Errorf("w[0] = %g, want 0", w[0])
	}
	if w[n/2] != 1 {
		t.Errorf("w[n/2] = %g, want 1", w[n/2])
	}
	if math.Abs(float64(w[n/4])-0.5) > 1e-6 {
		t.Errorf("w[n/4] = %g, want 0.5", w[n/4])
	}
	// Periodic symmetry: w[i] == w[n-i] for 0 < i < n.
	for i := 1; i < n; i++ {
		if math.Abs(float64(w[i]-w[n-i])) > 1e-6 {
			t.Errorf("w[%d]=%g != w[%d]=%g", i, w[i], n-i, w[n-i])
		}
	}
}

func TestWolaNormHannQuarterHopIsConstant(t *testing.T) {
	const n = 1024
	w := hannPeriodic(n)
	norm := wolaNorm(w, w, n/4)
	if len(norm) != n/4 {
		t.Fatalf("len %d, want %d", len(norm), n/4)
	}
	for i, v := range norm {
		if math.Abs(float64(v)-1.5) > 1e-5 {
			t.Fatalf("norm[%d] = %g, want 1.5", i, v)
		}
	}
	// hop n/8: 8 overlapping Hann^2 copies sum to 3.0 as well.
	for i, v := range wolaNorm(w, w, n/8) {
		if math.Abs(float64(v)-3.0) > 1e-5 {
			t.Fatalf("hop n/8 norm[%d] = %g, want 3.0", i, v)
		}
	}
}

func TestWolaNormOtherHopsArePositive(t *testing.T) {
	const n = 256
	w := hannPeriodic(n)
	for _, hop := range []int{n / 2, n / 16, 1} {
		for i, v := range wolaNorm(w, w, hop) {
			if !(v > 0.4) {
				t.Fatalf("hop %d norm[%d] = %g, want > 0.4", hop, i, v)
			}
		}
	}
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	for i, v := range wolaNorm(ones, ones, n/4) {
		if v != 4 {
			t.Fatalf("rect norm[%d] = %g, want 4", i, v)
		}
	}
}
