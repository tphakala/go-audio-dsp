package dsp_test

import (
	"math"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

func TestFactorFromDB(t *testing.T) {
	cases := []struct {
		dB   float64
		want float64
		tol  float64
	}{
		{0, 1, 0},                    // exact
		{20, 10, 1e-9},               // +20 dB = 10x
		{-20, 0.1, 1e-12},            // -20 dB = 0.1x
		{6.020599913279624, 2, 1e-9}, // +6.02 dB = 2x
		{-6.020599913279624, 0.5, 1e-9},
	}
	for _, c := range cases {
		got := dsp.FactorFromDB(c.dB)
		if math.Abs(got-c.want) > c.tol {
			t.Errorf("FactorFromDB(%g) = %g, want %g (tol %g)", c.dB, got, c.want, c.tol)
		}
	}
	if dsp.FactorFromDB(0) != 1 {
		t.Errorf("FactorFromDB(0) must be exactly 1, got %g", dsp.FactorFromDB(0))
	}
}
