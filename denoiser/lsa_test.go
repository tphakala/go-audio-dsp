package denoiser

import (
	"math"
	"testing"
)

// TestLSAUnityBound pins the shortcut in estimatorGain: from lsaVUnity upward the
// MMSE-LSA correction exp(0.5*E1(v)) is exactly 1 in float32, so returning w
// without evaluating it changes no output bit. It also checks the bound is not
// vacuous: just below it the correction is still 1, but at v = 1 it is not.
func TestLSAUnityBound(t *testing.T) {
	for v := float64(lsaVUnity); v <= lsaVMax; v += 0.01 {
		if got := float32(math.Exp(0.5 * expint1(v))); got != 1 {
			t.Fatalf("correction at v=%g is %v, want exactly 1", v, got)
		}
	}
	if got := float32(math.Exp(0.5 * expint1(1))); got == 1 {
		t.Errorf("correction at v=1 is %v: the test would not notice a wrong bound", got)
	}
	// The shortcut and the full formula agree on the gain, bit for bit.
	for _, xi := range []float32{20, 100, 1e4} {
		for _, gamma := range []float32{20, 150, 1e5} {
			w := xi / (1 + xi)
			v := min(max(float64(w*gamma), lsaVMin), lsaVMax)
			if v < lsaVUnity {
				continue
			}
			full := w * float32(math.Exp(0.5*expint1(v)))
			if got := estimatorGain(MMSELSA, xi, gamma); got != full {
				t.Errorf("xi=%g gamma=%g: shortcut %v, full formula %v", xi, gamma, got, full)
			}
		}
	}
}
