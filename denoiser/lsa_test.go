package denoiser

import (
	"math"
	"testing"
)

// TestLSAUnityBound pins the clamp of v to lsaVUnity in the MMSE-LSA gain: from
// lsaVUnity upward the correction exp(0.5*E1(v)) is exactly 1 in float32, so
// clamping changes no output bit. It also checks the bound is not vacuous: at
// v = 1 the correction is not 1.
func TestLSAUnityBound(t *testing.T) {
	for v := float64(lsaVUnity); v <= 100; v += 0.01 {
		if got := float32(math.Exp(0.5 * expint1Ref(v))); got != 1 {
			t.Fatalf("correction at v=%g is %v, want exactly 1", v, got)
		}
	}
	if got := float32(math.Exp(0.5 * expint1Ref(1))); got == 1 {
		t.Errorf("correction at v=1 is %v: the test would not notice a wrong bound", got)
	}
}
