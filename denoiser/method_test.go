package denoiser

import "testing"

// TestStrengthInvalidFallbacks covers the out-of-range paths of the Strength
// vocabulary: String falls back to the numeric form, and ParamsFor returns
// Medium's knobs. resolve() rejects an invalid Strength before calling
// ParamsFor, but ParamsFor is exported and reachable directly, so its fallback
// is a real path.
func TestStrengthInvalidFallbacks(t *testing.T) {
	if got := Strength(99).String(); got != "Strength(99)" {
		t.Errorf("Strength(99).String() = %q, want %q", got, "Strength(99)")
	}
	if ParamsFor(Strength(99)) != ParamsFor(Medium) {
		t.Error("ParamsFor(invalid) must fall back to Medium's params")
	}
}
