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

// TestStrengthNamesAndValidity pins the Strength vocabulary: the integer values
// are an identifier contract (consumers store them, and Strength(0) is the
// Config zero value), so LightMedium is appended rather than inserted.
func TestStrengthNamesAndValidity(t *testing.T) {
	cases := []struct {
		s     Strength
		value int
		name  string
		valid bool
	}{
		{Medium, 0, "medium", true},
		{Light, 1, "light", true},
		{Heavy, 2, "heavy", true},
		{LightMedium, 3, "light-medium", true},
		{Strength(4), 4, "Strength(4)", false},
		{Strength(-1), -1, "Strength(-1)", false},
	}
	for _, c := range cases {
		if int(c.s) != c.value {
			t.Errorf("%v: value %d, want %d", c.name, int(c.s), c.value)
		}
		if got := c.s.String(); got != c.name {
			t.Errorf("Strength(%d).String() = %q, want %q", c.value, got, c.name)
		}
		if got := c.s.Valid(); got != c.valid {
			t.Errorf("Strength(%d).Valid() = %v, want %v", c.value, got, c.valid)
		}
	}
}
