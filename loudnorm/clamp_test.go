package loudnorm

import (
	"math"
	"testing"
)

// TestClampGainDB pins the symmetric clamp and the limited flag, including the
// magnitude treatment of a negative ceiling, the exact boundary, and a NaN gain
// or a NaN ceiling (each must clamp to 0 rather than slip through and corrupt the
// signal).
func TestClampGainDB(t *testing.T) {
	cases := []struct {
		name      string
		gainDB    float64
		maxAbsDB  float64
		want      float64
		wantLimit bool
	}{
		{"within positive", 5, 30, 5, false},
		{"within negative", -5, 30, -5, false},
		{"above clamps to positive ceiling", 42, 30, 30, true},
		{"below clamps to negative ceiling", -42, 30, -30, true},
		{"at positive boundary is not limited", 30, 30, 30, false},
		{"at negative boundary is not limited", -30, 30, -30, false},
		{"zero gain", 0, 30, 0, false},
		{"negative ceiling treated as magnitude", 42, -30, 30, true},
		{"positive infinity clamps to ceiling", math.Inf(1), 30, 30, true},
		{"negative infinity clamps to ceiling", math.Inf(-1), 30, -30, true},
		{"nan gain clamps to zero", math.NaN(), 30, 0, true},
		{"nan ceiling clamps to zero", 42, math.NaN(), 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			got, limited := ClampGainDB(c.gainDB, c.maxAbsDB)
			if got != c.want || limited != c.wantLimit {
				t.Errorf("ClampGainDB(%g, %g) = %g, %v; want %g, %v", c.gainDB, c.maxAbsDB, got, limited, c.want, c.wantLimit)
			}
		})
	}
}
