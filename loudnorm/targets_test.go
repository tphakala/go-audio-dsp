package loudnorm

import (
	"errors"
	"math"
	"testing"
)

func TestValidateTargets(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	cases := []struct {
		name         string
		target, ceil float64
		want         error
	}{
		{"clip defaults", -23, -2, nil},
		{"loud target", -0.1, -1, nil},
		{"ceiling zero", -23, 0, nil},
		{"target zero", 0, -1, ErrTargetOutOfRange},
		{"target positive", 5, -1, ErrTargetOutOfRange},
		{"target at gate", -70, -1, ErrTargetOutOfRange},
		{"target below gate", -75, -1, ErrTargetOutOfRange},
		{"target NaN", nan, -1, ErrTargetOutOfRange},
		{"target +Inf", inf, -1, ErrTargetOutOfRange},
		{"target -Inf", -inf, -1, ErrTargetOutOfRange},
		{"ceiling positive", -23, 1, ErrCeilingInvalid},
		{"ceiling NaN", -23, nan, ErrCeilingInvalid},
		{"ceiling +Inf", -23, inf, ErrCeilingInvalid},
		{"ceiling -Inf", -23, -inf, ErrCeilingInvalid},
	}
	for _, c := range cases {
		err := ValidateTargets(c.target, c.ceil)
		if !errors.Is(err, c.want) || (c.want == nil && err != nil) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.want)
		}
	}
}

func TestExportedConstants(t *testing.T) {
	if AbsoluteGateLUFS != -70 {
		t.Errorf("AbsoluteGateLUFS = %v, want -70", AbsoluteGateLUFS)
	}
	if MinSampleRate != 8000 {
		t.Errorf("MinSampleRate = %v, want 8000", MinSampleRate)
	}
}
