package loudnorm

import (
	"errors"
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

// TestPlanClampedGainBytesComposition pins that the helper is exactly
// MeasureBytes then PlanGain then ClampGainDB, so it cannot drift from the parts
// its documentation names. res is PlanGain's untouched Result, so res.Input
// carries the measurement.
func TestPlanClampedGainBytesComposition(t *testing.T) {
	opts := DefaultOptions()
	b := bytesLE(sineInt16(-20, 1000, 1.0, 48000))

	wantMeas, err := MeasureBytes(b, opts.SampleRate, opts.Channels)
	if err != nil {
		t.Fatal(err)
	}
	wantRes := PlanGain(wantMeas, opts)
	wantGain, wantLimited := ClampGainDB(wantRes.GainDB, DefaultMaxGainDB)

	clampedGainDB, res, limited, err := PlanClampedGainBytes(b, opts, DefaultMaxGainDB)
	if err != nil {
		t.Fatal(err)
	}
	if clampedGainDB != wantGain || res != wantRes || limited != wantLimited {
		t.Fatalf("PlanClampedGainBytes = (%g, %+v, %v); want (%g, %+v, %v)",
			clampedGainDB, res, limited, wantGain, wantRes, wantLimited)
	}
	if res.Input != wantMeas {
		t.Fatalf("res.Input = %+v, want the measurement %+v", res.Input, wantMeas)
	}
}

// TestPlanClampedGainBytesClampFires uses a tiny ceiling so a real gain is
// clamped, pinning that the returned gain is limited to the ceiling with the
// sign of the planned gain, while the untouched Result keeps the unclamped gain.
func TestPlanClampedGainBytesClampFires(t *testing.T) {
	opts := DefaultOptions()
	// A loud clip needs a large attenuation to reach the target, so the planned
	// gain is well outside the tiny ceiling below.
	b := bytesLE(sineInt16(-6, 1000, 1.0, 48000))
	const tinyMax = 1.0

	clampedGainDB, res, limited, err := PlanClampedGainBytes(b, opts, tinyMax)
	if err != nil {
		t.Fatal(err)
	}
	if !limited {
		t.Fatalf("expected the clamp to fire (planned gain %g exceeds %g)", res.GainDB, tinyMax)
	}
	if math.Abs(clampedGainDB) != tinyMax {
		t.Errorf("clamped gain magnitude = %g, want %g", math.Abs(clampedGainDB), tinyMax)
	}
	if (clampedGainDB > 0) != (res.GainDB > 0) {
		t.Errorf("clamped gain sign %g does not match planned gain sign %g", clampedGainDB, res.GainDB)
	}
	if res.GainDB == clampedGainDB {
		t.Errorf("Result.GainDB %g should keep the unclamped value, not the clamped %g", res.GainDB, clampedGainDB)
	}
}

// TestPlanClampedGainBytesSilence pins that silence yields a zero gain and does
// not fire the clamp, so a quiet clip is never lifted into its noise floor.
func TestPlanClampedGainBytesSilence(t *testing.T) {
	opts := DefaultOptions()
	b := make([]byte, 48000*2) // one second of interleaved int16 zeros, mono
	clampedGainDB, res, limited, err := PlanClampedGainBytes(b, opts, DefaultMaxGainDB)
	if err != nil {
		t.Fatal(err)
	}
	if clampedGainDB != 0 || res.GainDB != 0 || limited {
		t.Fatalf("silence: clampedGainDB=%g res.GainDB=%g limited=%v, want 0, 0, false", clampedGainDB, res.GainDB, limited)
	}
	if !math.IsInf(res.Input.IntegratedLUFS, -1) {
		t.Errorf("silence IntegratedLUFS = %g, want -Inf", res.Input.IntegratedLUFS)
	}
}

// TestPlanClampedGainBytesErrors pins the error paths: an odd length and an
// invalid sample rate both surface through MeasureBytes with every output zeroed.
func TestPlanClampedGainBytesErrors(t *testing.T) {
	opts := DefaultOptions()
	if _, _, _, err := PlanClampedGainBytes(make([]byte, 5), opts, DefaultMaxGainDB); !errors.Is(err, ErrOddByteLength) {
		t.Errorf("odd length: err = %v, want ErrOddByteLength", err)
	}

	bad := DefaultOptions()
	bad.SampleRate = 4
	b := bytesLE([]int16{0, 0, 0, 0})
	clampedGainDB, res, limited, err := PlanClampedGainBytes(b, bad, DefaultMaxGainDB)
	if !errors.Is(err, ErrSampleRateTooLow) {
		t.Errorf("low rate: err = %v, want ErrSampleRateTooLow", err)
	}
	if clampedGainDB != 0 || res != (Result{}) || limited {
		t.Errorf("on error every output must be zero, got %g, %+v, %v", clampedGainDB, res, limited)
	}
}
