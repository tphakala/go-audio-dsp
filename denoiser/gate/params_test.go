package gate

import (
	"errors"
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestParamsValidate pins each Params guard: NaN and negative MaxAttenuationDB are
// rejected while +Inf is accepted (full gating), a non-finite ThresholdDB and a
// non-positive or infinite TransitionDB/FloorWindowSec are rejected, and negative
// smoothing widths are rejected. Deleting any case lets an invalid Params through.
func TestParamsValidate(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	cases := []struct {
		name string
		mut  func(*Params)
		ok   bool
	}{
		{"valid medium", func(*Params) {}, true},
		{"NaN MaxAttenuationDB", func(p *Params) { p.MaxAttenuationDB = nan }, false},
		{"negative MaxAttenuationDB", func(p *Params) { p.MaxAttenuationDB = -1 }, false},
		{"+Inf MaxAttenuationDB accepted", func(p *Params) { p.MaxAttenuationDB = inf }, true},
		{"NaN ThresholdDB", func(p *Params) { p.ThresholdDB = nan }, false},
		{"+Inf ThresholdDB", func(p *Params) { p.ThresholdDB = inf }, false},
		{"zero TransitionDB", func(p *Params) { p.TransitionDB = 0 }, false},
		{"negative TransitionDB", func(p *Params) { p.TransitionDB = -1 }, false},
		{"+Inf TransitionDB", func(p *Params) { p.TransitionDB = inf }, false},
		{"negative FreqSmoothBins", func(p *Params) { p.FreqSmoothBins = -1 }, false},
		{"negative TimeSmoothFrames", func(p *Params) { p.TimeSmoothFrames = -1 }, false},
		{"zero FloorWindowSec", func(p *Params) { p.FloorWindowSec = 0 }, false},
		{"negative FloorWindowSec", func(p *Params) { p.FloorWindowSec = -1 }, false},
		{"+Inf FloorWindowSec", func(p *Params) { p.FloorWindowSec = inf }, false},
	}
	for _, tc := range cases {
		p := ParamsFor(denoiser.Medium)
		tc.mut(&p)
		err := p.validate()
		switch {
		case tc.ok && err != nil:
			t.Errorf("%s: validate returned %v, want nil", tc.name, err)
		case !tc.ok && err == nil:
			t.Errorf("%s: validate returned nil, want an error", tc.name)
		case !tc.ok && !errors.Is(err, ErrInvalidConfig):
			t.Errorf("%s: error %v is not ErrInvalidConfig", tc.name, err)
		}
	}
}

// TestNewRejectsUnknownStrength and accepts +Inf MaxAttenuationDB through the full
// New path, covering resolve's Strength check and the +Inf floor case end to end.
func TestNewStrengthAndInfFloor(t *testing.T) {
	if _, err := New(Config{SampleRate: 48000, Strength: denoiser.Strength(99)}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("unknown Strength: New err %v, want ErrInvalidConfig", err)
	}
	p := ParamsFor(denoiser.Medium)
	p.MaxAttenuationDB = float32(math.Inf(1))
	if _, err := New(Config{SampleRate: 48000, Params: &p}); err != nil {
		t.Errorf("+Inf MaxAttenuationDB: New err %v, want nil (full gating)", err)
	}
}

// TestParamsForUnknownIsMedium pins the fallback ParamsFor gives for an unknown
// strength.
func TestParamsForUnknownIsMedium(t *testing.T) {
	if got, want := ParamsFor(denoiser.Strength(42)), ParamsFor(denoiser.Medium); got != want {
		t.Errorf("ParamsFor(unknown) = %+v, want Medium %+v", got, want)
	}
}
