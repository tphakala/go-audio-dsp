package denoiser

import (
	"errors"
	"math"
	"testing"
)

func TestAutoFrameSize(t *testing.T) {
	cases := map[int]int{
		8000: 128, 16000: 256, 22050: 512, 24000: 512, 32000: 512,
		44100: 1024, 48000: 1024, 96000: 2048,
	}
	for sr, want := range cases {
		if got := autoFrameSize(sr); got != want {
			t.Errorf("autoFrameSize(%d) = %d, want %d", sr, got, want)
		}
	}
}

func TestConfigDefaults(t *testing.T) {
	rc, p, err := Config{SampleRate: 48000}.resolve()
	if err != nil {
		t.Fatal(err)
	}
	if rc.FrameSize != 1024 || rc.HopSize != 256 {
		t.Fatalf("frame/hop = %d/%d, want 1024/256", rc.FrameSize, rc.HopSize)
	}
	if p != ParamsFor(Medium) {
		t.Fatalf("params = %+v, want Medium", p)
	}
	custom := Params{MaxAttenuationDB: 3, OverSubtraction: 1, SNRSmoothing: 0.9, MinPriorSNRDB: -10, Estimator: Wiener, TrackWindowSec: 1}
	_, p, err = Config{SampleRate: 16000, Strength: Heavy, Params: &custom}.resolve()
	if err != nil {
		t.Fatal(err)
	}
	if p != custom {
		t.Fatalf("explicit Params not used: %+v", p)
	}
}

// TestConfigInvalid checks that resolve rejects malformed Config and Params
// values, including NaN and non-finite knobs, with ErrInvalidConfig.
func TestConfigInvalid(t *testing.T) {
	bad := []Config{
		{},
		{SampleRate: -1},
		{SampleRate: 48000, FrameSize: 1000},
		{SampleRate: 48000, FrameSize: 32},
		{SampleRate: 48000, HopSize: 300},
		{SampleRate: 48000, HopSize: 2048},
		{SampleRate: 48000, HopSize: -1},
		{SampleRate: 48000, FrameSize: 256, HopSize: 256}, // ovl==1: no overlap, periodic Hann cannot reconstruct
		{SampleRate: 48000, Strength: Strength(99)},
		{SampleRate: 48000, Params: &Params{MaxAttenuationDB: -1, OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 1}},
		{SampleRate: 48000, Params: &Params{MaxAttenuationDB: float32(math.NaN()), OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 1}}, // NaN slips past a bare < 0 check
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 1, MinPriorSNRDB: float32(math.NaN())}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 1, MinPriorSNRDB: float32(math.Inf(1))}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 0, SNRSmoothing: 0.9, TrackWindowSec: 1}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 1, TrackWindowSec: 1}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 0.9, FreqSmoothBins: -1, TrackWindowSec: 1}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 0}},
		{SampleRate: 48000, Params: &Params{OverSubtraction: 1, SNRSmoothing: 0.9, TrackWindowSec: 1, Estimator: Estimator(7)}},
	}
	for i, c := range bad {
		if _, _, err := c.resolve(); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("case %d (%+v): err = %v, want ErrInvalidConfig", i, c, err)
		}
	}
}

func TestStrengthsAreOrdered(t *testing.T) {
	l, m, h := ParamsFor(Light), ParamsFor(Medium), ParamsFor(Heavy)
	if !(l.MaxAttenuationDB < m.MaxAttenuationDB && m.MaxAttenuationDB < h.MaxAttenuationDB) {
		t.Errorf("MaxAttenuationDB not increasing: %g %g %g", l.MaxAttenuationDB, m.MaxAttenuationDB, h.MaxAttenuationDB)
	}
	for _, p := range []Strength{Light, Medium, Heavy} {
		if err := ParamsFor(p).validate(); err != nil {
			t.Errorf("%v: %v", p, err)
		}
		if p.String() == "" {
			t.Errorf("%d has empty String()", p)
		}
	}
	if Strength(0) != Medium {
		t.Error("zero Strength must be Medium")
	}
	if Estimator(0) != MMSELSA {
		t.Error("zero Estimator must be MMSELSA")
	}
}

// TestMaxAttenuationInfAccepted documents that a +Inf MaxAttenuationDB is a
// valid "bottomless" gain floor (full gating). Unlike a NaN MaxAttenuationDB or
// a non-finite MinPriorSNRDB, which resolve rejects, +Inf here is intentional.
func TestMaxAttenuationInfAccepted(t *testing.T) {
	cfg := Config{SampleRate: 48000, Params: &Params{
		MaxAttenuationDB: float32(math.Inf(1)),
		OverSubtraction:  1,
		SNRSmoothing:     0.9,
		MinPriorSNRDB:    -18,
		TrackWindowSec:   1,
	}}
	if _, _, err := cfg.resolve(); err != nil {
		t.Fatalf("+Inf MaxAttenuationDB should be accepted, got %v", err)
	}
}
