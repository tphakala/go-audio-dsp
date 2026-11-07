package denoiser

import (
	"errors"
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/internal/audiotest"
)

func TestAutoFrameSize(t *testing.T) {
	cases := map[int]int{
		8000: 128, 16000: 256, 22050: 512, 24000: 512, 32000: 512,
		44100: 1024, 48000: 1024, 96000: 2048,
	}
	for sr, want := range cases {
		rc, _, err := Config{SampleRate: sr}.resolve()
		if err != nil || rc.FrameSize != want {
			t.Errorf("resolve(%d) FrameSize = %d (err %v), want %d", sr, rc.FrameSize, err, want)
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
	l, lm, m, h := ParamsFor(Light), ParamsFor(LightMedium), ParamsFor(Medium), ParamsFor(Heavy)
	if !(l.MaxAttenuationDB < lm.MaxAttenuationDB && lm.MaxAttenuationDB < m.MaxAttenuationDB && m.MaxAttenuationDB < h.MaxAttenuationDB) {
		t.Errorf("MaxAttenuationDB not increasing: %g %g %g %g", l.MaxAttenuationDB, lm.MaxAttenuationDB, m.MaxAttenuationDB, h.MaxAttenuationDB)
	}
	// Every strength Valid admits must index a populated table row.
	for s := Strength(0); s.Valid(); s++ {
		if err := ParamsFor(s).validate(); err != nil {
			t.Errorf("%v: %v", s, err)
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

// TestLightMediumIsLightWithOverrides pins LightMedium to the configuration
// that was evaluated on real recordings (Light with seven knobs overridden) and
// checks the Strength path and the explicit-Params path give bit-identical
// output through the two-pass learned-profile Denoise.
func TestLightMediumIsLightWithOverrides(t *testing.T) {
	want := ParamsFor(Light)
	want.MaxAttenuationDB = 9
	want.OverSubtraction = 1.1
	want.SNRSmoothing = 0.955
	want.MinPriorSNRDB = -16.5
	want.FreqSmoothBins = 0
	want.Estimator = MMSELSA
	want.TrackWindowSec = 2
	if got := ParamsFor(LightMedium); got != want {
		t.Fatalf("ParamsFor(LightMedium) = %+v, want %+v", got, want)
	}
	clip := audiotest.MakeSynthClip(48000, -40, -20, false, 7)
	viaStrength, err := Denoise(clip.Mix, Config{SampleRate: 48000, Strength: LightMedium})
	if err != nil {
		t.Fatal(err)
	}
	viaParams, err := Denoise(clip.Mix, Config{SampleRate: 48000, Params: &want})
	if err != nil {
		t.Fatal(err)
	}
	if len(viaStrength) != len(viaParams) {
		t.Fatalf("length %d != %d", len(viaStrength), len(viaParams))
	}
	for i := range viaParams {
		if math.Float32bits(viaStrength[i]) != math.Float32bits(viaParams[i]) {
			t.Fatalf("differs at %d: %g vs %g", i, viaStrength[i], viaParams[i])
		}
	}
}
