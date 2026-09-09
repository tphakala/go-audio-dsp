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
	if p != Medium.Params() {
		t.Fatalf("params = %+v, want Medium", p)
	}
	custom := Params{MaxAttenuationDB: 3, OverSubtraction: 1, SNRSmoothing: 0.9, MinPriorSNRDB: -10, Estimator: Wiener, TrackWindowSec: 1}
	_, p, err = Config{SampleRate: 16000, Preset: Heavy, Params: &custom}.resolve()
	if err != nil {
		t.Fatal(err)
	}
	if p != custom {
		t.Fatalf("explicit Params not used: %+v", p)
	}
}

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
		{SampleRate: 48000, Preset: Preset(99)},
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

func TestPresetsAreOrdered(t *testing.T) {
	l, m, h := Light.Params(), Medium.Params(), Heavy.Params()
	if !(l.MaxAttenuationDB < m.MaxAttenuationDB && m.MaxAttenuationDB < h.MaxAttenuationDB) {
		t.Errorf("MaxAttenuationDB not increasing: %g %g %g", l.MaxAttenuationDB, m.MaxAttenuationDB, h.MaxAttenuationDB)
	}
	for _, p := range []Preset{Light, Medium, Heavy} {
		if err := p.Params().validate(); err != nil {
			t.Errorf("%v: %v", p, err)
		}
		if p.String() == "" {
			t.Errorf("%d has empty String()", p)
		}
	}
	if Preset(0) != Medium {
		t.Error("zero Preset must be Medium")
	}
	if Estimator(0) != MMSELSA {
		t.Error("zero Estimator must be MMSELSA")
	}
}
