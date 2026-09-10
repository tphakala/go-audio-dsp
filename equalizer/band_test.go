package equalizer

import (
	"errors"
	"math"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

func TestFilterTypeString(t *testing.T) {
	cases := map[FilterType]string{
		LowPass:       "low-pass",
		HighPass:      "high-pass",
		AllPass:       "all-pass",
		BandPass:      "band-pass",
		BandReject:    "band-reject",
		LowShelf:      "low-shelf",
		HighShelf:     "high-shelf",
		Peaking:       "peaking",
		FilterType(0): "FilterType(0)",
		FilterType(9): "FilterType(9)",
	}
	for ft, want := range cases {
		if got := ft.String(); got != want {
			t.Errorf("FilterType(%d).String() = %q, want %q", int(ft), got, want)
		}
	}
}

func TestConfigInvalid(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"sample rate 0", Config{SampleRate: 0}},
		{"sample rate negative", Config{SampleRate: -1}},
		{"zero band (invalid type)", Config{SampleRate: 48000, Bands: []Band{{}}}},
		{"unknown type", Config{SampleRate: 48000, Bands: []Band{{Type: FilterType(99), Frequency: 1000, Q: 1}}}},
		{"frequency 0", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 0, Q: 1}}}},
		{"frequency NaN", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: math.NaN(), Q: 1}}}},
		{"frequency at nyquist", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 24000, Q: 1}}}},
		{"frequency above nyquist", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 30000, Q: 1}}}},
		{"Q zero", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 1000, Q: 0}}}},
		{"Q negative", Config{SampleRate: 48000, Bands: []Band{{Type: HighPass, Frequency: 1000, Q: -1}}}},
		{"Q NaN", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 1000, Q: math.NaN()}}}},
		{"width zero", Config{SampleRate: 48000, Bands: []Band{{Type: BandPass, Frequency: 1000, WidthHz: 0}}}},
		{"width negative", Config{SampleRate: 48000, Bands: []Band{{Type: BandReject, Frequency: 1000, WidthHz: -5}}}},
		{"width too wide", Config{SampleRate: 48000, Bands: []Band{{Type: BandReject, Frequency: 100, WidthHz: 300}}}},
		{"peaking gain NaN", Config{SampleRate: 48000, Bands: []Band{{Type: Peaking, Frequency: 1000, WidthHz: 100, GainDB: math.NaN()}}}},
		{"shelf gain Inf", Config{SampleRate: 48000, Bands: []Band{{Type: LowShelf, Frequency: 1000, Q: 1, GainDB: math.Inf(1)}}}},
		{"passes negative", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 1000, Q: 1, Passes: -1}}}},
		{"passes too many", Config{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 1000, Q: 1, Passes: maxPasses + 1}}}},
		// A finite, in-range frequency near Nyquist can still push the poles onto
		// the unit circle (marginally unstable); the coefficient check rejects it.
		{"near-nyquist unstable", Config{SampleRate: 48000, Bands: []Band{{Type: BandReject, Frequency: 23999, WidthHz: 100}}}},
		// A finite but absurd gain overflows the section coefficients.
		{"gain overflows coefficients", Config{SampleRate: 48000, Bands: []Band{{Type: Peaking, Frequency: 1000, WidthHz: 100, GainDB: 4000}}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			_, err := New(c.cfg)
			if !errors.Is(err, dsp.ErrInvalidConfig) {
				t.Fatalf("New(%s) err = %v, want dsp.ErrInvalidConfig", c.name, err)
			}
		})
	}
}

// TestConfigLenient checks that a field a filter type does not use is ignored,
// not rejected: a LowPass with a stray GainDB and WidthHz still builds, and a
// BandPass with a stray Q still builds.
func TestConfigLenient(t *testing.T) {
	cfgs := []Config{
		{SampleRate: 48000, Bands: []Band{{Type: LowPass, Frequency: 1000, Q: 1, GainDB: 12, WidthHz: 50}}},
		{SampleRate: 48000, Bands: []Band{{Type: BandPass, Frequency: 1000, WidthHz: 200, Q: 5}}},
	}
	for i, cfg := range cfgs {
		if _, err := New(cfg); err != nil {
			t.Errorf("cfg %d: New with ignored fields set failed: %v", i, err)
		}
	}
}

func TestPassesZeroMeansOne(t *testing.T) {
	cases := []struct {
		passes int
		want   int
	}{{0, 1}, {1, 1}, {3, 3}}
	for _, c := range cases {
		e, err := New(Config{SampleRate: 48000, Bands: []Band{{Type: HighPass, Frequency: 100, Q: 0.7071, Passes: c.passes}}})
		if err != nil {
			t.Fatal(err)
		}
		if e.NumSections() != c.want {
			t.Errorf("Passes %d: NumSections() = %d, want %d", c.passes, e.NumSections(), c.want)
		}
	}
}
