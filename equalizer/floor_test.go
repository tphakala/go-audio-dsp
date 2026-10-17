package equalizer

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// TestResponseIsJSONEncodable pins that no response value is -Inf or +Inf for a
// finite frequency, so encoding/json accepts it. An RBJ notch evaluated at its
// own f0 is the case that used to reach -Inf.
func TestResponseIsJSONEncodable(t *testing.T) {
	const fs = 48000
	freqs := []float64{0, 440, 1000, 24000, 1e308}
	bw, err := NewButterworthBandPass(900, 1100, 4, fs)
	if err != nil {
		t.Fatal(err)
	}
	null, err := NewBiquadFilter([]Biquad{{}}, fs)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]Band{
		"notch 440":  {Type: BandReject, Frequency: 440, WidthHz: 440.0 / 30},
		"notch 1000": {Type: BandReject, Frequency: 1000, WidthHz: 1000.0 / 30},
		"highpass":   {Type: HighPass, Frequency: 1000, Q: 0.7071},
		"lowpass":    {Type: LowPass, Frequency: 100, Q: 0.7071},
	}
	check := func(name string, pts []ResponsePoint) {
		t.Helper()
		if _, err := json.Marshal(pts); err != nil {
			t.Errorf("%s: json.Marshal: %v", name, err)
		}
		for _, p := range pts {
			if math.IsInf(p.GainDB, 0) || math.IsNaN(p.GainDB) {
				t.Errorf("%s at %g Hz: GainDB = %v, want finite", name, p.Hz, p.GainDB)
			}
		}
	}
	for name, b := range cases {
		f := mustFilter(t, b, fs)
		e, err := New(Config{SampleRate: fs, Bands: []Band{b}})
		if err != nil {
			t.Fatal(err)
		}
		c := NewFilterChain()
		if err := c.AddFilter(f); err != nil {
			t.Fatal(err)
		}
		check(name+" filter", f.Response(freqs))
		check(name+" equalizer", e.Response(freqs))
		check(name+" chain", c.Response(freqs))
	}
	check("butterworth", bw.Response(freqs))
	check("biquad null", null.Response(freqs))
}

// TestResponseHugeFrequencyIsFinite pins that a huge finite frequency is
// reduced modulo the sample rate instead of overflowing to NaN.
func TestResponseHugeFrequencyIsFinite(t *testing.T) {
	f := mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6}, 48000)
	p := f.Response([]float64{1e308})[0]
	if math.IsNaN(p.GainDB) || math.IsNaN(p.PhaseRad) {
		t.Errorf("Response(1e308) = %+v, want finite", p)
	}
	// The response is periodic in the sample rate.
	a := f.Response([]float64{1000})[0]
	b := f.Response([]float64{1000 + 48000})[0]
	if math.Abs(a.GainDB-b.GainDB) > 1e-9 {
		t.Errorf("GainDB at 1000 Hz = %v, at 49000 Hz = %v, want equal", a.GainDB, b.GainDB)
	}
}

// TestLogSweepRejectsDegenerateRanges pins that a range with no room for n
// distinct float64 points, or an infinite top, is an error rather than a grid
// with duplicate, out-of-range or infinite points.
func TestLogSweepRejectsDegenerateRanges(t *testing.T) {
	for _, c := range []struct {
		name   string
		lo, hi float64
		n      int
	}{
		{"fHigh +Inf", 20, math.Inf(1), 10},
		{"fHigh +Inf, two points", 20, math.Inf(1), 2}, // without the guard this is [20 +Inf]
		{"one ulp wide", 20, math.Nextafter(20, 21), 5},
		{"1e-11 wide", 1000, 1000 + 1e-11, 256},
		{"float64 limit", math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64, 3},
	} {
		if _, err := LogSweep(c.lo, c.hi, c.n); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("LogSweep(%s) err = %v, want ErrInvalidConfig", c.name, err)
		}
	}
}

// TestResponseSectionFloor pins the per-section floor: an RBJ notch at its own
// f0 reads SectionFloorDB exactly (per pass), a value above the floor is
// untouched, and a non-finite frequency stays NaN.
func TestResponseSectionFloor(t *testing.T) {
	const fs = 48000
	for _, f0 := range []float64{440, 1000, 3137, 5000, 15000} {
		for _, passes := range []int{1, 16} {
			e, err := New(Config{SampleRate: fs, Bands: []Band{{Type: BandReject, Frequency: f0, WidthHz: f0 / 30, Passes: passes}}})
			if err != nil {
				t.Fatal(err)
			}
			if g, want := e.Response([]float64{f0})[0].GainDB, float64(passes)*SectionFloorDB; g != want {
				t.Errorf("notch %g Hz, %d passes: GainDB = %v, want exactly %v", f0, passes, g, want)
			}
		}
	}
	if g := gainAt(t, band(HighPass, 1000, 0.7071, 0, 0, 0), 0.1); math.Abs(g-(-160.02)) > 0.01 {
		t.Errorf("HighPass 1 kHz at 0.1 Hz = %.4f dB, want -160.02 (above the floor, unchanged)", g)
	}
	e, err := New(band(Peaking, 1000, 0, 400, 6, 0))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if g := e.Response([]float64{f})[0].GainDB; !math.IsNaN(g) {
			t.Errorf("Response(%v) GainDB = %v, want NaN", f, g)
		}
	}
}
