package equalizer

import (
	"errors"
	"math"
	"testing"
)

// gainAt builds an equalizer for cfg and returns its response magnitude in dB at
// frequency f.
func gainAt(t *testing.T, cfg Config, f float64) float64 {
	t.Helper()
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e.Response([]float64{f})[0].GainDB
}

func band(ft FilterType, freq, q, width, gain float64, passes int) Config {
	return Config{SampleRate: 48000, Bands: []Band{
		{Type: ft, Frequency: freq, Q: q, WidthHz: width, GainDB: gain, Passes: passes},
	}}
}

// TestResponseKnownPoints checks the magnitude response against analytic facts
// of each filter type. These hold independently of the coefficient formulas, so
// a wrong sign or a missing a0 normalization moves the response off the known
// value and fails here.
func TestResponseKnownPoints(t *testing.T) {
	const butter = 0.7071067811865476 // 1/sqrt(2), Butterworth Q: -3.01 dB at f0

	// Second-order Butterworth low/high pass are -3.01 dB at the corner.
	if g := gainAt(t, band(HighPass, 1000, butter, 0, 0, 0), 1000); math.Abs(g-(-3.0103)) > 0.05 {
		t.Errorf("HighPass at f0 = %.4f dB, want -3.01", g)
	}
	if g := gainAt(t, band(LowPass, 1000, butter, 0, 0, 0), 1000); math.Abs(g-(-3.0103)) > 0.05 {
		t.Errorf("LowPass at f0 = %.4f dB, want -3.01", g)
	}
	// High-pass passes the top and rejects far below the corner.
	if g := gainAt(t, band(HighPass, 1000, butter, 0, 0, 0), 8000); math.Abs(g) > 0.1 {
		t.Errorf("HighPass at 8*f0 = %.4f dB, want ~0", g)
	}
	if g := gainAt(t, band(HighPass, 1000, butter, 0, 0, 0), 125); g > -20 {
		t.Errorf("HighPass three octaves below f0 = %.4f dB, want < -20", g)
	}
	// Peaking has gain exactly GainDB at its center.
	if g := gainAt(t, band(Peaking, 1000, 0, 200, 6, 0), 1000); math.Abs(g-6) > 0.02 {
		t.Errorf("Peaking(+6) at f0 = %.4f dB, want 6", g)
	}
	if g := gainAt(t, band(Peaking, 1000, 0, 200, -9, 0), 1000); math.Abs(g-(-9)) > 0.02 {
		t.Errorf("Peaking(-9) at f0 = %.4f dB, want -9", g)
	}
	// All-pass is unity magnitude everywhere.
	for _, f := range []float64{50, 1000, 10000} {
		if g := gainAt(t, band(AllPass, 1000, 2, 0, 0, 0), f); math.Abs(g) > 1e-4 {
			t.Errorf("AllPass at %g Hz = %g dB, want 0", f, g)
		}
	}
	// Band-reject is a deep notch at the center (finite, very negative; -Inf only
	// when the magnitude evaluates to exactly zero).
	if g := gainAt(t, band(BandReject, 1000, 0, 100, 0, 0), 1000); g > -100 {
		t.Errorf("BandReject at f0 = %.4f dB, want < -100", g)
	}
	// Low-shelf: gain at DC is GainDB, far above the corner it returns to 0 dB.
	if g := gainAt(t, band(LowShelf, 1000, butter, 0, 6, 0), 0); math.Abs(g-6) > 0.02 {
		t.Errorf("LowShelf(+6) at DC = %.4f dB, want 6", g)
	}
	if g := gainAt(t, band(LowShelf, 1000, butter, 0, 6, 0), 20000); math.Abs(g) > 0.2 {
		t.Errorf("LowShelf(+6) well above f0 = %.4f dB, want ~0", g)
	}
	// High-shelf: gain near Nyquist is GainDB, far below the corner it is 0 dB.
	if g := gainAt(t, band(HighShelf, 1000, butter, 0, -6, 0), 20000); math.Abs(g-(-6)) > 0.2 {
		t.Errorf("HighShelf(-6) near Nyquist = %.4f dB, want -6", g)
	}
	if g := gainAt(t, band(HighShelf, 1000, butter, 0, -6, 0), 20); math.Abs(g) > 0.2 {
		t.Errorf("HighShelf(-6) well below f0 = %.4f dB, want ~0", g)
	}
	// Passes cascades identical sections: two passes double the dB.
	one := gainAt(t, band(HighPass, 1000, butter, 0, 0, 1), 1000)
	two := gainAt(t, band(HighPass, 1000, butter, 0, 0, 2), 1000)
	if math.Abs(two-2*one) > 1e-9 {
		t.Errorf("Passes=2 gain %.6f dB, want 2x Passes=1 gain %.6f dB", two, 2*one)
	}
}

// TestResponseNonFiniteFrequency pins the documented behavior that a non-finite
// frequency yields a NaN GainDB (rather than -Inf, which means an exact null).
func TestResponseNonFiniteFrequency(t *testing.T) {
	e, err := New(band(HighPass, 1000, 0.7071, 0, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if g := e.Response([]float64{math.NaN()})[0].GainDB; !math.IsNaN(g) {
		t.Errorf("Response(NaN) GainDB = %v, want NaN", g)
	}
}

func rms(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)))
}

// TestResponseAgainstSweptSine drives steady sines through ProcessInto and
// checks the measured gain matches what Response predicts. This validates the
// time-domain recurrence and state handling, which Response does not share, so
// it catches a recurrence bug even where Response would agree with itself.
func TestResponseAgainstSweptSine(t *testing.T) {
	const fs = 48000
	cfg := testConfig()
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []float64{100, 300, 1000, 3000, 8000} {
		e.Reset()
		const n = fs // 1 second, whole cycles dominate
		in := make([]float32, n)
		for i := range in {
			in[i] = float32(math.Sin(2 * math.Pi * f * float64(i) / fs))
		}
		out := make([]float32, n)
		if _, err := e.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
		half := n / 2 // discard the transient
		measured := 20 * math.Log10(rms(out[half:])/rms(in[half:]))
		if math.IsNaN(measured) || math.IsInf(measured, 0) {
			t.Fatalf("f=%g Hz: measured gain is non-finite (%g)", f, measured)
		}
		predicted := e.Response([]float64{f})[0].GainDB
		if math.Abs(measured-predicted) > 0.3 {
			t.Errorf("f=%g Hz: measured %.3f dB, Response predicts %.3f dB", f, measured, predicted)
		}
	}
}

func TestLogSweep(t *testing.T) {
	freqs, err := LogSweep(20, 20000, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(freqs) != 100 {
		t.Fatalf("len = %d, want 100", len(freqs))
	}
	if freqs[0] != 20 || freqs[99] != 20000 {
		t.Fatalf("endpoints = %g, %g, want 20, 20000", freqs[0], freqs[99])
	}
	ratio := freqs[1] / freqs[0]
	for i := 1; i < len(freqs); i++ {
		if r := freqs[i] / freqs[i-1]; math.Abs(r-ratio) > 1e-12 {
			t.Fatalf("ratio at %d = %.15g, want constant %.15g", i, r, ratio)
		}
	}

	for _, c := range []struct {
		name             string
		lo, hi           float64
		n                int
	}{
		{"fLow 0", 0, 100, 10},
		{"fLow negative", -1, 100, 10},
		{"fHigh <= fLow", 100, 100, 10},
		{"n < 2", 20, 20000, 1},
	} {
		if _, err := LogSweep(c.lo, c.hi, c.n); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("LogSweep(%s) err = %v, want ErrInvalidConfig", c.name, err)
		}
	}
}
