package equalizer

import (
	"errors"
	"math"
	"strconv"
	"testing"
)

// thirdOctave returns the edges of the base-10 one-third-octave band centred on
// centre, the centre times 10^(-0.05) and 10^(+0.05).
func thirdOctave(centre float64) (lower, upper float64) {
	return centre * math.Pow(10, -0.05), centre * math.Pow(10, 0.05)
}

// butterworthDB is the analytic magnitude in dB of a digital Butterworth
// band-pass of prototype order n with its -3.01 dB points on lower and upper
// (bilinear map with prewarped edges), before the gain normalization: with
// O = tan(pi f / fs), the low-pass to band-pass substitution gives
// |H|^2 = 1 / (1 + x^(2n)), x = (O^2 - O1*O2) / ((O2 - O1) * O).
func butterworthDB(lower, upper, fs float64, n int, f float64) float64 {
	o1, o2 := math.Tan(math.Pi*lower/fs), math.Tan(math.Pi*upper/fs)
	o := math.Tan(math.Pi * f / fs)
	x := (o*o - o1*o2) / ((o2 - o1) * o)
	return -10 * math.Log10(1+math.Pow(x*x, float64(n)))
}

// TestButterworthBandPassMatchesAnalytic compares the realised cascade with the
// analytic Butterworth magnitude, normalized to 0 dB at the geometric centre, so
// a missing edge prewarp, a mispaired root, a wrong real-pole factor or a wrong
// section gain moves the response off the oracle.
func TestButterworthBandPassMatchesAnalytic(t *testing.T) {
	type bandSpec struct {
		name  string
		edges func(fs float64) (lower, upper float64)
	}
	third := func(centre float64) bandSpec {
		return bandSpec{
			name:  "third-octave " + strconv.FormatFloat(centre, 'g', -1, 64),
			edges: func(float64) (float64, float64) { return thirdOctave(centre) },
		}
	}
	// nearNyquist is the third-octave band whose upper edge sits at frac of the
	// Nyquist frequency. At 0.99 the prototype's real pole maps to two real
	// roots, the regime a complex-roots-only design would get wrong.
	nearNyquist := func(frac float64) bandSpec {
		return bandSpec{
			name: "third-octave at " + strconv.FormatFloat(frac, 'g', -1, 64) + " of Nyquist",
			edges: func(fs float64) (float64, float64) {
				upper := frac * fs / 2
				return upper * math.Pow(10, -0.1), upper
			},
		}
	}
	specs := []bandSpec{
		third(25.118864315095802), // the 25 Hz band: the narrowest poles
		third(1000),
		nearNyquist(0.95),
		nearNyquist(0.99),
		{"wide 100 Hz to 10 kHz", func(float64) (float64, float64) { return 100, 10000 }},
	}
	for _, order := range []int{1, 2, 3, 4, 8} {
		for _, rate := range []int{44100, 48000, 96000, 192000, 384000} {
			for _, sp := range specs {
				t.Run(strconv.Itoa(order)+"/"+strconv.Itoa(rate)+"/"+sp.name, func(t *testing.T) {
					t.Parallel()
					fs := float64(rate)
					lower, upper := sp.edges(fs)
					f, err := NewButterworthBandPass(lower, upper, order, rate)
					if err != nil {
						t.Fatal(err)
					}
					centre := math.Sqrt(lower * upper)
					ref := butterworthDB(lower, upper, fs, order, centre)
					var freqs []float64
					for i := -200; i <= 200; i++ {
						if fr := centre * math.Exp2(float64(i)/50); fr < fs/2 {
							freqs = append(freqs, fr)
						}
					}
					var worst, worstAt float64
					checked := 0
					for i, p := range f.Response(freqs) {
						want := butterworthDB(lower, upper, fs, order, freqs[i]) - ref
						if want < -150 {
							continue // the realised cascade bottoms out in round-off here
						}
						checked++
						if d := math.Abs(p.GainDB - want); d > worst {
							worst, worstAt = d, freqs[i]
						}
					}
					if checked < 100 {
						t.Fatalf("only %d points above -150 dB; the test is not covering the response", checked)
					}
					if worst > 1e-5 {
						t.Errorf("max |realised - analytic| = %.3g dB at %.4g Hz, want <= 1e-5", worst, worstAt)
					}
				})
			}
		}
	}
}

// TestButterworthBandPassLevels pins the levels at the anchor points of a
// one-third-octave band at 1 kHz, 48 kHz, far from Nyquist: 0 dB at the
// geometric centre and -3.0103 dB on both edges, with one section per prototype
// pole.
func TestButterworthBandPassLevels(t *testing.T) {
	lower, upper := thirdOctave(1000)
	for _, order := range []int{1, 2, 3, 4, 5} {
		f, err := NewButterworthBandPass(lower, upper, order, 48000)
		if err != nil {
			t.Fatal(err)
		}
		if f.NumSections() != order {
			t.Errorf("order %d: NumSections = %d, want %d", order, f.NumSections(), order)
		}
		pts := f.Response([]float64{math.Sqrt(lower * upper), lower, upper})
		if g := pts[0].GainDB; math.Abs(g) > 1e-9 {
			t.Errorf("order %d: centre gain = %.3g dB, want 0", order, g)
		}
		for _, p := range pts[1:] {
			if math.Abs(p.GainDB-(-3.010299956639812)) > 1e-6 {
				t.Errorf("order %d: gain at edge %.4g Hz = %.7f dB, want -3.0103", order, p.Hz, p.GainDB)
			}
		}
	}
}

// TestButterworthBandPassEdgeLevelNearNyquist pins what the constructor
// documents about the edge level: the gain is normalized at the geometric
// centre, which lies a little below the peak, so an edge reads above -3.0103 dB.
// For order 4 and a one-third-octave band at 0.95 of Nyquist the difference is
// within 0.002 dB; for order 1 and an upper edge of 21 kHz at 48 kHz it is
// 0.22 dB.
func TestButterworthBandPassEdgeLevelNearNyquist(t *testing.T) {
	const rate = 48000
	edge := func(order int, upper float64) float64 {
		lower := upper * math.Pow(10, -0.1)
		f, err := NewButterworthBandPass(lower, upper, order, rate)
		if err != nil {
			t.Fatal(err)
		}
		return f.Response([]float64{upper})[0].GainDB
	}
	if g := edge(4, 0.95*rate/2); math.Abs(g-(-3.0103)) > 0.002 || g < -3.0103 {
		t.Errorf("order 4 at 0.95 of Nyquist: upper edge reads %.4f dB, want -3.0103 within 0.002 and not below it", g)
	}
	if g := edge(1, 21000); math.Abs(g-(-2.787)) > 0.005 {
		t.Errorf("order 1, upper edge 21 kHz at 48 kHz: reads %.4f dB, want about -2.787", g)
	}
}

// TestButterworthBandPassInvalid pins the construction checks: every rejection
// wraps ErrInvalidConfig and returns no filter.
func TestButterworthBandPassInvalid(t *testing.T) {
	tests := []struct {
		name         string
		lower, upper float64
		order, rate  int
	}{
		{"zero rate", 900, 1100, 3, 0},
		{"negative rate", 900, 1100, 3, -48000},
		{"order 0", 900, 1100, 0, 48000},
		{"negative order", 900, 1100, -1, 48000},
		{"order too large", 900, 1100, MaxPasses + 1, 48000},
		{"lower 0", 0, 1100, 3, 48000},
		{"lower negative", -5, 1100, 3, 48000},
		{"lower NaN", math.NaN(), 1100, 3, 48000},
		{"upper NaN", 900, math.NaN(), 3, 48000},
		{"upper = lower", 900, 900, 3, 48000},
		{"upper < lower", 1100, 900, 3, 48000},
		{"upper = Nyquist", 900, 24000, 3, 48000},
		{"upper above Nyquist", 900, 30000, 3, 48000},
		{"upper +Inf", 900, math.Inf(1), 3, 48000},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, err := NewButterworthBandPass(tc.lower, tc.upper, tc.order, tc.rate)
			if !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("err = %v, want ErrInvalidConfig", err)
			}
			if f != nil {
				t.Error("filter is non-nil on error")
			}
		})
	}
}

// TestButterworthBandPassRejectsPrecisionCollapse covers the stability check: at
// a sample rate so high that a 25 Hz band collapses toward DC, the poles round
// onto the unit circle in float64, and the constructor reports it instead of
// returning a filter that rings forever.
func TestButterworthBandPassRejectsPrecisionCollapse(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("the rate needs a 64-bit int")
	}
	rate := int(math.Pow10(14))
	lower, upper := thirdOctave(25.118864315095802)
	if f, err := NewButterworthBandPass(lower, upper, 4, rate); !errors.Is(err, ErrInvalidConfig) || f != nil {
		t.Errorf("25 Hz band at 1e14 Hz: got (%v, %v), want an error wrapping ErrInvalidConfig", f, err)
	}
}

// TestButterworthBandPassImpulseEnergy pins that the Direct Form I recurrence
// realises the design for the narrowest poles: the energy of the impulse
// response over 3 s equals the frequency-domain energy (Parseval) of the
// designed magnitude. A recurrence that lost precision, or a filter that had not
// decayed, would differ.
func TestButterworthBandPassImpulseEnergy(t *testing.T) {
	const rate = 192000
	lower, upper := thirdOctave(25.118864315095802)
	f, err := NewButterworthBandPass(lower, upper, 4, rate)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float64, 3*rate)
	buf[0] = 1
	f.ApplyFloat64(buf)
	var timeEnergy float64
	for _, v := range buf {
		timeEnergy += v * v
	}

	// (1/pi) * integral over [0, pi] of |H|^2 dw, trapezoid on a log frequency
	// grid dense enough for the 25 Hz band at 192 kHz.
	const points = 400000
	lo, hi := math.Log(0.01), math.Log(rate/2)
	freqs := make([]float64, points+1)
	for i := range freqs {
		freqs[i] = math.Exp(lo + (hi-lo)*float64(i)/points)
	}
	var freqEnergy, prevF, prevG float64
	for i, p := range f.Response(freqs) {
		g := math.Pow(10, p.GainDB/10)
		freqEnergy += (g + prevG) / 2 * (freqs[i] - prevF)
		prevF, prevG = freqs[i], g
	}
	freqEnergy /= rate / 2

	if rel := math.Abs(timeEnergy-freqEnergy) / freqEnergy; rel > 1e-6 {
		t.Errorf("impulse energy %.9g vs frequency-domain %.9g: relative difference %.3g, want <= 1e-6", timeEnergy, freqEnergy, rel)
	}
}
