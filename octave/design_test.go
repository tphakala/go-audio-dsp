package octave

import (
	"math"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/equalizer"
)

// filterNoiseMS is the mean square a filter reads for white noise of the given
// variance, from its numerically integrated power response:
// variance * (2/fs) * integral over [0, fs/2] of |H(f)|^2 df. The grid is
// logarithmic from 0.02 Hz to Nyquist, which resolves the 25 Hz band at high
// rates where a linear grid is too coarse; the band-pass response below 0.02 Hz
// is negligible against the band's own level.
func filterNoiseMS(f *equalizer.Filter, rate int, variance float64) float64 {
	nyq := float64(rate) / 2
	const points = 100000
	lo, hi := math.Log(0.02), math.Log(nyq)
	freqs := make([]float64, points+1)
	for i := range freqs {
		freqs[i] = math.Exp(lo + (hi-lo)*float64(i)/points)
	}
	var sum, prevF, prevG float64
	for i, p := range f.Response(freqs) {
		g := math.Pow(10, p.GainDB/10)
		sum += (g + prevG) / 2 * (freqs[i] - prevF)
		prevF, prevG = freqs[i], g
	}
	return variance * sum / nyq
}

func TestFilterDesignString(t *testing.T) {
	cases := []struct {
		d    FilterDesign
		want string
	}{
		{FilterBiquad, "biquad"},
		{FilterButterworth, "butterworth"},
		{FilterDesign(7), "FilterDesign(7)"},
		{FilterDesign(-1), "FilterDesign(-1)"},
	}
	for _, c := range cases {
		if got := c.d.String(); got != c.want {
			t.Errorf("FilterDesign(%d).String() = %q, want %q", int(c.d), got, c.want)
		}
	}
	if FilterBiquad != 0 {
		t.Error("the zero FilterDesign must be FilterBiquad: the default design is the zero value")
	}
}

// TestNewAcceptsButterworth pins the accepted edge of the Butterworth
// constraints that TestNewErrors rejects just beyond: Order left at 0 and a
// NyquistMargin up to 0.95.
func TestNewAcceptsButterworth(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"defaults", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth}},
		{"margin 0.95", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, NyquistMargin: 0.95}},
		{"margin 0.5", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, NyquistMargin: 0.5}},
		{"explicit biquad design", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterBiquad, Order: 3, NyquistMargin: 1}},
	}
	for _, c := range cases {
		if _, err := New(c.cfg); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
	}
}

// TestButterworthNoiseBias pins the level of broadband noise in the Butterworth
// profile: about +0.11 dB above the rectangular ideal, against +1.93 dB for the
// default design and +0.20 dB for a prototype of order 3, so the order cannot
// drop unnoticed. The bias is computed from the integrated response of each
// band's filter. Near Nyquist the bilinear warping lowers it a little (top band
// at 48 kHz: +0.085 dB).
func TestButterworthNoiseBias(t *testing.T) {
	for _, rate := range []int{48000, 96000} {
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: FilterButterworth})
		for i, b := range m.Bands() {
			bias := db(filterNoiseMS(m.filters[i], rate, 1) / ((b.Upper - b.Lower) / (float64(rate) / 2)))
			if bias < 0.07 || bias > 0.12 {
				t.Errorf("%d Hz band %v: noise bias %+.4f dB, want within [+0.07, +0.12]", rate, b.Nominal, bias)
			}
			if b.Upper < 0.3*float64(rate)/2 && math.Abs(bias-0.111) > 0.005 {
				t.Errorf("%d Hz band %v: noise bias %+.4f dB, want +0.111 within 0.005", rate, b.Nominal, bias)
			}
		}
	}
}

// TestButterworthMatchesFilterReference pins that the meter's Butterworth bands
// are the equalizer's Butterworth band-pass of the documented prototype order
// over each band's edges: the per-period mean square equals a reference that
// runs the whole input through one such filter and accumulates the squares
// sample by sample. The arithmetic is the same operation sequence, so the
// comparison is exact.
func TestButterworthMatchesFilterReference(t *testing.T) {
	for _, rate := range []int{48000, 192000} {
		in := noise(2*rate+777, 5, 0.5)
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: FilterButterworth})
		ps := periods(m, in)
		for i, b := range m.Bands() {
			f, err := equalizer.NewButterworthBandPass(b.Lower, b.Upper, butterworthOrder, rate)
			if err != nil {
				t.Fatal(err)
			}
			y := make([]float64, len(in))
			for j, v := range in {
				y[j] = float64(v)
			}
			f.ApplyFloat64(y)
			for p := range ps {
				var acc float64
				for _, v := range y[p*rate : (p+1)*rate] {
					acc += v * v
				}
				if want := acc / float64(rate); ps[p][i] != want {
					t.Errorf("%d Hz band %v period %d: %.17g, reference %.17g", rate, b.Nominal, p, ps[p][i], want)
				}
			}
		}
	}
}

// TestBandFilterIndependentOfSelection pins that a band's filter depends only on
// the band, the rate and the design: a single-band meter reads exactly what the
// same band reads in the full meter.
func TestBandFilterIndependentOfSelection(t *testing.T) {
	const rate = 48000
	in := noise(3*rate/2, 9, 0.5)
	for _, design := range designs {
		full := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate / 2, Filter: design})
		want := periods(full, in)
		for i, b := range full.Bands() {
			single := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate / 2, MinFreq: b.Nominal, MaxFreq: b.Nominal, Filter: design})
			if single.NumBands() != 1 {
				t.Fatalf("%v band %v: single-band meter has %d bands", design, b.Nominal, single.NumBands())
			}
			got := periods(single, in)
			if len(got) != len(want) {
				t.Fatalf("%v band %v: %d periods, want %d", design, b.Nominal, len(got), len(want))
			}
			for p := range got {
				if got[p][0] != want[p][i] {
					t.Errorf("%v band %v period %d: single-band %.17g, full meter %.17g", design, b.Nominal, p, got[p][0], want[p][i])
				}
			}
		}
	}
}

// TestDefaultDesignUnchanged pins that the zero Filter is the biquad design of
// earlier releases: naming FilterBiquad changes nothing, and every band equals
// an equalizer RBJ band-pass with the octave-width formula, bit for bit.
func TestDefaultDesignUnchanged(t *testing.T) {
	const rate = 48000
	in := noise(2*rate+123, 3, 0.5)
	def := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate})
	want := periods(def, in)
	explicit := periods(mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: FilterBiquad}), in)
	if len(explicit) != len(want) {
		t.Fatalf("%d periods, want %d", len(explicit), len(want))
	}
	for p := range want {
		if !slices.Equal(explicit[p], want[p]) {
			t.Fatalf("period %d: explicit FilterBiquad differs from the zero value", p)
		}
	}

	r := math.Pow(10, 0.1)
	for i, b := range def.Bands() {
		f, err := equalizer.NewFilter(equalizer.Band{
			Type: equalizer.BandPass, Frequency: b.Exact, WidthHz: 2 * b.Exact * (r - 1) / (r + 1), Passes: 1,
		}, rate)
		if err != nil {
			t.Fatal(err)
		}
		y := make([]float64, len(in))
		for j, v := range in {
			y[j] = float64(v)
		}
		f.ApplyFloat64(y)
		for p := range want {
			var acc float64
			for _, v := range y[p*rate : (p+1)*rate] {
				acc += v * v
			}
			if ref := acc / rate; want[p][i] != ref {
				t.Errorf("band %v period %d: %.17g, equalizer reference %.17g", b.Nominal, p, want[p][i], ref)
			}
		}
	}
}
