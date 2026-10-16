package octave

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/equalizer"
)

var testRates = []int{44100, 48000, 96000, 192000, 256000}

// designs lists the filter profiles that the meter-level contract tests run
// over.
var designs = []FilterDesign{FilterBiquad, FilterButterworth}

// isoNominal is the ISO 266 1/3-octave series from 25 Hz to 20 kHz.
var isoNominal = []float64{
	25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
	1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
}

func mustMeter(t *testing.T, cfg Config) *Meter {
	t.Helper()
	m, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func sine(freq float64, rate, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / float64(rate)))
	}
	return s
}

// periods runs in through m and returns every emitted period as a copy.
func periods(m *Meter, in []float32) [][]float64 {
	var out [][]float64
	m.Process(in, func(ms []float64) { out = append(out, slices.Clone(ms)) })
	return out
}

func db(ms float64) float64 { return 10 * math.Log10(ms) }

func TestBandTable(t *testing.T) {
	m := mustMeter(t, Config{SampleRate: 96000, PeriodSamples: 96000})
	bands := m.Bands()
	if len(bands) != len(isoNominal) || m.NumBands() != len(isoNominal) {
		t.Fatalf("got %d bands, want %d", len(bands), len(isoNominal))
	}
	for i, b := range bands {
		if b.Nominal != isoNominal[i] {
			t.Errorf("band %d nominal = %v, want %v", i, b.Nominal, isoNominal[i])
		}
		if want := 1000 * math.Pow(10, float64(i-16)/10); math.Abs(b.Exact-want) > 1e-9*want {
			t.Errorf("band %d exact = %v, want %v", i, b.Exact, want)
		}
		if r := b.Upper / b.Lower; math.Abs(r-math.Pow(10, 0.1)) > 1e-12 {
			t.Errorf("band %d edge ratio = %v, want 10^0.1", i, r)
		}
		if math.Abs(b.Lower*b.Upper-b.Exact*b.Exact) > 1e-9*b.Exact*b.Exact {
			t.Errorf("band %d is not geometrically centred", i)
		}
	}
}

func TestNyquistCutPerRate(t *testing.T) {
	// The top band is the highest whose upper edge stays under 0.95 * Nyquist.
	wantTop := map[int]float64{44100: 16000, 48000: 20000, 96000: 20000, 192000: 20000, 256000: 20000}
	for _, rate := range testRates {
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate})
		bands := m.Bands()
		top := bands[len(bands)-1]
		if top.Nominal != wantTop[rate] {
			t.Errorf("%d Hz: top band %v, want %v", rate, top.Nominal, wantTop[rate])
		}
		for _, b := range bands {
			if !(b.Upper < 0.95*float64(rate)/2) {
				t.Errorf("%d Hz: band %v upper edge %v is not below 0.95 Nyquist", rate, b.Nominal, b.Upper)
			}
		}
		if bands[0].Nominal != 25 || !slices.IsSortedFunc(bands, func(a, b Band) int {
			switch {
			case a.Nominal < b.Nominal:
				return -1
			case a.Nominal > b.Nominal:
				return 1
			}
			return 0
		}) {
			t.Errorf("%d Hz: bands not ascending from 25 Hz", rate)
		}
	}
	if got := mustMeter(t, Config{SampleRate: 44100, PeriodSamples: 1}).NumBands(); got != 29 {
		t.Errorf("44.1 kHz band count = %d, want 29", got)
	}
	// The margin decides a band whose upper edge falls between 0.95 and 1.0 of
	// Nyquist: at 47 kHz the 20 kHz band's edge is at 22449 Hz, Nyquist 23500.
	if top := mustMeter(t, Config{SampleRate: 47000, PeriodSamples: 1}).Bands(); top[len(top)-1].Nominal != 16000 {
		t.Errorf("47 kHz default margin top band = %v, want 16000", top[len(top)-1].Nominal)
	}
	if top := mustMeter(t, Config{SampleRate: 47000, PeriodSamples: 1, NyquistMargin: 1}).Bands(); top[len(top)-1].Nominal != 20000 {
		t.Errorf("47 kHz margin 1 top band = %v, want 20000", top[len(top)-1].Nominal)
	}
	// A wider range and a looser margin select more bands at a high rate.
	ultra := mustMeter(t, Config{SampleRate: 256000, PeriodSamples: 1, MaxFreq: 100000, NyquistMargin: 1})
	if top := ultra.Bands()[ultra.NumBands()-1].Nominal; top != 100000 {
		t.Errorf("ultrasonic top band = %v, want 100000", top)
	}
}

func TestNewErrors(t *testing.T) {
	nan := math.NaN()
	cases := []struct {
		name string
		cfg  Config
	}{
		{"zero rate", Config{PeriodSamples: 100}},
		{"negative rate", Config{SampleRate: -1, PeriodSamples: 100}},
		{"zero period", Config{SampleRate: 48000}},
		{"NaN min", Config{SampleRate: 48000, PeriodSamples: 1, MinFreq: nan}},
		{"negative min", Config{SampleRate: 48000, PeriodSamples: 1, MinFreq: -5}},
		{"Inf max", Config{SampleRate: 48000, PeriodSamples: 1, MaxFreq: math.Inf(1)}},
		{"max below min", Config{SampleRate: 48000, PeriodSamples: 1, MinFreq: 1000, MaxFreq: 500}},
		{"NaN margin", Config{SampleRate: 48000, PeriodSamples: 1, NyquistMargin: nan}},
		{"margin above 1", Config{SampleRate: 48000, PeriodSamples: 1, NyquistMargin: 1.5}},
		{"negative margin", Config{SampleRate: 48000, PeriodSamples: 1, NyquistMargin: -0.1}},
		{"order too high", Config{SampleRate: 48000, PeriodSamples: 1, Order: 9}},
		{"negative order", Config{SampleRate: 48000, PeriodSamples: 1, Order: -1}},
		{"unknown design", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterDesign(2)}},
		{"negative design", Config{SampleRate: 48000, PeriodSamples: 1, Filter: -1}},
		{"Butterworth with Order 1", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, Order: 1}},
		{"Butterworth with Order 4", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, Order: 4}},
		{"Butterworth with negative Order", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, Order: -1}},
		{"Butterworth margin 0.96", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, NyquistMargin: 0.96}},
		{"Butterworth margin 1", Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth, NyquistMargin: 1}},
		{"no band fits", Config{SampleRate: 40, PeriodSamples: 1}},
		{"range between bands", Config{SampleRate: 48000, PeriodSamples: 1, MinFreq: 26, MaxFreq: 30}},
	}
	for _, c := range cases {
		m, err := New(c.cfg)
		if !errors.Is(err, dsp.ErrInvalidConfig) || !errors.Is(err, ErrInvalidConfig) || m != nil {
			t.Errorf("%s: got (%v, %v), want an error wrapping ErrInvalidConfig", c.name, m, err)
		}
	}
}

// TestSineAtEachCentre pins the reference level: a full scale sine at a band's
// exact centre reads -3.01 dB in that band (the peak gain is 0 dB) at every rate,
// in both filter profiles.
func TestSineAtEachCentre(t *testing.T) {
	for _, design := range designs {
		for _, rate := range []int{44100, 48000, 96000, 256000} {
			for _, b := range mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: design}).Bands() {
				m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, MinFreq: b.Nominal, MaxFreq: b.Nominal, Filter: design})
				ps := periods(m, sine(b.Exact, rate, 3*rate))
				if len(ps) != 3 {
					t.Fatalf("%v %d Hz: %d periods, want 3", design, rate, len(ps))
				}
				i := 0
				// A band below 100 Hz holds a fractional number of cycles of its centre
				// sine in a one second period, which shifts the mean square slightly.
				if got := db(ps[2][i]); math.Abs(got-(-3.0103)) > 0.01+0.7/b.Exact {
					t.Errorf("%v %d Hz band %v: centre sine reads %.4f dB, want -3.0103", design, rate, b.Nominal, got)
				}
			}
		}
	}
}

// levelAt returns the level in dB that a unit sine at f reads in the given band
// after the start-up transient (the third of three one second periods).
func levelAt(t *testing.T, rate int, nominal, f float64, order int) float64 {
	t.Helper()
	m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, MinFreq: nominal, MaxFreq: nominal, Order: order})
	ps := periods(m, sine(f, rate, 3*rate))
	return db(ps[2][0])
}

// edgeNearFraction is the fraction of Nyquist above which a band's upper edge
// counts as near Nyquist. The bandwidth prewarp keeps the edges within 0.15 dB
// below it, and degrades gracefully above it, where the bilinear transform warps
// the response.
const edgeNearFraction = 0.3

// TestEdgeResponse pins the bandwidth design: a sine at a band's lower or upper
// edge reads 3.01 dB below the centre level (6.02 dB total) with Order 1 at
// every rate, within 0.15 dB while the band's upper edge is under 0.3 of
// Nyquist. Nearer Nyquist, where the bilinear transform warps the response, the
// error must stay under 40% of the error of a design without the bandwidth
// prewarp, which misses the edge by several dB there. Order n reads n times the
// edge attenuation in the well-conditioned bands.
func TestEdgeResponse(t *testing.T) {
	for _, rate := range []int{44100, 48000, 96000, 256000} {
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate})
		for _, b := range m.Bands() {
			near := b.Upper/(float64(rate)/2) > edgeNearFraction
			for _, f := range []float64{b.Lower, b.Upper} {
				err := math.Abs(levelAt(t, rate, b.Nominal, f, 1) - (-6.0206))
				if !near && err > 0.15 {
					t.Errorf("%d Hz band %v: edge %.1f Hz is %.3f dB off -6.02", rate, b.Nominal, f, err)
				}
				if near {
					un := math.Abs(unprewarpedEdgeDB(b.Exact, b.Lower, b.Upper, f, rate) - 3.0103 - (-6.0206))
					if err > 0.4*un {
						t.Errorf("%d Hz band %v: edge %.1f Hz is %.3f dB off -6.02, un-prewarped design %.3f dB off", rate, b.Nominal, f, err, un)
					}
				}
			}
		}
	}
	for _, order := range []int{2, 3} {
		const rate = 48000
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Order: order})
		for _, i := range []int{3, 16} {
			b := m.Bands()[i]
			want := -3.0103 - float64(order)*3.0103
			if got := levelAt(t, rate, b.Nominal, b.Upper, order); math.Abs(got-want) > 0.3 {
				t.Errorf("order %d band %v: upper edge reads %.3f dB, want %.2f", order, b.Nominal, got, want)
			}
			if got := levelAt(t, rate, b.Nominal, b.Exact, order); math.Abs(got-(-3.0103)) > 0.01 {
				t.Errorf("order %d band %v: centre reads %.4f dB, want -3.0103", order, b.Nominal, got)
			}
		}
	}
}

// unprewarpedEdgeDB is the edge response of the design this package replaces: an
// RBJ band-pass with alpha = sin(w0)/(2Q) and Q = fc/(hi-lo), no bandwidth
// prewarp. It is evaluated analytically so the test needs no second meter.
func unprewarpedEdgeDB(fc, lo, hi, f float64, rate int) float64 {
	fs := float64(rate)
	w0 := 2 * math.Pi * fc / fs
	alpha := math.Sin(w0) / (2 * (fc / (hi - lo)))
	a0 := 1 + alpha
	b0, b2 := alpha/a0, -alpha/a0
	a1, a2 := -2*math.Cos(w0)/a0, (1-alpha)/a0
	w := 2 * math.Pi * f / fs
	num := complex(b0, 0) + complex(b2*math.Cos(2*w), -b2*math.Sin(2*w))
	den := complex(1+a1*math.Cos(w)+a2*math.Cos(2*w), -a1*math.Sin(w)-a2*math.Sin(2*w))
	mag := math.Hypot(real(num), imag(num)) / math.Hypot(real(den), imag(den))
	return 20 * math.Log10(mag)
}

// TestEdgeResponseCatchesUnprewarpedDesign shows the edge test can fail: the
// un-prewarped design misses the edge level by more than 3 dB in the top band
// at 48 kHz, far outside what TestEdgeResponse allows.
func TestEdgeResponseCatchesUnprewarpedDesign(t *testing.T) {
	m := mustMeter(t, Config{SampleRate: 48000, PeriodSamples: 48000})
	b := m.Bands()[m.NumBands()-1]
	got := unprewarpedEdgeDB(b.Exact, b.Lower, b.Upper, b.Upper, 48000) - 3.0103 // at the upper edge, relative to full scale
	if math.Abs(got-(-6.0206)) < 3 {
		t.Errorf("un-prewarped design reads %.2f dB at the top band's upper edge; the edge test could not catch it", got)
	}
}

// refBandPassMS is an independent float64 reference for one band: the RBJ
// band-pass written out directly from the cookbook with the octave-bandwidth
// alpha, run sample by sample over the whole input, returning the mean square
// per period.
func refBandPassMS(in []float32, rate int, b Band, order, period int) []float64 {
	fs := float64(rate)
	w0 := 2 * math.Pi * b.Exact / fs
	bw := math.Log2(b.Upper / b.Lower)
	alpha := math.Sin(w0) * math.Sinh(math.Ln2/2*bw*w0/math.Sin(w0))
	a0 := 1 + alpha
	b0, b2 := alpha/a0, -alpha/a0
	a1, a2 := -2*math.Cos(w0)/a0, (1-alpha)/a0
	type state struct{ x1, x2, y1, y2 float64 }
	st := make([]state, order)
	var out []float64
	var acc float64
	n := 0
	for _, s := range in {
		v := float64(s)
		for k := range st {
			y := b0*v + b2*st[k].x2 - a1*st[k].y1 - a2*st[k].y2
			st[k].x2, st[k].x1 = st[k].x1, v
			st[k].y2, st[k].y1 = st[k].y1, y
			v = y
		}
		acc += v * v
		n++
		if n == period {
			out = append(out, acc/float64(period))
			acc, n = 0, 0
		}
	}
	return out
}

func noise(n int, seed uint64, amp float64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed*7+1))
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(amp * (2*r.Float64() - 1))
	}
	return s
}

// TestMatchesFloat64Reference cross-checks every band against a direct float64
// implementation of the same design, for Order 1 and 3.
func TestMatchesFloat64Reference(t *testing.T) {
	for _, rate := range []int{48000, 192000} {
		for _, order := range []int{1, 3} {
			in := noise(2*rate+777, 5, 0.5)
			m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Order: order})
			ps := periods(m, in)
			for i, b := range m.Bands() {
				ref := refBandPassMS(in, rate, b, order, rate)
				if len(ref) != len(ps) {
					t.Fatalf("period count %d, reference %d", len(ps), len(ref))
				}
				for p := range ref {
					if rel := math.Abs(ps[p][i]-ref[p]) / ref[p]; rel > 1e-9 {
						t.Errorf("%d Hz order %d band %v period %d: %.12g, reference %.12g (rel %.2g)", rate, order, b.Nominal, p, ps[p][i], ref[p], rel)
					}
				}
			}
		}
	}
}

// analyticNoiseMS is the mean square the band reads for white noise of the given
// variance, from the numerically integrated power response of the same design:
// variance * (2/fs) * integral over [0, fs/2] of |H(f)|^2 df.
func analyticNoiseMS(t *testing.T, rate int, b Band, order int, variance float64) float64 {
	t.Helper()
	r := math.Pow(10, 0.1)
	eq, err := equalizer.New(equalizer.Config{SampleRate: rate, Bands: []equalizer.Band{{
		Type: equalizer.BandPass, Frequency: b.Exact, WidthHz: 2 * b.Exact * (r - 1) / (r + 1), Passes: order,
	}}})
	if err != nil {
		t.Fatal(err)
	}
	nyq := float64(rate) / 2
	const points = 200000
	freqs := make([]float64, points+1)
	for i := range freqs {
		freqs[i] = nyq * float64(i) / points
	}
	resp := eq.Response(freqs)
	var sum float64
	for i := 1; i < len(resp); i++ {
		g0 := math.Pow(10, resp[i-1].GainDB/10)
		g1 := math.Pow(10, resp[i].GainDB/10)
		sum += (g0 + g1) / 2 * (freqs[i] - freqs[i-1])
	}
	return variance * sum / nyq
}

// TestNoiseLevelMatchesNoiseBandwidth checks the white noise level of the upper
// bands against the numerically integrated noise bandwidth of the design, at
// two rates, for the biquad profile at Orders 1 and 3 and for the Butterworth
// profile. The bands used are wide enough that 30 s of noise pins the level to
// about 0.1 dB.
func TestNoiseLevelMatchesNoiseBandwidth(t *testing.T) {
	const amp = 0.5
	variance := amp * amp / 3 // uniform on [-amp, amp]
	cases := []struct {
		name string
		cfg  Config
	}{
		{"biquad order 1", Config{Order: 1}},
		{"biquad order 3", Config{Order: 3}},
		{"butterworth", Config{Filter: FilterButterworth}},
	}
	for _, rate := range []int{48000, 96000} {
		for ci, c := range cases {
			cfg := c.cfg
			cfg.SampleRate, cfg.PeriodSamples = rate, rate
			m := mustMeter(t, cfg)
			ps := periods(m, noise(30*rate, uint64(rate+ci), amp))
			for i, b := range m.Bands() {
				if b.Nominal < 1000 {
					continue
				}
				var mean float64
				for _, p := range ps[1:] {
					mean += p[i]
				}
				mean /= float64(len(ps) - 1)
				// The biquad profiles are checked against an independent equalizer
				// built from the documented design; the Butterworth profile against
				// the integrated response of the meter's own filter.
				var want float64
				if cfg.Filter == FilterButterworth {
					want = filterNoiseMS(m.filters[i], rate, variance)
				} else {
					want = analyticNoiseMS(t, rate, b, cfg.Order, variance)
				}
				if d := db(mean) - db(want); math.Abs(d) > 0.2 {
					t.Errorf("%d Hz %s band %v: noise reads %.2f dB, analytic %.2f dB (%.2f dB off)", rate, c.name, b.Nominal, db(mean), db(want), d)
				}
			}
		}
	}
}

// TestNoiseBiasOfSingleBiquad documents the level bias of Order 1: white noise
// reads about 10*log10(pi/2) = 1.96 dB above the rectangular ideal
// variance*(Upper-Lower)/(fs/2) in the bands well below Nyquist, where the
// analog approximation holds.
func TestNoiseBiasOfSingleBiquad(t *testing.T) {
	const rate = 96000
	m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate})
	for _, b := range m.Bands() {
		if b.Nominal < 100 || b.Nominal > 2000 {
			continue
		}
		got := db(analyticNoiseMS(t, rate, b, 1, 1) / ((b.Upper - b.Lower) / (rate / 2.0)))
		if math.Abs(got-1.96) > 0.1 {
			t.Errorf("band %v: noise bias %.2f dB, want about 1.96", b.Nominal, got)
		}
	}
}

func TestChunkInvariance(t *testing.T) {
	const rate = 48000
	in := noise(3*rate+4321, 11, 0.4)
	for _, design := range designs {
		for _, period := range []int{rate, 1000, 4096, 5} {
			ref := mustMeter(t, Config{SampleRate: rate, PeriodSamples: period, Filter: design})
			want := periods(ref, in)
			r := rand.New(rand.NewPCG(3, 4))
			for trial := range 4 {
				m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: period, Filter: design})
				var got [][]float64
				for rest := in; len(rest) > 0; {
					n := min(1+r.IntN(2*chunk+500), len(rest))
					if trial == 0 {
						n = min(1, len(rest)) // single samples
						if len(got) > 3 {
							n = len(rest) // then everything else at once
						}
					}
					got = append(got, periods(m, rest[:n])...)
					rest = rest[n:]
				}
				if len(got) != len(want) {
					t.Fatalf("%v period %d trial %d: %d periods, want %d", design, period, trial, len(got), len(want))
				}
				for p := range want {
					if !slices.Equal(got[p], want[p]) {
						t.Fatalf("%v period %d trial %d: period %d differs from the one-call result", design, period, trial, p)
					}
				}
			}
		}
	}
}

func TestResetAndEmpty(t *testing.T) {
	const rate = 48000
	in := noise(rate+100, 21, 0.5)
	for _, design := range designs {
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: design})
		want := periods(m, in)
		m.Process(nil, func([]float64) { t.Error("emit called for empty input") })
		m.Process([]float32{}, nil)
		m.Reset()
		got := periods(m, in)
		if len(got) != len(want) || !slices.Equal(got[0], want[0]) {
			t.Errorf("%v: a Reset meter does not repeat the first period of a fresh one", design)
		}
		// A partial period is dropped by Reset.
		m.Process(in[:rate/2], nil)
		m.Reset()
		if got := periods(m, in); len(got) != len(want) || !slices.Equal(got[0], want[0]) {
			t.Errorf("%v: Reset kept part of a period", design)
		}
	}
}

func TestNonFiniteInputDoesNotPoison(t *testing.T) {
	const rate = 48000
	in := sine(1000, rate, 3*rate)
	in[1000] = float32(math.NaN())
	in[2000] = float32(math.Inf(1))
	for _, design := range designs {
		m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate, Filter: design})
		ps := periods(m, in)
		for p := range ps {
			for i, v := range ps[p] {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("%v: period %d band %d is %v", design, p, i, v)
				}
			}
		}
		if got := db(ps[2][16]); math.Abs(got-(-3.0103)) > 0.05 {
			t.Errorf("%v: 1 kHz band reads %.3f dB after the bad samples, want -3.01", design, got)
		}
	}
}

func TestDB(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{1, 0}, {0.5, -3.0103}, {0.1, -10}, {1e-10, -100},
		{0, -200}, {-1, -200}, {math.NaN(), -200}, {1e-30, -200},
		{math.Inf(1), 200}, {1e30, 200},
	}
	for _, c := range cases {
		if got := DB(c.in); math.Abs(got-c.want) > 1e-3 || math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("DB(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestNewRejectsUnstableBandPass covers the filter-design failure path: at a
// sample rate so high that a 25 Hz band collapses toward DC the coefficients are
// no longer stable, and New reports it as an invalid config instead of
// returning a meter that cannot be trusted, in both filter profiles.
func TestNewRejectsUnstableBandPass(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("the rate needs a 64-bit int")
	}
	rate := int(math.Pow10(14))
	for _, design := range designs {
		if m, err := New(Config{SampleRate: rate, PeriodSamples: 1, Filter: design}); !errors.Is(err, ErrInvalidConfig) || m != nil {
			t.Errorf("%v: got (%v, %v), want an error wrapping ErrInvalidConfig", design, m, err)
		}
	}
}
