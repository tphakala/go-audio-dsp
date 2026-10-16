package octave

import (
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/tphakala/go-audio-dsp/equalizer"
)

// The relative attenuation template for one-third-octave band-pass filters in
// IEC 61260:1995 (identical national adoption IS 6964:2001), Table 1 and the
// informative Annex B Table B.1, base-10 series, classes 1 and 2. Only the
// relative attenuation (clause 4.4) is checked; the effective bandwidth,
// linearity and time invariance requirements, and the acceptance limits of
// IEC 61260-1:2014, are not.
//
// A limit row gives the minimum and maximum attenuation in dB at a normalized
// frequency Omega = f / fm (and at 1 / Omega). Between rows the limits
// interpolate linearly in dB against log frequency (eq. 12).
type limitRow struct{ lo, hi float64 }

// templateExponents are the octave-band Table 1 frequencies as powers of
// G = 10^0.3: G^0, G^(1/8), G^(1/4), G^(3/8), G^(1/2) (the limit just inside the
// band edge), G^(1/2) again (the band edge row), then G^1 ... G^4. A row's
// one-third-octave breakpoint follows from eq. (10).
var templateExponents = [10]float64{0, 1.0 / 8, 1.0 / 4, 3.0 / 8, 0.5, 0.5, 1, 2, 3, 4}

const (
	rowInsideEdge = 4 // the limits just inside the band edge
	rowEdge       = 5 // the limits exactly at the band edge
)

var templateRows = map[int][10]limitRow{
	1: {{-0.3, 0.3}, {-0.3, 0.4}, {-0.3, 0.6}, {-0.3, 1.3}, {-0.3, 5.0}, {2.0, 5.0}, {17.5, math.Inf(1)}, {42, math.Inf(1)}, {61, math.Inf(1)}, {70, math.Inf(1)}},
	2: {{-0.5, 0.5}, {-0.5, 0.6}, {-0.5, 0.8}, {-0.5, 1.6}, {-0.5, 5.5}, {1.6, 5.5}, {16.5, math.Inf(1)}, {41, math.Inf(1)}, {55, math.Inf(1)}, {60, math.Inf(1)}},
}

// templatePrinted are the breakpoints as printed in Table B.1 (five decimals).
var templatePrinted = [10]float64{1.00000, 1.02667, 1.05575, 1.08746, 1.12202, 1.12202, 1.29437, 1.88173, 3.05365, 5.39195}

// templateBreakpoints computes the one-third-octave breakpoints from eq. (10)
// with G = 10^0.3 and b = 3: Omega_h = 1 + (G^(1/6) - 1)/(G^(1/2) - 1) *
// (Omega_oct - 1). The band edge breakpoint is exactly 10^0.05, the ratio of
// Band.Upper to Band.Exact; the printed 1.12202 is not, which is why the
// breakpoints are computed and not copied.
func templateBreakpoints() [10]float64 {
	g := math.Pow(10, 0.3)
	k := (math.Pow(g, 1.0/6) - 1) / (math.Pow(g, 0.5) - 1)
	var bp [10]float64
	for i, e := range templateExponents {
		bp[i] = 1 + k*(math.Pow(g, e)-1)
	}
	bp[rowInsideEdge], bp[rowEdge] = math.Pow(10, 0.05), math.Pow(10, 0.05)
	return bp
}

var templateBP = templateBreakpoints()

// interpolate returns the value at omega of the piecewise log-linear curve
// through (bp[i], vals[i]), clamped to the last value beyond bp's end.
func interpolate(bp, vals []float64, omega float64) float64 {
	last := len(bp) - 1
	if omega >= bp[last] {
		return vals[last]
	}
	for i := range last {
		if omega >= bp[i] && omega <= bp[i+1] {
			t := math.Log(omega/bp[i]) / math.Log(bp[i+1]/bp[i])
			return vals[i] + (vals[i+1]-vals[i])*t
		}
	}
	return vals[0]
}

// templateLimits returns the minimum and maximum relative attenuation in dB
// that the class (1 or 2) allows at normalized frequency omega, or its
// reciprocal. Inside the band edge both limits interpolate over the rows up to
// the inside-edge row. Exactly at the edge the row is [+2.0, +5.0] (class 1) or
// [+1.6, +5.5] (class 2). Outside the edge the minimum interpolates over the
// rows from the edge row and the maximum is +Inf: interpolating a +Inf row
// would give NaN, and a NaN limit makes a margin check pass silently.
func templateLimits(class int, omega float64) (lo, hi float64) {
	if omega < 1 {
		omega = 1 / omega
	}
	rows := templateRows[class]
	edge := templateBP[rowEdge]
	switch {
	case math.Abs(omega/edge-1) < 1e-9:
		return rows[rowEdge].lo, rows[rowEdge].hi
	case omega < edge:
		var los, his [rowInsideEdge + 1]float64
		for i := range los {
			los[i], his[i] = rows[i].lo, rows[i].hi
		}
		bp := templateBP[:rowInsideEdge+1]
		return interpolate(bp, los[:], omega), interpolate(bp, his[:], omega)
	default:
		var los [len(rows) - rowEdge]float64
		for i := range los {
			los[i] = rows[rowEdge+i].lo
		}
		return interpolate(templateBP[rowEdge:], los[:], omega), math.Inf(1)
	}
}

// templateMargin is the distance in dB from att to the nearest limit, positive
// inside the template. A NaN anywhere is a failure, not a pass: math.Min with a
// NaN is NaN, and NaN < x is false.
func templateMargin(class int, omega, att float64) float64 {
	lo, hi := templateLimits(class, omega)
	m := att - lo
	if !math.IsInf(hi, 1) { // avoid Inf - Inf for a null that is infinitely deep
		m = math.Min(m, hi-att)
	}
	if math.IsNaN(m) {
		return math.Inf(-1)
	}
	return m
}

// attenuationGrid evaluates the relative attenuation of f (loss at f compared
// with the loss at exact, in dB, positive for a loss) on a logarithmic grid of
// Omega = f / exact with the given step in decades from 1e-3 to 1e3 clipped to
// (0.5 Hz, Nyquist), plus both sides of every breakpoint.
func attenuationGrid(f *equalizer.Filter, exact float64, rate int, step float64) (omegas, att []float64) {
	nyq := float64(rate) / 2
	add := func(om float64) {
		if fr := exact * om; fr > 0.5 && fr < nyq {
			omegas = append(omegas, om)
		}
	}
	for i := int(math.Round(-3 / step)); i <= int(math.Round(3/step)); i++ {
		add(math.Pow(10, float64(i)*step))
	}
	for _, bp := range templateBP {
		add(bp)
		add(1 / bp)
	}
	freqs := make([]float64, len(omegas))
	for i, om := range omegas {
		freqs[i] = exact * om
	}
	ref := f.Response([]float64{exact})[0].GainDB
	att = make([]float64, len(freqs))
	for i, p := range f.Response(freqs) {
		att[i] = ref - p.GainDB
	}
	return omegas, att
}

// worstMargin returns the smallest margin over the grid and the Omega at which
// it occurs.
func worstMargin(class int, omegas, att []float64) (worst, at float64) {
	worst = math.Inf(1)
	for i, om := range omegas {
		if m := templateMargin(class, om, att[i]); m < worst {
			worst, at = m, om
		}
	}
	return worst, at
}

// fineStep is the grid step in decades of the full template check; coarseStep
// is used where many sample rates are swept.
const (
	fineStep   = 0.0005
	coarseStep = 0.002
)

// checkMeterTemplate asserts class 1 and class 2 for every band of m. It also
// asserts that the response at 0 Hz is a null (-Inf) or at least 70 dB down, so
// the range from 0 Hz is covered and not only (0.5 Hz, Nyquist).
func checkMeterTemplate(t *testing.T, m *Meter, rate int, step float64) {
	t.Helper()
	for i, b := range m.Bands() {
		omegas, att := attenuationGrid(m.filters[i], b.Exact, rate, step)
		for class := 1; class <= 2; class++ {
			if worst, at := worstMargin(class, omegas, att); !(worst >= 0) {
				t.Errorf("%d Hz band %v (upper edge %.4f of Nyquist): class %d margin %.3f dB at Omega %.5g, want >= 0",
					rate, b.Nominal, b.Upper/(float64(rate)/2), class, worst, at)
			}
		}
		dc := m.filters[i].Response([]float64{0, b.Exact})
		if !math.IsInf(dc[0].GainDB, -1) && dc[1].GainDB-dc[0].GainDB < 70 {
			t.Errorf("%d Hz band %v: response at 0 Hz is %.2f dB, only %.2f dB below the centre, want a null or >= 70 dB", rate, b.Nominal, dc[0].GainDB, dc[1].GainDB-dc[0].GainDB)
		}
	}
}

func TestTemplateLimits(t *testing.T) {
	// The computed breakpoints agree with every printed Table B.1 value.
	for i, p := range templatePrinted {
		if d := math.Abs(templateBP[i] - p); d > 1e-5 {
			t.Errorf("breakpoint %d: eq. (10) gives %.7f, Table B.1 prints %.5f", i, templateBP[i], p)
		}
	}

	near := func(got, want, tol float64) bool { return math.Abs(got-want) <= tol }
	type point struct {
		class          int
		omega          float64
		wantLo, wantHi float64
		tol            float64
	}
	inf := math.Inf(1)
	edge := math.Pow(10, 0.05)
	points := []point{
		{1, 1, -0.3, 0.3, 1e-12},
		{2, 1, -0.5, 0.5, 1e-12},
		{1, 1.29437, 17.5, inf, 0.01}, // the printed G^1 row, past the edge: the maximum is +Inf
		{1, 0.32748, 61, inf, 0.01},   // the printed G^3 row, by its reciprocal
		{1, 3.05365, 61, inf, 0.01},
		{2, 3.05365, 55, inf, 0.01},
		{1, 100, 70, inf, 1e-12},
		{2, 0.001, 60, inf, 1e-12},
		{1, edge, 2.0, 5.0, 1e-12},
		{2, edge, 1.6, 5.5, 1e-12},
		{1, 1 / edge, 2.0, 5.0, 1e-12},
		{1, edge * (1 - 1e-6), -0.3, 5.0, 1e-3},                        // just inside the edge
		{1, edge * (1 + 1e-6), 2.0, inf, 1e-3},                         // just outside the edge
		{2, edge * (1 - 1e-6), -0.5, 5.5, 1e-3},                        // just inside the edge
		{2, edge * (1 + 1e-6), 1.6, inf, 1e-3},                         // just outside the edge
		{1, math.Sqrt(templateBP[1] * templateBP[2]), -0.3, 0.5, 1e-9}, // log midpoint of 0.4 and 0.6
	}
	for _, p := range points {
		lo, hi := templateLimits(p.class, p.omega)
		if math.IsNaN(lo) || math.IsNaN(hi) {
			t.Errorf("class %d omega %.6g: limits (%v, %v) contain NaN", p.class, p.omega, lo, hi)
			continue
		}
		okHi := near(hi, p.wantHi, p.tol) || (math.IsInf(p.wantHi, 1) && math.IsInf(hi, 1))
		if !near(lo, p.wantLo, p.tol) || !okHi {
			t.Errorf("class %d omega %.6g: limits (%g, %g), want (%g, %g)", p.class, p.omega, lo, hi, p.wantLo, p.wantHi)
		}
	}
	// The limits are symmetric in Omega and 1/Omega.
	for _, om := range []float64{0.05, 0.3, 0.7, 0.9, 0.95} {
		loA, hiA := templateLimits(1, om)
		loB, hiB := templateLimits(1, 1/om)
		if !near(loA, loB, 1e-9) || (!near(hiA, hiB, 1e-9) && !(math.IsInf(hiA, 1) && math.IsInf(hiB, 1))) {
			t.Errorf("omega %g: limits (%g, %g) differ from those at 1/omega (%g, %g)", om, loA, hiA, loB, hiB)
		}
	}
}

// TestTemplateMarginTreatsNaNAsFailure pins that a NaN attenuation (or limit) is
// a failure, so a broken response cannot pass the template check silently.
func TestTemplateMarginTreatsNaNAsFailure(t *testing.T) {
	for _, om := range []float64{1, 1.05, 1.2, 4} {
		if m := templateMargin(1, om, math.NaN()); !math.IsInf(m, -1) {
			t.Errorf("omega %g: margin of a NaN attenuation = %v, want -Inf", om, m)
		}
	}
	// An infinitely deep null in the stop band passes.
	if m := templateMargin(1, 4, math.Inf(1)); !(m > 0) {
		t.Errorf("margin of an infinite attenuation in the stop band = %v, want > 0", m)
	}
}

// TestButterworthTemplate asserts the relative attenuation template for every
// band the Butterworth profile includes: at the default range for five rates,
// for every rate from 44.1 to 48 kHz in 50 Hz steps (the profile's reason to
// exist at BirdNET-Go's rates, plus 47131 Hz, where the 20 kHz band's upper edge
// sits just under the 0.95 margin), and for ultrasonic sets whose range reaches
// Nyquist.
func TestButterworthTemplate(t *testing.T) {
	for _, rate := range testRates {
		t.Run("default/"+strconv.Itoa(rate), func(t *testing.T) {
			t.Parallel()
			checkMeterTemplate(t, mustMeter(t, Config{SampleRate: rate, PeriodSamples: 1, Filter: FilterButterworth}), rate, fineStep)
		})
	}
	t.Run("rate sweep", func(t *testing.T) {
		t.Parallel()
		rates := make([]int, 0, 80)
		rates = append(rates, 47131)
		for rate := 44100; rate <= 48000; rate += 50 {
			rates = append(rates, rate)
		}
		for _, rate := range rates {
			checkMeterTemplate(t, mustMeter(t, Config{SampleRate: rate, PeriodSamples: 1, Filter: FilterButterworth}), rate, coarseStep)
		}
	})
	for _, rate := range []int{96000, 192000, 256000, 384000} {
		t.Run("ultrasonic/"+strconv.Itoa(rate), func(t *testing.T) {
			t.Parallel()
			m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: 1, MaxFreq: float64(rate) / 2, Filter: FilterButterworth})
			checkMeterTemplate(t, m, rate, fineStep)
		})
	}
}

// TestTemplateCatchesNonCompliantDesigns shows the template check can fail: each
// design below is outside class 1, and the check says so by the margin recorded
// with it.
func TestTemplateCatchesNonCompliantDesigns(t *testing.T) {
	margin := func(t *testing.T, f *equalizer.Filter, exact float64, rate int) float64 {
		t.Helper()
		omegas, att := attenuationGrid(f, exact, rate, fineStep)
		worst, at := worstMargin(1, omegas, att)
		t.Logf("class 1 margin %.3f dB at Omega %.5g", worst, at)
		return worst
	}
	band := func(centre float64) Band { return bandAt(int(math.Round(10 * math.Log10(centre/1000)))) }

	t.Run("default biquad design", func(t *testing.T) {
		m := mustMeter(t, Config{SampleRate: 48000, PeriodSamples: 1})
		i := slices.IndexFunc(m.Bands(), func(b Band) bool { return b.Nominal == 1000 })
		if got := margin(t, m.filters[i], m.Bands()[i].Exact, 48000); got > -30 {
			t.Errorf("1 kHz band of the default design: class 1 margin %.2f dB, want below -30 (measured -42.9)", got)
		}
	})
	t.Run("prototype order 3 at the 20 kHz band", func(t *testing.T) {
		b := band(19952.623149688792)
		f, err := equalizer.NewButterworthBandPass(b.Lower, b.Upper, 3, 48000)
		if err != nil {
			t.Fatal(err)
		}
		if got := margin(t, f, b.Exact, 48000); got > -5 {
			t.Errorf("order 3 at 48 kHz: class 1 margin %.2f dB, want below -5 (measured -11.5)", got)
		}
	})
	for _, k := range []float64{0.9, 1.1} {
		t.Run("misplaced edges "+strconv.FormatFloat(k, 'g', -1, 64), func(t *testing.T) {
			b := band(1000)
			// Edges at 10^(+-0.05*k) around the centre instead of 10^(+-0.05).
			f, err := equalizer.NewButterworthBandPass(b.Exact*math.Pow(10, -0.05*k), b.Exact*math.Pow(10, 0.05*k), butterworthOrder, 48000)
			if err != nil {
				t.Fatal(err)
			}
			if got := margin(t, f, b.Exact, 48000); got > -0.1 {
				t.Errorf("edges scaled by %g: class 1 margin %.3f dB, want below -0.1 (measured -0.23 and -0.36)", k, got)
			}
		})
	}
	t.Run("upper edge at 0.975 of Nyquist", func(t *testing.T) {
		const rate = 48000
		upper := 0.975 * rate / 2
		exact := upper / math.Pow(10, 0.05)
		f, err := equalizer.NewButterworthBandPass(exact*math.Pow(10, -0.05), upper, butterworthOrder, rate)
		if err != nil {
			t.Fatal(err)
		}
		if got := margin(t, f, exact, rate); got > -0.05 {
			t.Errorf("upper edge at 0.975 of Nyquist: class 1 margin %.3f dB, want below -0.05 (measured -0.24); this is why butterworthMaxMargin is 0.95", got)
		}
	})
	// The compliant profile passes the same check, so the failures above are the
	// designs' and not the helper's.
	t.Run("the Butterworth profile passes", func(t *testing.T) {
		m := mustMeter(t, Config{SampleRate: 48000, PeriodSamples: 1, Filter: FilterButterworth})
		i := slices.IndexFunc(m.Bands(), func(b Band) bool { return b.Nominal == 20000 })
		if got := margin(t, m.filters[i], m.Bands()[i].Exact, 48000); !(got > 0) {
			t.Errorf("20 kHz band of the Butterworth profile: class 1 margin %.3f dB, want > 0", got)
		}
	})
}
