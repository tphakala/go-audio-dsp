package equalizer

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

// allTypesConfig is a 16-band, 3-pass-per-band config covering every FilterType.
func allTypesConfig() Config {
	cfg := Config{SampleRate: 48000}
	types := []FilterType{LowPass, HighPass, AllPass, BandPass, BandReject, LowShelf, HighShelf, Peaking}
	for i := range 16 {
		b := Band{Type: types[i%len(types)], Frequency: 300 + 900*float64(i), Q: 0.8, WidthHz: 200, GainDB: float64(i%5) - 2, Passes: 3}
		cfg.Bands = append(cfg.Bands, b)
	}
	return cfg
}

func responseFreqs(t *testing.T) []float64 {
	t.Helper()
	freqs, err := LogSweep(1, 23999, 257)
	if err != nil {
		t.Fatal(err)
	}
	return append(freqs, 0)
}

// TestFilterChainResponseMatchesEqualizer pins that a chain of one Filter per
// band reports exactly the response of an Equalizer of the same bands.
func TestFilterChainResponseMatchesEqualizer(t *testing.T) {
	freqs := responseFreqs(t)
	for name, cfg := range map[string]Config{"testConfig": testConfig(), "allTypes": allTypesConfig()} {
		t.Run(name, func(t *testing.T) {
			e, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			got, want := chainOf(t, cfg).Response(freqs), e.Response(freqs)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("freq %g Hz: chain %+v, equalizer %+v", freqs[i], got[i], want[i])
				}
			}
		})
	}
}

// TestFilterChainResponseTracksAddFilter pins that Response follows each
// successful AddFilter and ignores a rejected one.
func TestFilterChainResponseTracksAddFilter(t *testing.T) {
	freqs := responseFreqs(t)
	cfg := testConfig()
	c := NewFilterChain()
	for n, b := range cfg.Bands {
		if err := c.AddFilter(mustFilter(t, b, cfg.SampleRate)); err != nil {
			t.Fatal(err)
		}
		e, err := New(Config{SampleRate: cfg.SampleRate, Bands: cfg.Bands[:n+1]})
		if err != nil {
			t.Fatal(err)
		}
		if got, want := c.Response(freqs), e.Response(freqs); !slices.Equal(got, want) {
			t.Fatalf("after %d filters the chain response differs from the equalizer's", n+1)
		}
	}
	before := c.Response(freqs)
	bad := mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6}, 44100)
	for name, f := range map[string]*Filter{"nil": nil, "rate mismatch": bad} {
		if err := c.AddFilter(f); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("%s: AddFilter err = %v, want ErrInvalidConfig", name, err)
		}
		if c.Len() != 3 || c.SampleRate() != 48000 || !slices.Equal(c.Response(freqs), before) {
			t.Fatalf("%s: a rejected AddFilter changed the chain", name)
		}
	}
}

// mixedChain builds the heterogeneous chain used by the sum and measured tests.
// The Butterworth band-pass is optional because its stopband runs deeper than
// the float32 audio path can measure.
func mixedChain(t *testing.T, withButterworth bool) *FilterChain {
	t.Helper()
	const fs = 48000
	c := NewFilterChain()
	add := func(f *Filter) {
		t.Helper()
		if err := c.AddFilter(f); err != nil {
			t.Fatal(err)
		}
	}
	add(mustFilter(t, Band{Type: HighPass, Frequency: 10, Q: 0.7071, Passes: 2}, fs))
	add(mustFilter(t, Band{Type: LowShelf, Frequency: 200, Q: 0.7071, GainDB: 4}, fs))
	add(mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: -6}, fs))
	add(mustFilter(t, Band{Type: HighShelf, Frequency: 8000, Q: 0.7071, GainDB: -3}, fs))
	ap := mustFilter(t, Band{Type: AllPass, Frequency: 3000, Q: 1}, fs)
	bf, err := NewBiquadFilter(biquadsOf(ap), fs)
	if err != nil {
		t.Fatal(err)
	}
	add(bf)
	if withButterworth {
		lower, upper := thirdOctave(1000)
		bw, err := NewButterworthBandPass(lower, upper, 4, fs)
		if err != nil {
			t.Fatal(err)
		}
		add(bw)
	}
	return c
}

// TestFilterChainResponseIsSumOfFilters cross-checks the chain against the
// per-Filter responses: dB and phase add.
func TestFilterChainResponseIsSumOfFilters(t *testing.T) {
	c := mixedChain(t, true)
	freqs, err := LogSweep(5, 20000, 200)
	if err != nil {
		t.Fatal(err)
	}
	got := c.Response(freqs)
	for i, f := range freqs {
		var db, ph float64
		for _, flt := range c.filters {
			p := flt.Response([]float64{f})[0]
			db += p.GainDB
			ph += p.PhaseRad
		}
		if math.Abs(got[i].GainDB-db) > 1e-9 {
			t.Fatalf("%g Hz: chain %g dB, sum of filters %g dB", f, got[i].GainDB, db)
		}
		if d := math.Remainder(got[i].PhaseRad-ph, 2*math.Pi); math.Abs(d) > 1e-9 {
			t.Fatalf("%g Hz: chain %g rad, sum of filters %g rad", f, got[i].PhaseRad, ph)
		}
	}
}

// singleBin returns the DFT of x at w rad/sample, so the same arithmetic serves
// input and output.
func singleBin[T float32 | float64](x []T, w float64) complex128 {
	var acc complex128
	for n, v := range x {
		acc += complex(float64(v), 0) * cmplx.Rect(1, -w*float64(n))
	}
	return acc
}

// TestFilterChainResponseMeasured compares Response with the gain and phase
// realized by ProcessInto (float32) and ApplyFloat64 on a sine at each test
// frequency, measured by single-bin DFTs over the last second (whole cycles,
// since every frequency is an integer Hz). Measured residuals for a 10 Hz
// high-pass at 48 and 384 kHz were at most 1.5e-6 dB and under 1e-7 rad; the
// bounds below sit above that so float32 rounding cannot flake the test. A
// predicted gain below the floor fails the fixture instead of loosening the
// bound, because float32 rounding noise limits what a deep stopband can show.
func TestFilterChainResponseMeasured(t *testing.T) {
	hp := func(fs, passes int) *FilterChain {
		c := NewFilterChain()
		if err := c.AddFilter(mustFilter(t, Band{Type: HighPass, Frequency: 10, Q: 0.7071, Passes: passes}, fs)); err != nil {
			t.Fatal(err)
		}
		return c
	}
	bwOnly := func() *FilterChain {
		lower, upper := thirdOctave(1000)
		bw, err := NewButterworthBandPass(lower, upper, 4, 48000)
		if err != nil {
			t.Fatal(err)
		}
		c := NewFilterChain()
		if err := c.AddFilter(bw); err != nil {
			t.Fatal(err)
		}
		return c
	}
	shelves := func() *FilterChain {
		c := NewFilterChain()
		for _, b := range []Band{
			{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6, Passes: 2},
			{Type: LowShelf, Frequency: 200, Q: 0.7071, GainDB: 4},
			{Type: HighShelf, Frequency: 8000, Q: 0.7071, GainDB: -3},
		} {
			if err := c.AddFilter(mustFilter(t, b, 48000)); err != nil {
				t.Fatal(err)
			}
		}
		return c
	}
	type row struct {
		name  string
		fs    int
		chain *FilterChain
		freqs []float64
		floor float64 // predicted gains below this fail the fixture
	}
	hpF48 := []float64{5, 10, 20, 100, 1000, 10000, 20000}
	hpF384 := []float64{5, 10, 20, 100, 1000, 10000, 100000}
	rows := []row{
		{"hp48 x1", 48000, hp(48000, 1), append([]float64{2}, hpF48...), -100},
		{"hp48 x2", 48000, hp(48000, 2), append([]float64{2}, hpF48...), -100},
		{"hp48 x4", 48000, hp(48000, 4), hpF48, -100},
		{"hp384 x1", 384000, hp(384000, 1), append([]float64{2}, hpF384...), -100},
		{"hp384 x2", 384000, hp(384000, 2), append([]float64{2}, hpF384...), -100},
		{"hp384 x4", 384000, hp(384000, 4), hpF384, -100},
		{"peak and shelves", 48000, shelves(), []float64{50, 200, 800, 1000, 1200, 4000, 8000, 16000}, -100},
		{"mixed", 48000, mixedChain(t, false), []float64{50, 200, 800, 1000, 1200, 4000, 8000, 16000}, -100},
		{"butterworth", 48000, bwOnly(), []float64{800, 891, 1000, 1122, 1300}, -100},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			pred := r.chain.Response(r.freqs)
			n := 2 * r.fs
			for k, f := range r.freqs {
				if pred[k].GainDB < r.floor {
					t.Fatalf("%g Hz predicted %g dB is below the measurable floor %g dB; fix the fixture", f, pred[k].GainDB, r.floor)
				}
				w := 2 * math.Pi * f / float64(r.fs)
				x32 := make([]float32, n)
				x64 := make([]float64, n)
				for i := range x32 {
					v := 0.5 * math.Sin(w*float64(i))
					x32[i] = float32(v)
					x64[i] = float64(x32[i])
				}
				tail := x32[n-r.fs:]
				xin := singleBin(tail, w)

				r.chain.Reset()
				y32 := make([]float32, n)
				if _, err := r.chain.ProcessInto(x32, y32); err != nil {
					t.Fatal(err)
				}
				r.chain.Reset()
				r.chain.ApplyFloat64(x64)
				r.chain.Reset()

				for leg, h := range map[string]complex128{
					"ProcessInto":  singleBin(y32[n-r.fs:], w) / xin,
					"ApplyFloat64": singleBin(x64[n-r.fs:], w) / xin,
				} {
					if d := math.Abs(20*math.Log10(cmplx.Abs(h)) - pred[k].GainDB); d > 1e-4 {
						t.Errorf("%s %g Hz: measured %g dB, predicted %g dB", leg, f, 20*math.Log10(cmplx.Abs(h)), pred[k].GainDB)
					}
					if d := math.Remainder(cmplx.Phase(h)-pred[k].PhaseRad, 2*math.Pi); math.Abs(d) > 1e-5 {
						t.Errorf("%s %g Hz: measured %g rad, predicted %g rad", leg, f, cmplx.Phase(h), pred[k].PhaseRad)
					}
				}
			}
		})
	}
}

// TestFilterChainResponseEmpty pins the identity response of an empty chain.
func TestFilterChainResponseEmpty(t *testing.T) {
	c := NewFilterChain()
	if c.SampleRate() != 0 || c.Len() != 0 {
		t.Fatalf("SampleRate %d, Len %d, want 0, 0", c.SampleRate(), c.Len())
	}
	for _, f := range []float64{0, 1000, 24000} {
		if got := c.Response([]float64{f})[0]; got != (ResponsePoint{Hz: f}) {
			t.Errorf("Response(%g) = %+v, want {Hz: %g, 0, 0}", f, got, f)
		}
	}
	nonFinite := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	dst := make([]ResponsePoint, len(nonFinite))
	if _, err := c.ResponseInto(dst, nonFinite); err != nil {
		t.Fatal(err)
	}
	for name, pts := range map[string][]ResponsePoint{"Response": c.Response(nonFinite), "ResponseInto": dst} {
		for i, p := range pts {
			if !math.IsNaN(p.GainDB) || !math.IsNaN(p.PhaseRad) {
				t.Errorf("%s(%g) = %+v, want NaN gain and phase", name, nonFinite[i], p)
			}
		}
	}
	if got := c.Response(nil); got == nil || len(got) != 0 {
		t.Errorf("Response(nil) = %#v, want non-nil empty", got)
	}
	if n, err := c.ResponseInto(nil, nil); n != 0 || err != nil {
		t.Errorf("ResponseInto(nil, nil) = %d, %v, want 0, nil", n, err)
	}
}

// TestFilterChainResponseIntoBufferTooSmall pins the dst contract.
func TestFilterChainResponseIntoBufferTooSmall(t *testing.T) {
	c := chainOf(t, testConfig())
	freqs := []float64{100, 1000, 10000}
	sentinel := ResponsePoint{Hz: -1, GainDB: -2, PhaseRad: -3}
	fill := func(n int) []ResponsePoint {
		d := make([]ResponsePoint, n)
		for i := range d {
			d[i] = sentinel
		}
		return d
	}

	short := fill(len(freqs) - 1)
	n, err := c.ResponseInto(short, freqs)
	if n != 0 || !errors.Is(err, dsp.ErrBufferTooSmall) {
		t.Fatalf("short dst: got %d, %v, want 0, ErrBufferTooSmall", n, err)
	}
	if !slices.Equal(short, fill(len(freqs)-1)) {
		t.Fatal("short dst was written")
	}

	long := fill(len(freqs) + 2)
	n, err = c.ResponseInto(long, freqs)
	if n != len(freqs) || err != nil {
		t.Fatalf("long dst: got %d, %v, want %d, nil", n, err, len(freqs))
	}
	if !slices.Equal(long[:n], c.Response(freqs)) {
		t.Error("ResponseInto differs from Response")
	}
	if !slices.Equal(long[n:], fill(2)) {
		t.Error("ResponseInto wrote past len(freqs)")
	}
}

// TestFilterChainResponseLeavesStateUntouched pins that Response and
// ResponseInto between two ProcessInto calls do not alter the stream.
func TestFilterChainResponseLeavesStateUntouched(t *testing.T) {
	x := eqSignal(4096)
	half := len(x) / 2
	freqs := []float64{50, 1000, 9000}

	run := func(c *FilterChain, probe bool) []float32 {
		out := make([]float32, len(x))
		if _, err := c.ProcessInto(x[:half], out[:half]); err != nil {
			t.Fatal(err)
		}
		if probe {
			_ = c.Response(freqs)
			if _, err := c.ResponseInto(make([]ResponsePoint, len(freqs)), freqs); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := c.ProcessInto(x[half:], out[half:]); err != nil {
			t.Fatal(err)
		}
		return out
	}
	if !slices.Equal(run(chainOf(t, testConfig()), true), run(chainOf(t, testConfig()), false)) {
		t.Fatal("Response changed the stream output")
	}
}

// TestSampleRate pins SampleRate on all three types.
func TestSampleRate(t *testing.T) {
	for _, tc := range filterCases() {
		if got := tc.build(t).SampleRate(); got != 48000 {
			t.Errorf("%s: Filter.SampleRate() = %d, want 48000", tc.name, got)
		}
	}
	if got := mustFilter(t, Band{Type: LowPass, Frequency: 1000, Q: 0.7071}, 384000).SampleRate(); got != 384000 {
		t.Errorf("Filter.SampleRate() = %d, want 384000", got)
	}
	for _, cfg := range []Config{{SampleRate: 48000}, testConfig(), {SampleRate: 96000}} {
		e, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if got := e.SampleRate(); got != cfg.SampleRate {
			t.Errorf("Equalizer.SampleRate() = %d, want %d", got, cfg.SampleRate)
		}
	}
	c := NewFilterChain()
	if c.SampleRate() != 0 {
		t.Errorf("empty chain SampleRate() = %d, want 0", c.SampleRate())
	}
	if err := c.AddFilter(mustFilter(t, Band{Type: LowPass, Frequency: 1000, Q: 0.7071}, 48000)); err != nil {
		t.Fatal(err)
	}
	if err := c.AddFilter(mustFilter(t, Band{Type: LowPass, Frequency: 1000, Q: 0.7071}, 44100)); err == nil {
		t.Fatal("rate mismatch accepted")
	}
	if c.SampleRate() != 48000 {
		t.Errorf("chain SampleRate() = %d, want 48000", c.SampleRate())
	}
}

// TestFilterChainResponseExtremeCascade pins that a cascade whose complex
// response product would overflow float64 (192 sections at +40 dB is 7680 dB, a
// magnitude of 1e384) or underflow to exactly zero (the mirror at -40 dB) still
// reports finite values equal to the sum of the per-section dB and phase.
func TestFilterChainResponseExtremeCascade(t *testing.T) {
	const fs, filters, passes = 48000, 12, 16
	freqs := []float64{200, 900, 1000, 1100, 5000}
	for _, gain := range []float64{40, -40} {
		band := Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: gain, Passes: passes}
		c := NewFilterChain()
		for range filters {
			if err := c.AddFilter(mustFilter(t, band, fs)); err != nil {
				t.Fatal(err)
			}
		}
		one := band
		one.Passes = 1
		single := mustFilter(t, one, fs).Response(freqs)
		for i, p := range c.Response(freqs) {
			n := float64(filters * passes)
			wantDB := n * single[i].GainDB
			if math.IsInf(p.GainDB, 0) || math.IsNaN(p.GainDB) {
				t.Fatalf("gain %+g dB, %g Hz: GainDB = %g, want finite %g", gain, freqs[i], p.GainDB, wantDB)
			}
			if math.Abs(p.GainDB-wantDB) > 1e-9*math.Max(1, math.Abs(wantDB)) {
				t.Errorf("gain %+g dB, %g Hz: GainDB = %.12g, want %.12g", gain, freqs[i], p.GainDB, wantDB)
			}
			if d := math.Remainder(p.PhaseRad-n*single[i].PhaseRad, 2*math.Pi); math.Abs(d) > 1e-9 {
				t.Errorf("gain %+g dB, %g Hz: PhaseRad = %g, want %g mod 2pi", gain, freqs[i], p.PhaseRad, n*single[i].PhaseRad)
			}
			if p.PhaseRad < -math.Pi || p.PhaseRad > math.Pi {
				t.Errorf("gain %+g dB, %g Hz: PhaseRad = %g outside [-pi, pi]", gain, freqs[i], p.PhaseRad)
			}
		}
	}
}

// TestFilterChainResponseExactNullFloored pins that a section whose magnitude is
// exactly zero contributes SectionFloorDB wherever it sits in the cascade,
// rather than being skipped, averaged away or turning the sum into -Inf.
func TestFilterChainResponseExactNullFloored(t *testing.T) {
	const fs = 48000
	null, err := NewBiquadFilter([]Biquad{{}}, fs) // B0=B1=B2=0: H(z) = 0 at every frequency
	if err != nil {
		t.Fatal(err)
	}
	peak := func() *Filter {
		return mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6}, fs)
	}
	freqs := []float64{100, 1000, 9000}
	orders := map[string][]*Filter{
		"null alone":  {null},
		"null first":  {null, peak(), peak()},
		"null middle": {peak(), null, peak()},
		"null last":   {peak(), peak(), null},
	}
	for name, fl := range orders {
		c := NewFilterChain()
		want := make([]float64, len(freqs))
		for _, f := range fl {
			if err := c.AddFilter(f); err != nil {
				t.Fatal(err)
			}
			for i, p := range f.Response(freqs) {
				if f != null {
					want[i] += p.GainDB
				}
			}
		}
		for i, p := range c.Response(freqs) {
			if math.IsInf(p.GainDB, 0) || math.IsNaN(p.GainDB) {
				t.Errorf("%s, %g Hz: GainDB = %g, want finite", name, freqs[i], p.GainDB)
			}
			if w := want[i] + SectionFloorDB; math.Abs(p.GainDB-w) > 1e-12 {
				t.Errorf("%s, %g Hz: GainDB = %.15g, want %.15g (floor plus the peaks)", name, freqs[i], p.GainDB, w)
			}
		}
	}
}
