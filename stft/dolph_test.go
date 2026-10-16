package stft

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

// dolphAttenuations are the attenuations the sidelobe tests sweep: a generic 50
// dB and three values in the 80 to 130 dB range used for spectrogram windows.
var dolphAttenuations = []float64{50, 86.4, 106.5, 126.6}

// dolphGens are both generators, named for subtests.
var dolphGens = []struct {
	name string
	gen  func(int, float64) ([]float32, error)
}{
	{"periodic", DolphChebyshev},
	{"symmetric", DolphChebyshevSymmetric},
}

// TestDolphChebyshevAgainstScipyGolden compares both generators with scipy
// chebwin output, so an agreement with the reference does not rest on the
// in-test formula sharing code with the implementation. The fixture is static
// test data; scipy runs offline (reference tools stay test-time only).
//
// The fixture was produced by the following script (scipy 1.18.1, numpy 2.5.4),
// run outside the repository with `uv run --no-project --with scipy --with numpy
// python -I gen_chebwin_golden.py out.json`:
//
//	import json, sys, warnings
//	import numpy as np, scipy
//	from scipy.signal.windows import chebwin
//
//	CASES = [  # (n, attenuation_db, sym)
//	    (3, 50.0, False), (7, 50.0, False), (8, 50.0, False), (31, 86.4, False),
//	    (64, 86.4, False), (64, 106.5, False), (64, 126.6, False), (127, 60.0, False),
//	    (1024, 150.0, False), (8192, 40.0, False), (8192, 86.4, False),
//	    (8192, 106.5, False), (8192, 126.6, False),
//	    (7, 50.0, True), (8, 50.0, True), (64, 106.5, True), (8192, 106.5, True),
//	]
//	FULL_MAX = 512
//	out = {
//	    "_source": "scipy.signal.windows.chebwin",
//	    "_scipy_version": scipy.__version__,
//	    "_numpy_version": np.__version__,
//	    "_note": "...",
//	    "cases": [],
//	}
//	with warnings.catch_warnings():
//	    warnings.simplefilter("ignore")  # chebwin warns below 45 dB
//	    for n, att, sym in CASES:
//	        w = chebwin(n, att, sym=sym)
//	        if n <= FULL_MAX:
//	            idx = list(range(n))
//	        else:
//	            idx = sorted(set(range(0, n, n // 64)) | {1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 2, n - 1})
//	        out["cases"].append({
//	            "name": f"{'symmetric' if sym else 'periodic'}_n{n}_att{att:g}",
//	            "n": n, "attenuation_db": att, "sym": sym,
//	            "indices": idx, "values": [float(w[i]) for i in idx],
//	        })
//	json.dump(out, open(sys.argv[1], "w"), indent=2)
//
// Windows longer than 512 samples store a decimated index subset (every n/64-th
// index plus the samples next to the center and near the last index), with the
// indices listed explicitly.
//
// Tolerance: float32 rounding is at most 2^-24 relative (about 6e-8), so 2e-7 of
// the reference is about 3 ulps. The absolute 1e-9 term (-180 dB re the peak of
// 1) covers tiny edge samples, where the float64 construction here and scipy's
// FFT differ by up to about 1.2e-10. The worst case over the fixture is about a
// quarter of the combined tolerance. The test fails if the periodic or symmetric
// convention flips, the attenuation maps wrongly, or the normalization moves off
// the peak.
func TestDolphChebyshevAgainstScipyGolden(t *testing.T) {
	data, err := os.ReadFile("testdata/scipy_chebwin_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Name          string    `json:"name"`
			N             int       `json:"n"`
			AttenuationDB float64   `json:"attenuation_db"`
			Sym           bool      `json:"sym"`
			Indices       []int     `json:"indices"`
			Values        []float64 `json:"values"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden.Cases) != 17 {
		t.Fatalf("loaded %d golden cases, want 17", len(golden.Cases))
	}
	for _, gc := range golden.Cases {
		t.Run(gc.Name, func(t *testing.T) {
			gen := DolphChebyshev
			if gc.Sym {
				gen = DolphChebyshevSymmetric
			}
			w, err := gen(gc.N, gc.AttenuationDB)
			if err != nil {
				t.Fatal(err)
			}
			if len(w) != gc.N {
				t.Fatalf("len = %d, want %d", len(w), gc.N)
			}
			if len(gc.Indices) != len(gc.Values) || len(gc.Indices) == 0 {
				t.Fatalf("fixture has %d indices and %d values", len(gc.Indices), len(gc.Values))
			}
			for k, i := range gc.Indices {
				ref := gc.Values[k]
				if d := math.Abs(float64(w[i]) - ref); d > 2e-7*math.Abs(ref)+1e-9 {
					t.Fatalf("w[%d] = %.9g, scipy %.9g (diff %.3g)", i, w[i], ref, d)
				}
			}
		})
	}
}

// TestDolphChebyshevPeriodicIsTruncatedSymmetric pins the periodic definition:
// the symmetric window of n+1 points with its last sample dropped, produced by
// one float64 core, so the match is bit for bit.
func TestDolphChebyshevPeriodicIsTruncatedSymmetric(t *testing.T) {
	for _, n := range []int{1, 2, 3, 7, 8, 64, 127, 1024} {
		for _, att := range []float64{50, 106.5} {
			per, err := DolphChebyshev(n, att)
			if err != nil {
				t.Fatal(err)
			}
			symN1, err := DolphChebyshevSymmetric(n+1, att)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(per, symN1[:n]) {
				t.Errorf("n=%d att=%g: periodic != symmetric(n+1)[:n]", n, att)
			}
			for i := 1; i < n; i++ {
				if per[i] != per[n-i] {
					t.Errorf("n=%d att=%g: periodic w[%d]=%g != w[%d]=%g", n, att, i, per[i], n-i, per[n-i])
					break
				}
			}
			sym, err := DolphChebyshevSymmetric(n, att)
			if err != nil {
				t.Fatal(err)
			}
			for i := range sym {
				if sym[i] != sym[n-1-i] {
					t.Errorf("n=%d att=%g: symmetric w[%d]=%g != w[%d]=%g", n, att, i, sym[i], n-1-i, sym[n-1-i])
					break
				}
			}
			// The two forms of one n must differ somewhere, or the comparison above
			// could not tell a convention swap.
			if n >= 2 && slices.Equal(per, sym) {
				t.Errorf("n=%d att=%g: periodic and symmetric windows are identical", n, att)
			}
		}
	}
}

// maxSidelobeDB returns the highest sidelobe of the float32 window w in dB
// relative to its DC gain. The response is the float64 zero-phase DTFT sampled at
// 32 points per DFT bin over [0, pi]; the main lobe ends at the first local
// minimum after DC, and everything beyond it counts as sidelobe.
func maxSidelobeDB(w []float32) float64 {
	n := len(w)
	pts := 16*n + 1
	mid := float64(n-1) / 2
	resp := make([]float64, pts)
	for k := range resp {
		om := math.Pi * float64(k) / float64(pts-1)
		var s float64
		for i, v := range w {
			s += float64(v) * math.Cos(om*(float64(i)-mid))
		}
		resp[k] = math.Abs(s)
	}
	k := 1
	for k+1 < pts && !(resp[k] <= resp[k-1] && resp[k] <= resp[k+1]) {
		k++
	}
	peak := 0.0
	for _, v := range resp[k:] {
		peak = max(peak, v)
	}
	return 20 * math.Log10(peak/resp[0])
}

// TestDolphChebyshevSymmetricSidelobeLevel checks the defining property: the
// highest sidelobe of the symmetric window sits at -attenuationDB. Both
// directions are pinned, because equiripple puts the highest sidelobe at the
// design level and the float32 cast moves it by at most about 0.3 dB. It fails
// for a wrong cosh argument, a wrong order or a wrong dB mapping.
func TestDolphChebyshevSymmetricSidelobeLevel(t *testing.T) {
	for _, n := range []int{8, 31, 64, 127, 512} {
		for _, att := range dolphAttenuations {
			w, err := DolphChebyshevSymmetric(n, att)
			if err != nil {
				t.Fatal(err)
			}
			got := maxSidelobeDB(w)
			if got < -att-0.5 || got > -att+0.5 {
				t.Errorf("n=%d att=%g: highest sidelobe %.2f dB, want within 0.5 dB of %g", n, att, got, -att)
			}
		}
	}
}

// TestDolphChebyshevPeriodicSidelobeCost pins the cost of dropping the last
// sample, which the DolphChebyshev godoc states: the periodic window's highest
// sidelobe sits a few dB above the design level. It fails if the periodic form
// silently becomes the symmetric n-point window, whose deviation is about 0.
func TestDolphChebyshevPeriodicSidelobeCost(t *testing.T) {
	for _, n := range []int{512, 1024} {
		for _, att := range dolphAttenuations {
			w, err := DolphChebyshev(n, att)
			if err != nil {
				t.Fatal(err)
			}
			got := maxSidelobeDB(w)
			if got < -att+3 || got > -att+4.7 {
				t.Errorf("n=%d att=%g: highest sidelobe %.2f dB, want in [%g, %g]", n, att, got, -att+3, -att+4.7)
			}
		}
	}
}

func TestDolphChebyshevInvalid(t *testing.T) {
	check := func(t *testing.T, w []float32, err error, substr string) {
		t.Helper()
		if !errors.Is(err, dsp.ErrInvalidConfig) {
			t.Fatalf("err = %v, want ErrInvalidConfig", err)
		}
		// ErrInvalidConfig is a shared sentinel, so the message pins which rule fired.
		if !strings.Contains(err.Error(), substr) {
			t.Errorf("err = %q, want to contain %q", err, substr)
		}
		if w != nil {
			t.Errorf("window = %v on error, want nil", w)
		}
	}
	for _, g := range dolphGens {
		t.Run(g.name, func(t *testing.T) {
			for _, n := range []int{0, -1, 32769} {
				w, err := g.gen(n, 100)
				check(t, w, err, "length")
			}
			for _, att := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0, -10, 19.999, 200.001} {
				w, err := g.gen(64, att)
				check(t, w, err, "attenuationDB")
			}
		})
	}
}

// TestDolphChebyshevDomainCorners evaluates both forms at the corners of the
// accepted domain. Every sample must be finite and positive, with the peak
// exactly 1: this pins the NaN-free guarantee and the length cap, past which the
// float64 evaluation measurably loses positivity.
func TestDolphChebyshevDomainCorners(t *testing.T) {
	cases := []struct {
		n   int
		att float64
	}{{1, 20}, {1, 200}, {2, 20}, {2, 200}, {3, 20}, {3, 200}, {32768, 20}, {32768, 200}}
	for _, c := range cases {
		if testing.Short() && c.n > 8192 {
			continue
		}
		for _, g := range dolphGens {
			name := g.name
			w, err := g.gen(c.n, c.att)
			if err != nil {
				t.Fatalf("%s n=%d att=%g: %v", name, c.n, c.att, err)
			}
			if len(w) != c.n {
				t.Fatalf("%s n=%d att=%g: len %d", name, c.n, c.att, len(w))
			}
			peak := float32(0)
			for i, v := range w {
				if !(v > 0) || v > 1 || math.IsInf(float64(v), 0) {
					t.Fatalf("%s n=%d att=%g: w[%d] = %g, want finite in (0, 1]", name, c.n, c.att, i, v)
				}
				peak = max(peak, v)
			}
			if peak != 1 {
				t.Errorf("%s n=%d att=%g: peak %g, want exactly 1", name, c.n, c.att, peak)
			}
		}
	}
}

// TestDolphChebyshevThroughConfig is an end-to-end check that the generators
// produce a valid CustomWindow, including an odd length placed inside a larger
// frame.
func TestDolphChebyshevThroughConfig(t *testing.T) {
	w, err := DolphChebyshev(256, 106.5)
	if err != nil {
		t.Fatal(err)
	}
	p, err := New(Config{FrameSize: 256, CustomWindow: w})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(p.Window(), w) {
		t.Error("Window() differs from the CustomWindow passed in")
	}

	short, err := DolphChebyshev(201, 86.4)
	if err != nil {
		t.Fatal(err)
	}
	p, err = New(Config{FrameSize: 256, WindowLength: 201, CustomWindow: short})
	if err != nil {
		t.Fatal(err)
	}
	got := p.Window()
	const off = (256 - 201) / 2 // 27
	if !slices.Equal(got[off:off+201], short) {
		t.Error("odd window not placed at the centered offset")
	}
	for i := range got {
		if (i < off || i >= off+201) && got[i] != 0 {
			t.Fatalf("pad w[%d] = %g, want 0", i, got[i])
		}
	}
}

func BenchmarkDolphChebyshev(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DolphChebyshev(8192, 106.5); err != nil {
			b.Fatal(err)
		}
	}
}
