package mel

import (
	"errors"
	"math"
	"testing"
)

// hzToMelRef and melToHzRef are independent float64 transcriptions of the
// librosa formulas, used as the oracle for HzToMel/MelToHz and the naive
// filterbank. They must not call the package functions under test.
func hzToMelRef(hz float64, scale MelScale) float64 {
	if scale == HTK {
		return 2595 * math.Log10(1+hz/700)
	}
	const fsp = 200.0 / 3.0
	const minLogHz = 1000.0
	const minLogMel = minLogHz / fsp
	logstep := math.Log(6.4) / 27
	if hz < minLogHz {
		return hz / fsp
	}
	return minLogMel + math.Log(hz/minLogHz)/logstep
}

func melToHzRef(mel float64, scale MelScale) float64 {
	if scale == HTK {
		return 700 * (math.Pow(10, mel/2595) - 1)
	}
	const fsp = 200.0 / 3.0
	const minLogHz = 1000.0
	const minLogMel = minLogHz / fsp
	logstep := math.Log(6.4) / 27
	if mel < minLogMel {
		return fsp * mel
	}
	return minLogHz * math.Exp(logstep*(mel-minLogMel))
}

// naiveMel builds the dense float64 mel filterbank straight from the librosa
// formulas (section 5), the oracle Filterbank.Dense() is checked against.
func naiveMel(sr, frameSize, numMels int, minHz, maxHz float64, scale MelScale, norm Norm) [][]float64 {
	numBins := frameSize/2 + 1
	nyq := float64(sr) / 2
	if maxHz <= 0 || maxHz > nyq {
		maxHz = nyq
	}
	fftfreqs := make([]float64, numBins)
	for k := range fftfreqs {
		fftfreqs[k] = float64(k) * float64(sr) / float64(frameSize)
	}
	melMin := hzToMelRef(minHz, scale)
	melMax := hzToMelRef(maxHz, scale)
	pts := make([]float64, numMels+2)
	for i := range pts {
		pts[i] = melToHzRef(melMin+(melMax-melMin)*float64(i)/float64(numMels+1), scale)
	}
	fb := make([][]float64, numMels)
	for m := range fb {
		fb[m] = make([]float64, numBins)
		lo, ctr, hi := pts[m], pts[m+1], pts[m+2]
		for k := range numBins {
			lower := (fftfreqs[k] - lo) / (ctr - lo)
			upper := (hi - fftfreqs[k]) / (hi - ctr)
			w := math.Max(0, math.Min(lower, upper))
			if norm == NormSlaney {
				w *= 2 / (hi - lo)
			}
			fb[m][k] = w
		}
	}
	return fb
}

func rowPeak(row []float64) float64 {
	p := 0.0
	for _, v := range row {
		p = math.Max(p, math.Abs(v))
	}
	return p
}

func TestFilterbankAgainstNaive(t *testing.T) {
	cases := []struct {
		name       string
		sr, n, mel int
		minHz, max float64
	}{
		{"bsgbat", 384000, 1024, 128, 9000, 150000},
		{"tenvad40", 16000, 1024, 40, 0, 0},
		{"small", 16000, 256, 16, 100, 7000},
		{"tiny", 8000, 64, 4, 0, 4000},
	}
	for _, tc := range cases {
		for _, scale := range []MelScale{Slaney, HTK} {
			for _, norm := range []Norm{NormSlaney, NormNone} {
				t.Run(tc.name, func(t *testing.T) {
					fb, err := NewFilterbank(FilterbankConfig{
						SampleRate: tc.sr, FrameSize: tc.n, NumMels: tc.mel,
						MinHz: tc.minHz, MaxHz: tc.max, Scale: scale, Norm: norm,
					})
					if err != nil {
						t.Fatalf("NewFilterbank: %v", err)
					}
					if fb.NumMels() != tc.mel || fb.NumBins() != tc.n/2+1 {
						t.Fatalf("shape = %dx%d, want %dx%d", fb.NumMels(), fb.NumBins(), tc.mel, tc.n/2+1)
					}
					ref := naiveMel(tc.sr, tc.n, tc.mel, tc.minHz, tc.max, scale, norm)
					dense := fb.Dense()
					for m := range ref {
						peak := rowPeak(ref[m])
						tol := 1e-6 * peak
						if peak == 0 {
							tol = 1e-12
						}
						for k := range ref[m] {
							if math.Abs(float64(dense[m][k])-ref[m][k]) > tol {
								t.Fatalf("row %d bin %d: got %g, want %g (peak %g)", m, k, dense[m][k], ref[m][k], peak)
							}
						}
					}
				})
			}
		}
	}
}

func TestFilterbankSparseSpansMatchDense(t *testing.T) {
	fb, err := NewFilterbank(FilterbankConfig{SampleRate: 16000, FrameSize: 512, NumMels: 24, MinHz: 50, MaxHz: 7000})
	if err != nil {
		t.Fatal(err)
	}
	dense := fb.Dense()
	for m := range fb.NumMels() {
		start, w := fb.Row(m)
		// Reconstruct the dense row from the sparse span and compare.
		got := make([]float32, fb.NumBins())
		copy(got[start:], w)
		for k := range fb.NumBins() {
			if got[k] != dense[m][k] {
				t.Fatalf("row %d bin %d: sparse %g != dense %g", m, k, got[k], dense[m][k])
			}
		}
		// The sparse weights slice has no leading or trailing zero (contiguous,
		// trimmed support); interior zeros may remain.
		if len(w) > 0 && (w[0] == 0 || w[len(w)-1] == 0) {
			t.Fatalf("row %d span not trimmed: start=%d w=%v", m, start, w)
		}
	}
}

func TestFilterbankNarrowAndEmptyRows(t *testing.T) {
	// The BSG-BAT shape places low-frequency triangles narrower than the bin
	// spacing, so the lowest rows cover a single bin (measured min width 1 at
	// FrameSize 1024; no row is fully empty at this resolution).
	fb, err := NewFilterbank(FilterbankConfig{SampleRate: 384000, FrameSize: 1024, NumMels: 128, MinHz: 9000, MaxHz: 150000})
	if err != nil {
		t.Fatal(err)
	}
	minW := 1 << 30
	for m := range fb.NumMels() {
		if _, w := fb.Row(m); len(w) < minW {
			minW = len(w)
		}
	}
	if minW != 1 {
		t.Errorf("BSG-BAT min row width = %d, want 1 (narrow single-bin low rows)", minW)
	}

	// A coarser transform (bins wider than the low triangles) yields fully empty
	// rows. librosa warns and continues; this package returns the empty span
	// silently, so it projects to 0. Assert the empty rows exactly match which
	// rows the naive float64 builder leaves all-zero.
	const sr, n, mels = 384000, 256, 128
	const minHz, maxHz = 9000.0, 150000.0
	cfb, err := NewFilterbank(FilterbankConfig{SampleRate: sr, FrameSize: n, NumMels: mels, MinHz: minHz, MaxHz: maxHz})
	if err != nil {
		t.Fatal(err)
	}
	ref := naiveMel(sr, n, mels, minHz, maxHz, Slaney, NormSlaney)
	empty := 0
	for m := range cfb.NumMels() {
		_, w := cfb.Row(m)
		refEmpty := rowPeak(ref[m]) == 0
		if (len(w) == 0) != refEmpty {
			t.Fatalf("row %d empty = %v, naive empty = %v", m, len(w) == 0, refEmpty)
		}
		if len(w) == 0 {
			empty++
		}
	}
	if empty == 0 {
		t.Fatal("expected fully empty rows in the coarse-bin config")
	}
}

func TestHzMelKnownPoints(t *testing.T) {
	// Slaney anchors: 1000 Hz == mel 15, 200/3 Hz == mel 1.
	if got := HzToMel(1000, Slaney); math.Abs(got-15) > 1e-9 {
		t.Errorf("HzToMel(1000, Slaney) = %g, want 15", got)
	}
	if got := HzToMel(200.0/3.0, Slaney); math.Abs(got-1) > 1e-12 {
		t.Errorf("HzToMel(200/3, Slaney) = %g, want 1", got)
	}
	if got := MelToHz(15, Slaney); math.Abs(got-1000) > 1e-9 {
		t.Errorf("MelToHz(15, Slaney) = %g, want 1000", got)
	}
	// HTK anchors: 0 Hz == mel 0, 1000 Hz == 2595*log10(1+1000/700).
	if got := HzToMel(0, HTK); math.Abs(got) > 1e-12 {
		t.Errorf("HzToMel(0, HTK) = %g, want 0", got)
	}
	if got, want := HzToMel(1000, HTK), 2595*math.Log10(1+1000.0/700); math.Abs(got-want) > 1e-9 {
		t.Errorf("HzToMel(1000, HTK) = %g, want %g", got, want)
	}
}

func TestHzMelRoundTrip(t *testing.T) {
	for _, scale := range []MelScale{Slaney, HTK} {
		for _, hz := range []float64{50, 500, 999, 1000, 1001, 8000, 96000} {
			m := HzToMel(hz, scale)
			back := MelToHz(m, scale)
			if math.Abs(back-hz) > 1e-6*hz+1e-9 {
				t.Errorf("scale %d: MelToHz(HzToMel(%g)) = %g", scale, hz, back)
			}
		}
	}
}

func TestFilterbankFromRowsRoundTrip(t *testing.T) {
	rows := [][]float32{
		{0, 0, 1, 2, 1, 0}, // trimmed to bins 2..4
		{0, 0, 0, 0, 0, 0}, // all zero -> empty span
		{3, 0, 4, 0, 0, 0}, // interior zero preserved: bins 0..2
	}
	fb, err := FilterbankFromRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	if fb.NumMels() != 3 || fb.NumBins() != 6 {
		t.Fatalf("shape = %dx%d, want 3x6", fb.NumMels(), fb.NumBins())
	}
	dense := fb.Dense()
	for m := range rows {
		for k := range rows[m] {
			if dense[m][k] != rows[m][k] {
				t.Fatalf("row %d bin %d: got %g, want %g", m, k, dense[m][k], rows[m][k])
			}
		}
	}
	// All-zero row is an empty span.
	if _, w := fb.Row(1); len(w) != 0 {
		t.Fatalf("all-zero row span len = %d, want 0", len(w))
	}
	// Interior zero preserved: row 2 span is bins 0..2 with a zero in the middle.
	if start, w := fb.Row(2); start != 0 || len(w) != 3 || w[1] != 0 {
		t.Fatalf("row 2 span = (start %d, %v), want (0, [3 0 4])", start, w)
	}
}

func TestFilterbankFromRowsInputCopied(t *testing.T) {
	rows := [][]float32{{0, 1, 2, 1, 0, 0}}
	fb, err := FilterbankFromRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	rows[0][2] = 99 // mutate caller slice after construction
	if _, w := fb.Row(0); w[1] == 99 {
		t.Fatal("Filterbank retained a reference to the caller's rows")
	}
}

func TestFilterbankFromRowsInvalid(t *testing.T) {
	cases := []struct {
		name string
		rows [][]float32
	}{
		{"no rows", nil},
		{"empty rows", [][]float32{}},
		{"ragged", [][]float32{{0, 1, 0}, {0, 1}}},
		{"too few bins", [][]float32{{1}}},
		{"nan", [][]float32{{0, float32(math.NaN()), 0}}},
		{"inf", [][]float32{{0, float32(math.Inf(1)), 0}}},
		{"negative", [][]float32{{0, 1, -0.5, 0}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := FilterbankFromRows(tc.rows); !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("err = %v, want ErrInvalidConfig", err)
			}
		})
	}
}

func TestNewFilterbankInvalid(t *testing.T) {
	base := FilterbankConfig{SampleRate: 16000, FrameSize: 256, NumMels: 20, MinHz: 50, MaxHz: 7000}
	mut := func(f func(*FilterbankConfig)) FilterbankConfig {
		c := base
		f(&c)
		return c
	}
	cases := []struct {
		name string
		cfg  FilterbankConfig
	}{
		{"sample rate", mut(func(c *FilterbankConfig) { c.SampleRate = 0 })},
		{"frame size not pow2", mut(func(c *FilterbankConfig) { c.FrameSize = 100 })},
		{"frame size too small", mut(func(c *FilterbankConfig) { c.FrameSize = 1 })},
		{"num mels", mut(func(c *FilterbankConfig) { c.NumMels = 0 })},
		{"min hz negative", mut(func(c *FilterbankConfig) { c.MinHz = -1 })},
		{"min hz nan", mut(func(c *FilterbankConfig) { c.MinHz = math.NaN() })},
		{"max hz nan", mut(func(c *FilterbankConfig) { c.MaxHz = math.NaN() })},
		{"max hz above nyquist", mut(func(c *FilterbankConfig) { c.MaxHz = 9000 })},
		{"min ge max", mut(func(c *FilterbankConfig) { c.MinHz = 6000; c.MaxHz = 5000 })},
		{"scale", mut(func(c *FilterbankConfig) { c.Scale = MelScale(9) })},
		{"norm", mut(func(c *FilterbankConfig) { c.Norm = Norm(9) })},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewFilterbank(tc.cfg); !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("err = %v, want ErrInvalidConfig", err)
			}
		})
	}
}
