package stft

import (
	"math"
	"testing"
)

func TestGenerateWindowHannPeriodic(t *testing.T) {
	const n = 16
	w := GenerateWindow(Hann, n)
	if len(w) != n {
		t.Fatalf("len %d, want %d", len(w), n)
	}
	if w[0] != 0 {
		t.Errorf("w[0] = %g, want 0", w[0])
	}
	if w[n/2] != 1 {
		t.Errorf("w[n/2] = %g, want 1", w[n/2])
	}
	if math.Abs(float64(w[n/4])-0.5) > 1e-6 {
		t.Errorf("w[n/4] = %g, want 0.5", w[n/4])
	}
	// Periodic symmetry: w[i] == w[n-i] for 0 < i < n.
	for i := 1; i < n; i++ {
		if math.Abs(float64(w[i]-w[n-i])) > 1e-6 {
			t.Errorf("w[%d]=%g != w[%d]=%g", i, w[i], n-i, w[n-i])
		}
	}
}

// TestGenerateWindowHannFormula pins the exact float64->float32 Hann the denoiser
// depends on for bit-identical framing. If this drifts, the denoiser's output
// bits move.
func TestGenerateWindowHannFormula(t *testing.T) {
	for _, n := range []int{64, 256, 1024} {
		w := GenerateWindow(Hann, n)
		for i := range w {
			want := float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
			if w[i] != want {
				t.Fatalf("n=%d w[%d] = %v, want exact %v", n, i, w[i], want)
			}
		}
	}
}

func TestGenerateWindowShapes(t *testing.T) {
	const n = 32
	rect := GenerateWindow(Rectangular, n)
	for i, v := range rect {
		if v != 1 {
			t.Fatalf("rect[%d] = %g, want 1", i, v)
		}
	}
	sym := GenerateWindow(HannSymmetric, n)
	if sym[0] != 0 || math.Abs(float64(sym[n-1])) > 1e-6 {
		t.Errorf("symmetric Hann endpoints = %g, %g, want ~0, ~0", sym[0], sym[n-1])
	}
	ham := GenerateWindow(Hamming, n)
	if math.Abs(float64(ham[0])-0.08) > 1e-6 { // 0.54 - 0.46
		t.Errorf("hamming[0] = %g, want ~0.08", ham[0])
	}
	// Degenerate lengths.
	if got := GenerateWindow(Hann, 1); len(got) != 1 || got[0] != 1 {
		t.Errorf("GenerateWindow(Hann,1) = %v, want [1]", got)
	}
	if got := GenerateWindow(Hann, 0); len(got) != 0 {
		t.Errorf("GenerateWindow(Hann,0) = %v, want empty", got)
	}
}

func TestWOLANormHannQuarterHopIsConstant(t *testing.T) {
	const n = 1024
	w := GenerateWindow(Hann, n)
	norm := WOLANorm(w, w, n/4)
	if len(norm) != n/4 {
		t.Fatalf("len %d, want %d", len(norm), n/4)
	}
	for i, v := range norm {
		if math.Abs(float64(v)-1.5) > 1e-5 {
			t.Fatalf("norm[%d] = %g, want 1.5", i, v)
		}
	}
	// hop n/8: 8 overlapping Hann^2 copies sum to 3.0 as well.
	for i, v := range WOLANorm(w, w, n/8) {
		if math.Abs(float64(v)-3.0) > 1e-5 {
			t.Fatalf("hop n/8 norm[%d] = %g, want 3.0", i, v)
		}
	}
}

func TestWOLANormOtherHopsArePositive(t *testing.T) {
	const n = 256
	w := GenerateWindow(Hann, n)
	for _, hop := range []int{n / 2, n / 16, 1} {
		for i, v := range WOLANorm(w, w, hop) {
			if !(v > 0.4) {
				t.Fatalf("hop %d norm[%d] = %g, want > 0.4", hop, i, v)
			}
		}
	}
	ones := GenerateWindow(Rectangular, n)
	for i, v := range WOLANorm(ones, ones, n/4) {
		if v != 4 {
			t.Fatalf("rect norm[%d] = %g, want 4", i, v)
		}
	}
}

func TestWindowString(t *testing.T) {
	cases := map[Window]string{
		Hann:          "hann",
		HannSymmetric: "hann-symmetric",
		Hamming:       "hamming",
		Rectangular:   "rectangular",
	}
	for w, want := range cases {
		if got := w.String(); got != want {
			t.Errorf("Window(%d).String() = %q, want %q", int(w), got, want)
		}
	}
	if got := Window(99).String(); got != "Window(99)" {
		t.Errorf("unknown window String() = %q, want %q", got, "Window(99)")
	}
}

func TestWOLANormEdgeCases(t *testing.T) {
	// hop < 1 returns an empty slice rather than panicking.
	if got := WOLANorm([]float32{1, 1}, []float32{1, 1}, 0); len(got) != 0 {
		t.Errorf("WOLANorm hop=0 len = %d, want 0", len(got))
	}
	// len(ws) < len(wa): wa is truncated to len(ws); no out-of-range access.
	got := WOLANorm([]float32{1, 1, 1, 1}, []float32{2, 2}, 2)
	if len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Errorf("WOLANorm (len(ws)<len(wa)) = %v, want [2 2]", got)
	}
	// len(wa) < len(ws): the loop is bounded by len(wa); no out-of-range access.
	got2 := WOLANorm([]float32{3, 3}, []float32{2, 2, 2, 2}, 2)
	if len(got2) != 2 || got2[0] != 6 || got2[1] != 6 {
		t.Errorf("WOLANorm (len(wa)<len(ws)) = %v, want [6 6]", got2)
	}
}
