package stft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

// testSignal is a deterministic mix of tones plus a little LCG noise.
func testSignal(n int) []float32 {
	x := make([]float32, n)
	var s uint32 = 777
	for i := range x {
		s = s*1664525 + 1013904223
		nz := (float64(s>>9)/float64(1<<23) - 1) * 0.05
		x[i] = float32(0.6*math.Sin(2*math.Pi*float64(i)*0.017) +
			0.3*math.Sin(2*math.Pi*float64(i)*0.11) + nz)
	}
	return x
}

// reflectIdx mirrors numpy "reflect" (edge not repeated), period 2*(n-1).
func reflectIdx(idx, n int) int {
	if n == 1 {
		return 0
	}
	period := (n - 1) * 2
	m := idx % period
	if m < 0 {
		m += period
	}
	if m < n {
		return m
	}
	return period - m
}

func sampleAt(sig []float32, idx int, pad PadMode) float64 {
	if idx >= 0 && idx < len(sig) {
		return float64(sig[idx])
	}
	if pad == PadReflect {
		return float64(sig[reflectIdx(idx, len(sig))])
	}
	return 0
}

// naiveSTFT computes the windowed real-input DFT half-spectrum per frame in
// float64, matching simd's framing: NoPad frame f starts at f*hop; a centered
// frame starts at f*hop - n/2 with the pad's out-of-range rule. It validates the
// package's framing, windowing and bin layout independently of simd.
func naiveSTFT(sig, window []float32, n, hop int, pad PadMode) [][]complex128 {
	frames := 0
	off := 0
	if pad == NoPad {
		if len(sig) >= n {
			frames = 1 + (len(sig)-n)/hop
		}
	} else {
		if len(sig) > 0 {
			frames = 1 + len(sig)/hop
		}
		off = n / 2
	}
	bins := n/2 + 1
	out := make([][]complex128, frames)
	for f := range frames {
		base := f*hop - off
		row := make([]complex128, bins)
		for k := range bins {
			var acc complex128
			for j := range n {
				x := sampleAt(sig, base+j, pad) * float64(window[j])
				ang := -2 * math.Pi * float64(k) * float64(j) / float64(n)
				acc += complex(x, 0) * cmplx.Exp(complex(0, ang))
			}
			row[k] = acc
		}
		out[f] = row
	}
	return out
}

func closeC(got complex64, want complex128) bool {
	dr := math.Abs(float64(real(got)) - real(want))
	di := math.Abs(float64(imag(got)) - imag(want))
	tol := 1e-3 + 1e-3*cmplx.Abs(want)
	return dr <= tol && di <= tol
}

func TestSpectrumMatchesNaiveDFT(t *testing.T) {
	for _, pad := range []PadMode{NoPad, PadZero, PadReflect} {
		for _, n := range []int{64, 128} {
			hop := n / 4
			sig := testSignal(5 * n)
			p, err := New(Config{FrameSize: n, HopSize: hop, Window: Hann})
			if err != nil {
				t.Fatal(err)
			}
			ref := naiveSTFT(sig, p.Window(), n, hop, pad)
			nf := p.NumFrames(len(sig), pad)
			if nf != len(ref) {
				t.Fatalf("pad=%d n=%d NumFrames=%d, want %d", pad, n, nf, len(ref))
			}
			dst := make([][]complex64, nf)
			for f := range dst {
				dst[f] = make([]complex64, p.NumBins())
			}
			if got := p.Spectrum(dst, sig, pad); got != nf {
				t.Fatalf("Spectrum wrote %d frames, want %d", got, nf)
			}
			for f := range ref {
				for k := range ref[f] {
					if !closeC(dst[f][k], ref[f][k]) {
						t.Fatalf("pad=%d n=%d frame %d bin %d: got %v want %v", pad, n, f, k, dst[f][k], ref[f][k])
					}
				}
			}
		}
	}
}

func TestPowerIntoMatchesNaive(t *testing.T) {
	const n, hop = 128, 32
	sig := testSignal(6 * n)
	p, err := New(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	ref := naiveSTFT(sig, p.Window(), n, hop, NoPad)
	bins := p.NumBins()
	dst := make([]float32, len(ref)*bins)
	if got := p.PowerInto(dst, sig, NoPad); got != len(ref) {
		t.Fatalf("PowerInto wrote %d frames, want %d", got, len(ref))
	}
	for f := range ref {
		for k := range bins {
			want := cmplx.Abs(ref[f][k])
			want *= want
			got := float64(dst[f*bins+k])
			tol := 1e-2 + 3e-3*want
			if !(math.Abs(got-want) <= tol) { // fail-closed on NaN
				t.Fatalf("power frame %d bin %d: got %g want %g", f, k, got, want)
			}
		}
	}
}

func TestNumFramesSemantics(t *testing.T) {
	p, err := New(Config{FrameSize: 256, HopSize: 64})
	if err != nil {
		t.Fatal(err)
	}
	if got := p.NumFrames(100, NoPad); got != 0 {
		t.Errorf("NoPad short: got %d, want 0", got)
	}
	if got := p.NumFrames(256, NoPad); got != 1 {
		t.Errorf("NoPad exact frame: got %d, want 1", got)
	}
	if got := p.NumFrames(256+3*64, NoPad); got != 4 {
		t.Errorf("NoPad: got %d, want 4", got)
	}
	if got := p.NumFrames(0, PadZero); got != 0 {
		t.Errorf("PadZero empty: got %d, want 0", got)
	}
	if got := p.NumFrames(640, PadZero); got != 1+640/64 {
		t.Errorf("PadZero: got %d, want %d", got, 1+640/64)
	}
}

func TestConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"frame not pow2", Config{FrameSize: 100}},
		{"frame too small", Config{FrameSize: 1}},
		{"hop too large", Config{FrameSize: 256, HopSize: 257}},
		{"hop negative", Config{FrameSize: 256, HopSize: -1}},
		{"bad window", Config{FrameSize: 256, Window: Window(99)}},
		{"custom window wrong len", Config{FrameSize: 256, CustomWindow: make([]float32, 100)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := New(c.cfg); !errors.Is(err, dsp.ErrInvalidConfig) {
				t.Errorf("New(%s) err = %v, want ErrInvalidConfig", c.name, err)
			}
			if _, err := NewAnalyzer(c.cfg); !errors.Is(err, dsp.ErrInvalidConfig) {
				t.Errorf("NewAnalyzer(%s) err = %v, want ErrInvalidConfig", c.name, err)
			}
		})
	}
}

func TestDefaultHop(t *testing.T) {
	p, err := New(Config{FrameSize: 1024})
	if err != nil {
		t.Fatal(err)
	}
	if p.HopSize() != 256 {
		t.Errorf("default hop = %d, want 256 (FrameSize/4)", p.HopSize())
	}
}

func TestCustomWindowIsCopied(t *testing.T) {
	cw := GenerateWindow(Hamming, 64)
	p, err := New(Config{FrameSize: 64, CustomWindow: cw})
	if err != nil {
		t.Fatal(err)
	}
	orig := p.Window()[10]
	cw[10] = 12345 // mutate caller's slice
	if p.Window()[10] != orig {
		t.Errorf("mutating the caller's CustomWindow changed the Plan window")
	}
}

func TestPlanAccessors(t *testing.T) {
	p, err := New(Config{FrameSize: 512, HopSize: 128})
	if err != nil {
		t.Fatal(err)
	}
	if p.FrameSize() != 512 || p.HopSize() != 128 || p.NumBins() != 257 {
		t.Errorf("accessors = %d/%d/%d, want 512/128/257", p.FrameSize(), p.HopSize(), p.NumBins())
	}
	if len(p.Window()) != 512 {
		t.Errorf("Window len = %d, want 512", len(p.Window()))
	}
}
