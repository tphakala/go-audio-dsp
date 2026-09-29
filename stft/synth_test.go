package stft

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// streamThrough analyzes x with leading n-hop zeros and synthesizes every frame
// unmodified. It keeps blocks from index floor((n-hop)/hop), the first that holds
// input, and returns the kept output plus lead, the count of leading-zero samples
// at its start (0 when hop divides n): input sample t is out[lead+t].
func streamThrough(t *testing.T, cfg Config, x []float32) (out []float32, lead int) {
	t.Helper()
	n, hop := cfg.FrameSize, cfg.HopSize
	an, err := NewAnalyzer(cfg)
	if err != nil {
		t.Fatal(err)
	}
	syn, err := NewSynthesizer(an)
	if err != nil {
		t.Fatal(err)
	}
	first := (n - hop) / hop
	lead = (n - hop) - first*hop
	an.Feed(make([]float32, n-hop), func([]complex64, []float32) {})
	padded := append(slices.Clone(x), make([]float32, n)...)
	out = make([]float32, len(padded))
	written, frames := 0, 0
	an.Feed(padded, func(spec []complex64, _ []float32) {
		syn.Add(spec)
		if frames >= first {
			written += syn.Finish(out[written:])
		} else {
			syn.Discard()
		}
		frames++
	})
	return out[:written], lead
}

func TestSynthesizerReconstructsInput(t *testing.T) {
	for _, c := range []Config{
		{FrameSize: 256, HopSize: 64, Window: Hann},
		{FrameSize: 512, HopSize: 128, Window: Hann},
		{FrameSize: 1024, HopSize: 256, Window: Hann},
		{FrameSize: 256, HopSize: 128, Window: Hann},
		{FrameSize: 256, HopSize: 64, Window: Hamming},
		{FrameSize: 256, HopSize: 96, Window: Hann}, // hop does not divide the frame
	} {
		x := testSignal(4000)
		got, lead := streamThrough(t, c, x)
		if len(got) < lead+len(x) {
			t.Fatalf("%+v: %d samples out, want at least %d", c, len(got), lead+len(x))
		}
		var maxErr float64
		for i := range x {
			maxErr = math.Max(maxErr, math.Abs(float64(got[lead+i]-x[i])))
		}
		if maxErr > 1e-4 {
			t.Errorf("%+v: max reconstruction error %g", c, maxErr)
		}
	}
}

func TestNewSynthesizerRejectsVanishingOverlap(t *testing.T) {
	for _, cfg := range []Config{
		{FrameSize: 256, HopSize: 256, Window: Hann},
		{FrameSize: 256, HopSize: 64, Window: Hann, WindowLength: 32},
	} {
		an, err := NewAnalyzer(cfg)
		if err != nil {
			t.Fatalf("%+v: NewAnalyzer: %v", cfg, err)
		}
		if _, err := NewSynthesizer(an); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("%+v: NewSynthesizer err = %v, want ErrInvalidConfig", cfg, err)
		}
	}
}

// TestNewSynthesizerNormFloorBand pins the overlap-sum floor: a flat window whose
// summed square overlap is just above 1e-8 is accepted, one just below is not.
func TestNewSynthesizerNormFloorBand(t *testing.T) {
	const n, hop = 256, 64 // 4 overlapping frames, so the sum is 4*v*v
	for _, c := range []struct {
		v      float32
		accept bool
	}{{1e-3, true}, {1e-5, false}} {
		w := make([]float32, n)
		for i := range w {
			w[i] = c.v
		}
		an, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop, CustomWindow: w})
		if err != nil {
			t.Fatal(err)
		}
		_, err = NewSynthesizer(an)
		if c.accept && err != nil {
			t.Errorf("window %g: rejected: %v", c.v, err)
		}
		if !c.accept && !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("window %g: err = %v, want ErrInvalidConfig", c.v, err)
		}
	}
}

func TestSynthesizerAddLeavesSpecUnchanged(t *testing.T) {
	an, err := NewAnalyzer(Config{FrameSize: 256, HopSize: 64, Window: Hann})
	if err != nil {
		t.Fatal(err)
	}
	syn, err := NewSynthesizer(an)
	if err != nil {
		t.Fatal(err)
	}
	spec := make([]complex64, an.NumBins())
	for k := range spec {
		spec[k] = complex(float32(k), float32(-k))
	}
	want := slices.Clone(spec)
	syn.Add(spec)
	if !slices.Equal(spec, want) {
		t.Error("Add modified spec")
	}
}

func TestSynthesizerFinishClipsAndDiscards(t *testing.T) {
	an, err := NewAnalyzer(Config{FrameSize: 256, HopSize: 64, Window: Hann})
	if err != nil {
		t.Fatal(err)
	}
	syn, err := NewSynthesizer(an)
	if err != nil {
		t.Fatal(err)
	}
	spec := make([]complex64, an.NumBins())
	for k := range spec {
		spec[k] = complex(1, 0)
	}
	syn.Add(spec)
	syn.Discard()
	syn.Add(spec)
	out := make([]float32, 100)
	if got := syn.Finish(out); got != 64 {
		t.Errorf("full block wrote %d, want 64", got)
	}
	syn.Add(spec)
	if got := syn.Finish(out[:10]); got != 10 {
		t.Errorf("clipped block wrote %d, want 10", got)
	}
	// Reset clears the accumulator: a kept block after Reset is silent.
	syn.Add(spec)
	syn.Reset()
	out2 := make([]float32, 64)
	for i := range out2 {
		out2[i] = 1
	}
	syn.Finish(out2)
	for i, v := range out2 {
		if v != 0 {
			t.Fatalf("after Reset out[%d] = %g, want 0", i, v)
		}
	}
}
