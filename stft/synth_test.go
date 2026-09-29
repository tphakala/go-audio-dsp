package stft

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// streamThrough analyzes x with leading n-hop zeros and synthesizes every frame
// unmodified, returning the kept output. It mirrors the flagship streaming loop.
func streamThrough(t *testing.T, cfg Config, x []float32) []float32 {
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
	ovl := n / hop
	an.Feed(make([]float32, n-hop), func([]complex64, []float32) {})
	padded := append(slices.Clone(x), make([]float32, n)...)
	out := make([]float32, len(padded))
	written, frames := 0, 0
	an.Feed(padded, func(spec []complex64, _ []float32) {
		syn.Add(spec)
		if frames >= ovl-1 {
			written += syn.Finish(out[written:])
		} else {
			syn.Discard()
		}
		frames++
	})
	return out[:written]
}

func TestSynthesizerReconstructsInput(t *testing.T) {
	for _, c := range []Config{
		{FrameSize: 256, HopSize: 64, Window: Hann},
		{FrameSize: 512, HopSize: 128, Window: Hann},
		{FrameSize: 1024, HopSize: 256, Window: Hann},
		{FrameSize: 256, HopSize: 128, Window: Hann},
		{FrameSize: 256, HopSize: 64, Window: Hamming},
	} {
		x := testSignal(4000)
		got := streamThrough(t, c, x)
		if len(got) < len(x) {
			t.Fatalf("%+v: %d samples out, want at least %d", c, len(got), len(x))
		}
		var maxErr float64
		for i := range x {
			maxErr = math.Max(maxErr, math.Abs(float64(got[i]-x[i])))
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
