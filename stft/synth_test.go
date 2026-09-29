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
	newSyn := func() (*Synthesizer, []complex64) {
		an, err := NewAnalyzer(Config{FrameSize: 256, HopSize: 64, Window: Hann})
		if err != nil {
			t.Fatal(err)
		}
		syn, err := NewSynthesizer(an)
		if err != nil {
			t.Fatal(err)
		}
		// One cosine bin: its frame is not zero under the Hann window, so every
		// block of the frame carries signal.
		spec := make([]complex64, an.NumBins())
		spec[4] = complex(1, 0)
		return syn, spec
	}
	nonzero := func(x []float32) bool {
		for _, v := range x {
			if v != 0 {
				return true
			}
		}
		return false
	}

	// Reference blocks of one frame: block 0 and block 1.
	ref, spec := newSyn()
	ref.Add(spec)
	b0, b1 := make([]float32, 64), make([]float32, 64)
	ref.Finish(b0)
	ref.Finish(b1)
	if !nonzero(b0) || !nonzero(b1) {
		t.Fatal("test spectrum yields a silent block; the assertions below would be vacuous")
	}

	// Discard drops exactly the oldest block: the next Finish is block 1.
	syn, spec := newSyn()
	syn.Add(spec)
	syn.Discard()
	got := make([]float32, 64)
	syn.Finish(got)
	if !slices.Equal(got, b1) {
		t.Error("after Add and Discard, Finish did not return the second block")
	}

	// Finish clips to len(out) and reports the samples written.
	syn, spec = newSyn()
	syn.Add(spec)
	short := make([]float32, 10)
	if n := syn.Finish(short); n != 10 || !slices.Equal(short, b0[:10]) {
		t.Errorf("clipped Finish wrote %d samples, want 10 matching the block head", n)
	}

	// Reset clears the accumulator: nothing from before it reaches later blocks.
	syn, spec = newSyn()
	syn.Add(spec)
	syn.Reset()
	after := make([]float32, 64)
	after[0] = 1 // a stale destination value must be overwritten too
	if n := syn.Finish(after); n != 64 || nonzero(after) {
		t.Errorf("after Reset, Finish wrote %d samples, nonzero %v; want a silent block", n, nonzero(after))
	}
}
