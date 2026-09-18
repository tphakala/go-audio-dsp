//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package mel

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

func TestZeroAlloc(t *testing.T) {
	cfg := Config{
		SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40,
		MinHz: 500, MaxHz: 20000, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	}
	sig := toneSig(48000, 48000, []float64{1200, 4300}, 0.4) // 1 s

	ex, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Whole-clip ComputeInto into a pre-sized, reused Matrix, both NoPad and the
	// centered reflect path (which also fills the pad scratch and feeds three
	// chunks). The centered matrix has one extra frame, so size for the max.
	nFrames := ex.NumFrames(len(sig), stft.PadReflect)
	m := Matrix{Data: make([]float32, ex.NumMels()*nFrames)}
	for _, pad := range []stft.PadMode{stft.NoPad, stft.PadZero, stft.PadReflect} {
		want := ex.NumFrames(len(sig), pad)
		// Checked warm-up: a regression that writes no columns must fail here rather
		// than sneak through AllocsPerRun, which ignores the result.
		if got, err := ex.ComputeInto(&m, sig, pad); err != nil || got != want {
			t.Fatalf("ComputeInto(%v) warm-up = (%d, %v), want (%d, nil)", pad, got, err, want)
		}
		compute := func() { _, _ = ex.ComputeInto(&m, sig, pad) }
		if got := testing.AllocsPerRun(20, compute); got != 0 {
			t.Errorf("ComputeInto(%v) allocated %v times, want 0", pad, got)
		}
	}

	// Streaming Feed with a non-allocating callback, the live-stream shape.
	cs, err := NewColumnSource(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var sink float64
	feed := func() {
		cs.Feed(sig, func(col []float32, center int64) {
			sink += float64(col[0]) + float64(center)
		})
		cs.Reset()
	}
	// Checked warm-up: cs shares FrameSize/HopSize with ex, so it must emit one
	// column per NoPad frame.
	emitted := 0
	cs.Feed(sig, func(col []float32, center int64) { emitted++ })
	cs.Reset()
	if want := ex.NumFrames(len(sig), stft.NoPad); emitted != want {
		t.Fatalf("Feed warm-up emitted %d columns, want %d", emitted, want)
	}
	if got := testing.AllocsPerRun(20, feed); got != 0 {
		t.Errorf("Feed allocated %v times, want 0", got)
	}
	_ = sink
}
