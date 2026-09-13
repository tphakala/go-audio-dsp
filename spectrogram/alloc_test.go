//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package spectrogram

import "testing"

func TestZeroAlloc(t *testing.T) {
	const sr, n, hop = 48000, 1024, 256
	sig := sine(48000, sr, 1200, 0.6) // 1 s

	// Whole-clip ComputeInto into a pre-sized, reused Matrix.
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: DB, GainDB: 3, DynamicRangeDB: 100})
	if err != nil {
		t.Fatal(err)
	}
	m := Matrix{Data: make([]float32, s.Bins()*s.NumFrames(len(sig)))}
	compute := func() { _, _ = s.ComputeInto(&m, sig) }
	compute() // warm up
	if got := testing.AllocsPerRun(20, compute); got != 0 {
		t.Errorf("ComputeInto allocated %v times, want 0", got)
	}

	// Streaming Feed with a non-allocating callback, the live-waterfall shape.
	cs, err := NewColumnSource(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: DB, GainDB: 3, DynamicRangeDB: 100})
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
	feed() // warm up
	if got := testing.AllocsPerRun(20, feed); got != 0 {
		t.Errorf("Feed allocated %v times, want 0", got)
	}
	_ = sink
}
