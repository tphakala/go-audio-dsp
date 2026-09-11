//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package stft

import "testing"

func TestZeroAlloc(t *testing.T) {
	const n, hop = 512, 128

	// Streaming Feed with a local closure that inverse-transforms and reads power,
	// the shape the denoiser uses. This must stay allocation-free per Feed.
	a, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	in := testSignal(4096)
	synth := make([]float32, n)
	var sink float64
	feed := func() {
		a.Feed(in, func(spec []complex64, power []float32) {
			a.Inverse(synth, spec)
			sink += float64(power[0])
		})
		a.Reset()
	}
	feed() // warm up
	if got := testing.AllocsPerRun(30, feed); got != 0 {
		t.Errorf("Feed allocated %v times, want 0", got)
	}
	_ = sink

	// Whole-clip PowerInto.
	p, err := New(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	sig := testSignal(8192)
	dst := make([]float32, p.NumFrames(len(sig), NoPad)*p.NumBins())
	power := func() { p.PowerInto(dst, sig, NoPad) }
	power() // warm up
	if got := testing.AllocsPerRun(30, power); got != 0 {
		t.Errorf("PowerInto allocated %v times, want 0", got)
	}
}
