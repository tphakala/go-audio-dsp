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

func TestSynthesizerZeroAlloc(t *testing.T) {
	const n, hop = 512, 128
	a, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	syn, err := NewSynthesizer(a)
	if err != nil {
		t.Fatal(err)
	}
	in := testSignal(4096)
	out := make([]float32, hop)
	run := func() {
		a.Feed(in, func(spec []complex64, _ []float32) {
			syn.Add(spec)
			syn.Finish(out)
		})
		a.Reset()
		syn.Reset()
	}
	run() // warm up
	if allocs := testing.AllocsPerRun(20, run); allocs != 0 {
		t.Errorf("Synthesizer Add/Finish allocates %v per pass, want 0", allocs)
	}
}

// A warmed EdgeAnalyzer allocates nothing across Reset, Feed and Finish, with the
// cached edge windows (N=256, hop 127, whose drain needs edge windows) and with
// the per-frame rebuild past the cache cap (hop 1 needs more windows than the
// slabs hold).
func TestEdgeAnalyzerZeroAlloc(t *testing.T) {
	cases := []struct {
		name      string
		shape     WindowShape
		n, hop, l int
	}{
		{"hann-cached", WindowShape{Family: ShapeHann}, 256, 127, 24000},
		{"dolph-cached", WindowShape{Family: ShapeDolphChebyshev, AttenuationDB: 106.5}, 256, 127, 24000},
		{"hann-past-cap", WindowShape{Family: ShapeHann}, 128, 1, 400},
		{"dolph-past-cap", WindowShape{Family: ShapeDolphChebyshev, AttenuationDB: 106.5}, 128, 1, 400},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, err := NewEdgeAnalyzer(EdgeConfig{FrameSize: c.n, Shape: c.shape})
			if err != nil {
				t.Fatal(err)
			}
			in := testSignal(c.l)
			var sink float64
			onFrame := func(power []float32) { sink += float64(power[1]) }
			run := func() {
				if err := a.Reset(c.hop); err != nil {
					t.Fatal(err)
				}
				a.Feed(in, onFrame)
				a.Finish(onFrame)
			}
			// Checked warm-up: a regression that emits no frames must fail here
			// rather than pass AllocsPerRun, which ignores emissions.
			emitted := 0
			count := func([]float32) { emitted++ }
			if err := a.Reset(c.hop); err != nil {
				t.Fatal(err)
			}
			a.Feed(in, count)
			a.Finish(count)
			if want := a.NumFrames(int64(c.l)); int64(emitted) != want {
				t.Fatalf("warm-up emitted %d frames, want %d", emitted, want)
			}
			before := a.windowBuilds
			if got := testing.AllocsPerRun(20, run); got != 0 {
				t.Errorf("Reset+Feed+Finish allocated %v times, want 0", got)
			}
			// The cached cases must rebuild nothing; the hop-1 cases exceed the
			// cache cap, so they must rebuild (else they would not test that path).
			rebuilt := a.windowBuilds != before
			if cached := c.hop != 1; rebuilt == cached {
				t.Errorf("warmed repeat runs rebuilt windows = %v, want %v", rebuilt, !cached)
			}
			_ = sink
		})
	}
}
