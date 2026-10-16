//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package spectrogram

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

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
	// Checked warm-up: a regression that writes no columns must fail here rather
	// than sneak through AllocsPerRun, which ignores the result.
	if got, err := s.ComputeInto(&m, sig); err != nil || got != s.NumFrames(len(sig)) {
		t.Fatalf("ComputeInto warm-up = (%d, %v), want (%d, nil)", got, err, s.NumFrames(len(sig)))
	}
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
	// Checked warm-up: cs shares FrameSize/HopSize with s, so it must emit one
	// column per frame. A regression that emits nothing must fail here rather
	// than sneak through AllocsPerRun, which ignores emissions.
	emitted := 0
	cs.Feed(sig, func(col []float32, center int64) { emitted++ })
	cs.Reset()
	if want := s.NumFrames(len(sig)); emitted != want {
		t.Fatalf("Feed warm-up emitted %d columns, want %d", emitted, want)
	}
	if got := testing.AllocsPerRun(20, feed); got != 0 {
		t.Errorf("Feed allocated %v times, want 0", got)
	}
	_ = sink
}

func TestFitZeroAlloc(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	long := make([]float32, 16*soxClipRate) // 16 s: longer than the 15 s span, so columns truncate
	for i := range long {
		long[i] = clip[i%len(clip)]
	}
	cases := []struct {
		name  string
		n, x  int
		shape stft.WindowShape
		sig   []float32
		span  int64
		cols  int
	}{
		// 1 s at sm: the drain runs and the last block is partial.
		{"sm hann", 256, 258, hannShape, clip, 15 * soxClipRate, 18},
		{"sm dolph", 256, 258, dolphShape(100), clip, 15 * soxClipRate, 18},
		// 16 s at lg and a 15 s span: Columns are reached first and the drain is skipped.
		{"lg hann truncated", 1024, 1026, hannShape, long, 15 * soxClipRate, 1026},
		{"lg dolph truncated", 1024, 1026, dolphShape(100), long, 15 * soxClipRate, 1026},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newTestFit(t, soxClipRate, c.n, c.x, c.shape)
			m := Matrix{Data: make([]float32, f.Bins()*c.x)}
			// Checked warm-up: a regression that emits nothing must fail here, not
			// slip through AllocsPerRun, which ignores the result.
			if got, err := f.ComputeInto(&m, c.sig, c.span); err != nil || got != c.cols {
				t.Fatalf("ComputeInto warm-up = (%d, %v), want (%d, nil)", got, err, c.cols)
			}
			if got := testing.AllocsPerRun(10, func() { _, _ = f.ComputeInto(&m, c.sig, c.span) }); got != 0 {
				t.Errorf("ComputeInto allocated %v times, want 0", got)
			}

			var sink float64
			emitted := 0
			stream := func() {
				emitted = 0
				_ = f.Reset(c.span)
				f.Feed(c.sig, func(col []float32, index int) {
					sink += float64(col[0]) + float64(index)
					emitted++
				})
				f.Finish(func(col []float32, index int) {
					sink += float64(col[0]) + float64(index)
					emitted++
				})
			}
			stream()
			if emitted != c.cols {
				t.Fatalf("stream warm-up emitted %d columns, want %d", emitted, c.cols)
			}
			if got := testing.AllocsPerRun(10, stream); got != 0 {
				t.Errorf("Reset+Feed+Finish allocated %v times, want 0", got)
			}
			_ = sink
		})
	}
}
