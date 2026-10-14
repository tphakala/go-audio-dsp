//go:build !race

package gate

import (
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestZeroAllocSteadyState pins that the hot path allocates nothing per call once
// warmed, on the blind, learned and capped floors. Construction and LearnNoise may
// allocate; ProcessInto and FlushInto may not. Any per-frame make turns it red.
// (Excluded under -race, which adds its own bookkeeping allocations.)
func TestZeroAllocSteadyState(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	for _, mode := range []string{modeBlind, modeLearned, modeCapped} {
		p := ParamsFor(denoiser.Medium)
		g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
		if err != nil {
			t.Fatal(err)
		}
		switch mode {
		case modeLearned:
			if err := g.LearnNoise(whiteNoise(frame*4, 0.1, 1)); err != nil {
				t.Fatal(err)
			}
		case modeCapped:
			if err := g.setBlindCap(slices.Repeat([]float32{1e-3}, g.bins)); err != nil {
				t.Fatal(err)
			}
		}
		in := whiteNoise(4*hop, 0.1, 2)
		out := make([]float32, g.MaxOutputLen(len(in)))
		for range 2000 { // warm past the blind warm-up and fill the pipeline
			if _, err := g.ProcessInto(in, out); err != nil {
				t.Fatal(err)
			}
		}
		if a := testing.AllocsPerRun(200, func() {
			if _, err := g.ProcessInto(in, out); err != nil {
				t.Fatal(err)
			}
		}); a != 0 {
			t.Errorf("%s: ProcessInto allocated %.1f objects/op, want 0", mode, a)
		}

		fb := make([]float32, g.Latency()+g.HopSize())
		if a := testing.AllocsPerRun(200, func() {
			if _, err := g.ProcessInto(in, out); err != nil {
				t.Fatal(err)
			}
			if _, err := g.FlushInto(fb); err != nil {
				t.Fatal(err)
			}
		}); a != 0 {
			t.Errorf("%s: ProcessInto+FlushInto allocated %.1f objects/op, want 0", mode, a)
		}
	}
}
