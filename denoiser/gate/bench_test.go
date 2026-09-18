package gate

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// BenchmarkProcessInto streams 1 s of 48 kHz audio through a learned-floor gate,
// reporting per-call allocations. The gating benchmark for the SIMD posture: run
// with and without SIMD_DISABLE=all and expect at or below the flagship's
// per-frame cost, with zero steady-state allocations.
func BenchmarkProcessInto(b *testing.B) {
	const sr, frame, hop = 48000, 1024, 256
	p := ParamsFor(denoiser.Medium)
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
	if err != nil {
		b.Fatal(err)
	}
	if err := g.LearnNoise(whiteNoise(frame*4, 0.1, 1)); err != nil {
		b.Fatal(err)
	}
	in := whiteNoise(sr, 0.1, 2)
	out := make([]float32, g.MaxOutputLen(len(in)))
	if _, err := g.ProcessInto(in, out); err != nil { // warm up
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := g.ProcessInto(in, out); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkProcessIntoBlind benchmarks the blind (no learned floor) path, whose
// per-frame rolling-median tracker makes it the more expensive of the two floor
// paths. Kept separate from BenchmarkProcessInto (learned) so the blind cost and
// any regression in it are visible, not hidden behind the cheaper learned path.
func BenchmarkProcessIntoBlind(b *testing.B) {
	const sr, frame, hop = 48000, 1024, 256
	p := ParamsFor(denoiser.Medium)
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
	if err != nil {
		b.Fatal(err)
	}
	in := whiteNoise(sr, 0.1, 2)
	out := make([]float32, g.MaxOutputLen(len(in)))
	for range 3 { // warm past the blind quarter-window before timing
		if _, err := g.ProcessInto(in, out); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := g.ProcessInto(in, out); err != nil {
			b.Fatal(err)
		}
	}
}
