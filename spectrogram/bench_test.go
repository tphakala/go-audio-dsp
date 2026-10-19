package spectrogram

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// benchClip benchmarks whole-clip ComputeInto for a fixed config and clip.
func benchClip(b *testing.B, cfg Config, clipSamples int) {
	b.Helper()
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	sig := sine(clipSamples, cfg.SampleRate, 1000, 0.6)
	m := Matrix{Data: make([]float32, s.Bins()*s.NumFrames(len(sig)))}
	// One-time checked call: the measured loop discards its result, so guard here
	// that ComputeInto actually writes columns.
	if got, err := s.ComputeInto(&m, sig); err != nil || got == 0 {
		b.Fatalf("ComputeInto = (%d, %v)", got, err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = s.ComputeInto(&m, sig)
	}
}

// BenchmarkComputeBirdClip: 3 s at 48 kHz, 1024-point DFT, dB scale, the clip
// spectrogram shape.
func BenchmarkComputeBirdClip(b *testing.B) {
	benchClip(b, Config{SampleRate: 48000, FrameSize: 1024, HopSize: 256, Scale: DB, GainDB: 3, DynamicRangeDB: 100}, 48000*3)
}

// BenchmarkComputeBatClip: a shorter clip at a larger DFT, standing in for the
// ultrasonic profile (resampling itself is out of scope for this package).
func BenchmarkComputeBatClip(b *testing.B) {
	benchClip(b, Config{SampleRate: 96000, FrameSize: 4096, HopSize: 1024, Scale: DB, GainDB: 3, DynamicRangeDB: 100}, 96000)
}

// BenchmarkStreamFeed: streaming one 1 s clip through the column source.
func BenchmarkStreamFeed(b *testing.B) {
	cs, err := NewColumnSource(Config{SampleRate: 48000, FrameSize: 1024, HopSize: 256, Scale: DB, GainDB: 3, DynamicRangeDB: 100})
	if err != nil {
		b.Fatal(err)
	}
	sig := sine(48000, 48000, 1000, 0.6)
	// One-time emission check: the measured loop discards emissions, so guard
	// here that the stream actually emits columns.
	emitted := 0
	cs.Feed(sig, func(col []float32, center int64) { emitted++ })
	cs.Reset()
	if emitted == 0 {
		b.Fatalf("stream emitted %d columns, want > 0", emitted)
	}
	var sink float64
	b.ReportAllocs()
	for b.Loop() {
		cs.Feed(sig, func(col []float32, center int64) { sink += float64(col[0]) })
		cs.Reset()
	}
	_ = sink
}

// BenchmarkFit: a warmed Fit.ComputeInto over a 15 s clip at 24 kHz with a 15 s
// span, the BirdNET-Go clip shape, at the lg (1024) and xl (2048) frame sizes
// with Hann and Dolph-Chebyshev windows.
func BenchmarkFit(b *testing.B) {
	const rate, samples = 24000, 15 * 24000
	sig := sine(samples, rate, 3000, 0.5)
	for _, g := range []struct {
		name string
		n    int
	}{{"lg", 1024}, {"xl", 2048}} {
		for _, w := range []struct {
			name  string
			shape stft.WindowShape
		}{{hannName, hannShape}, {"dolph", dolphShape(100)}} {
			b.Run(g.name+"/"+w.name, func(b *testing.B) {
				f, err := NewFit(FitConfig{SampleRate: rate, FrameSize: g.n, Columns: g.n + 2, Window: w.shape})
				if err != nil {
					b.Fatal(err)
				}
				m := Matrix{Data: make([]float32, f.Bins()*(g.n+2))}
				// One-time checked call: the loop discards its result.
				if got, err := f.ComputeInto(&m, sig, samples); err != nil || got == 0 {
					b.Fatalf("ComputeInto = (%d, %v)", got, err)
				}
				b.ReportAllocs()
				for b.Loop() {
					_, _ = f.ComputeInto(&m, sig, samples)
				}
			})
		}
	}
}

// BenchmarkFitFresh: a Fit built from scratch for every render, the way a
// request handler uses it, over a 48 kHz clip with the span equal to the clip
// length. Unlike BenchmarkFit it constructs the Fit inside the loop, so it
// measures what a later render of a configuration costs once the process-wide
// edge window cache holds its windows (the checked call before the loop fills
// the cache). The first-use cost of a configuration is the "cold" mode of
// stft's BenchmarkEdgeAnalyzer. The n2048-15s pair is the render of a clip drawn
// up to half of its band, without and with a MaxHz cap at that frequency.
func BenchmarkFitFresh(b *testing.B) {
	const rate = 48000
	shape := dolphShape(100)
	for _, g := range []struct {
		name    string
		n       int
		seconds int
		columns int
		maxHz   float64
	}{
		{"n4096-3s", 4096, 3, 2050, 0},
		{"n8192-15s", 8192, 15, 4098, 0},
		{"n2048-15s", 2048, 15, 1024, 0},
		{"n2048-15s-max12k", 2048, 15, 1024, 12000},
	} {
		b.Run(g.name, func(b *testing.B) {
			samples := g.seconds * rate
			sig := sine(samples, rate, 3000, 0.5)
			cfg := FitConfig{SampleRate: rate, FrameSize: g.n, Columns: g.columns, Window: shape, MaxHz: g.maxHz}
			f, err := NewFit(cfg)
			if err != nil {
				b.Fatal(err)
			}
			m := Matrix{Data: make([]float32, f.Bins()*g.columns)}
			// One-time checked call: the loop discards its result.
			if got, err := f.ComputeInto(&m, sig, int64(samples)); err != nil || got == 0 {
				b.Fatalf("ComputeInto = (%d, %v)", got, err)
			}
			b.ReportAllocs()
			for b.Loop() {
				f, err := NewFit(cfg)
				if err != nil {
					b.Fatal(err)
				}
				_, _ = f.ComputeInto(&m, sig, int64(samples))
			}
		})
	}
}
