package denoiser

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/internal/audiotest"
	"github.com/tphakala/go-audio-dsp/pcm"
)

// benchClip is a 10 s 48 kHz mono clip (noise at -40 dBFS with signal bursts),
// deterministic across runs.
func benchClip() audiotest.SynthClip { return audiotest.MakeSynthClip(48000, -40, -20, false, 1) }

// BenchmarkProcessIntoSteadyState48k measures the zero-allocation streaming
// path: a warmed-up Denoiser processing fixed 100 ms chunks into a caller-owned
// output buffer. In steady state it must not allocate (see allocs/op).
func BenchmarkProcessIntoSteadyState48k(b *testing.B) {
	clip := benchClip()
	d, err := New(Config{SampleRate: clip.SR})
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4800 // 100 ms at 48 kHz
	in := clip.Mix[:chunk]
	out := make([]float32, chunk+d.n) // len(in)+HopSize suffices; +FrameSize is ample
	if _, err := d.ProcessInto(in, out); err != nil {
		b.Fatal(err) // warm past the initial latency
	}
	b.SetBytes(int64(chunk * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := d.ProcessInto(in, out); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkProcess48k measures the allocating convenience path (Process returns
// a fresh slice each call): the ProcessInto cost plus the per-call output
// allocation.
func BenchmarkProcess48k(b *testing.B) {
	clip := benchClip()
	d, err := New(Config{SampleRate: clip.SR})
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4800
	in := clip.Mix[:chunk]
	if _, err := d.Process(in); err != nil {
		b.Fatal(err) // warm past the initial latency
	}
	b.SetBytes(int64(chunk * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := d.Process(in); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDenoiseAuto48k measures the offline whole-clip path with an
// automatically measured quiet-window noise profile: the realistic BirdNET-Go
// clip-export cost (profiling, streaming, and flush over the whole clip).
func BenchmarkDenoiseAuto48k(b *testing.B) {
	clip := benchClip()
	cfg := Config{SampleRate: clip.SR}
	b.SetBytes(int64(len(clip.Mix) * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Denoise(clip.Mix, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDenoiseWithProfile48k measures the offline path with a profile
// measured once and reused (for example one profile per station), so the
// per-clip cost excludes noise estimation.
func BenchmarkDenoiseWithProfile48k(b *testing.B) {
	clip := benchClip()
	cfg := Config{SampleRate: clip.SR}
	d, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	prof, err := d.NoiseProfileFromSamples(clip.Noise)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(clip.Mix) * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DenoiseWithProfile(clip.Mix, prof, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// benchLongClip is a 300 s 48 kHz mono clip (the benchClip pattern tiled), built
// once per benchmark outside the timed loop.
func benchLongClip() []float32 {
	base := benchClip().Mix
	out := make([]float32, 0, 300*48000)
	for len(out) < 300*48000 {
		out = append(out, base...)
	}
	return out[:300*48000]
}

// BenchmarkEstimateNoiseProfile48k measures quiet-window noise estimation over
// a whole 10 s clip held as float32.
func BenchmarkEstimateNoiseProfile48k(b *testing.B) {
	clip := benchClip()
	d, err := New(Config{SampleRate: clip.SR})
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(clip.Mix) * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := d.EstimateNoiseProfile(clip.Mix); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEstimateNoiseProfile300s48k is the whole-buffer estimate over a
// 300 s clip, the size where the float32 copy dominates memory.
func BenchmarkEstimateNoiseProfile300s48k(b *testing.B) {
	x := benchLongClip()
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := d.EstimateNoiseProfile(x); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNoiseProfileEstimatorFloat32Chunks48k feeds the benchClip to a
// NoiseProfileEstimator in 100 ms float32 chunks.
func BenchmarkNoiseProfileEstimatorFloat32Chunks48k(b *testing.B) {
	clip := benchClip()
	d, err := New(Config{SampleRate: clip.SR})
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4800
	e := d.NewNoiseProfileEstimator()
	b.SetBytes(int64(len(clip.Mix) * 4))
	b.ReportAllocs()
	for b.Loop() {
		e.Reset()
		for x := clip.Mix; len(x) > 0; x = x[min(chunk, len(x)):] {
			e.AddFloat32(x[:min(chunk, len(x))])
		}
		if _, err := e.Profile(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNoiseProfileEstimatorInt16Chunks48k is the int16-resident flow: the
// clip is converted to int16 once, outside the loop, and fed in 100 ms chunks.
func BenchmarkNoiseProfileEstimatorInt16Chunks48k(b *testing.B) {
	clip := benchClip()
	pcmClip := make([]int16, len(clip.Mix))
	pcm.Float32ToInt16(pcmClip, clip.Mix)
	d, err := New(Config{SampleRate: clip.SR})
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4800
	e := d.NewNoiseProfileEstimator()
	b.SetBytes(int64(len(pcmClip) * 2))
	b.ReportAllocs()
	for b.Loop() {
		e.Reset()
		for x := pcmClip; len(x) > 0; x = x[min(chunk, len(x)):] {
			e.AddInt16(x[:min(chunk, len(x))])
		}
		if _, err := e.Profile(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNoiseProfileEstimator300s48kFloat32Chunks is the 300 s clip fed in
// 100 ms chunks; compare allocs and B/op with BenchmarkEstimateNoiseProfile300s48k.
// A fresh estimator per iteration includes its fixed buffers in B/op.
func BenchmarkNoiseProfileEstimator300s48kFloat32Chunks(b *testing.B) {
	x := benchLongClip()
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		b.Fatal(err)
	}
	const chunk = 4800
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	for b.Loop() {
		e := d.NewNoiseProfileEstimator()
		for rest := x; len(rest) > 0; rest = rest[min(chunk, len(rest)):] {
			e.AddFloat32(rest[:min(chunk, len(rest))])
		}
		if _, err := e.Profile(); err != nil {
			b.Fatal(err)
		}
	}
}
