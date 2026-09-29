package denoiser

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/internal/audiotest"
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
