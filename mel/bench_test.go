package mel

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// BenchmarkComputeBatClip measures a whole 1 s BSG-BAT clip at 384 kHz through the
// power log-mel path.
func BenchmarkComputeBatClip(b *testing.B) {
	cfg := Config{
		SampleRate: 384000, FrameSize: 1024, HopSize: 768, NumMels: 128,
		MinHz: 9000, MaxHz: 150000, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	}
	ex, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	sig := toneSig(384000, 384000, []float64{12000, 48000, 96000}, 0.3)
	m := Matrix{Data: make([]float32, ex.NumMels()*ex.NumFrames(len(sig), stft.NoPad))}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ex.ComputeInto(&m, sig, stft.NoPad); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStreamFeedVAD measures 1 s of 16 kHz audio through the TEN VAD-shaped
// streaming path (768-sample window centered in a 1024 FFT).
func BenchmarkStreamFeedVAD(b *testing.B) {
	cfg := Config{
		SampleRate: 16000, FrameSize: 1024, WindowLength: 768, WindowAlign: stft.AlignCenter,
		HopSize: 256, NumMels: 40, MinHz: 0, MaxHz: 0, Input: InputPower, Log: LogNatural, LogFloor: 1e-10,
	}
	cs, err := NewColumnSource(cfg)
	if err != nil {
		b.Fatal(err)
	}
	sig := toneSig(16000, 16000, []float64{300, 1200, 3400}, 0.3)
	var sink float64
	onCol := func(col []float32, _ int64) { sink += float64(col[0]) }
	b.ReportAllocs()
	for b.Loop() {
		cs.Feed(sig, onCol)
		cs.Reset()
	}
	_ = sink
}

// BenchmarkProjectSparse measures just the per-frame sparse projection for both
// consumer shapes.
func BenchmarkProjectSparse(b *testing.B) {
	shapes := map[string]Config{
		"bsgbat128": {SampleRate: 384000, FrameSize: 1024, NumMels: 128, MinHz: 9000, MaxHz: 150000, Input: InputPower, Log: Log10, LogOffset: 1e-6},
		"vad40":     {SampleRate: 16000, FrameSize: 1024, NumMels: 40, MinHz: 0, MaxHz: 0, Input: InputPower, Log: LogNatural, LogFloor: 1e-10},
	}
	for name := range shapes {
		cfg := shapes[name]
		b.Run(name, func(b *testing.B) {
			_, pr, err := buildEngine(cfg)
			if err != nil {
				b.Fatal(err)
			}
			numBins := cfg.FrameSize/2 + 1
			power := make([]float32, numBins)
			for k := range power {
				power[k] = float32(k%17) * 0.01
			}
			dst := make([]float32, pr.fb.numMels)
			b.ReportAllocs()
			for b.Loop() {
				pr.apply(dst, power)
			}
		})
	}
}
