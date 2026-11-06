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

// BenchmarkStreamFeedVAD measures 1 s of 16 kHz audio through a VAD-like streaming
// shape (a 768-sample window centered in a 1024 FFT). The exact TEN VAD front-end
// parameters are not yet pinned (see issue #32); this is a representative
// small-window streaming configuration, not a claim about that model.
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
		// The -mag variants exercise the InputMagnitude sqrt path that the power
		// variants above skip: bsgbat128-mag covers a sub-band, so the sqrt runs over
		// only part of the spectrum, while vad40-mag spans the full band.
		"bsgbat128-mag": {SampleRate: 384000, FrameSize: 1024, NumMels: 128, MinHz: 9000, MaxHz: 150000, Input: InputMagnitude, Log: Log10, LogOffset: 1e-6},
		"vad40-mag":     {SampleRate: 16000, FrameSize: 1024, NumMels: 40, MinHz: 0, MaxHz: 0, Input: InputMagnitude, Log: LogNatural, LogFloor: 1e-10},
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
				pr.fromPower(dst, power)
			}
		})
	}
	b.Run("real-n1024", func(b *testing.B) {
		cfg := Config{SampleRate: 48000, FrameSize: 1024, NumMels: 96, MinHz: 500, MaxHz: 15000, Input: InputReal, Log: PowerLaw, PowerExponent: 0.4}
		_, pr, err := buildEngine(cfg)
		if err != nil {
			b.Fatal(err)
		}
		spec := make([]complex64, cfg.FrameSize/2+1)
		for k := range spec {
			spec[k] = complex(float32(k%17)*0.01-0.08, float32(k%13)*0.01)
		}
		dst := make([]float32, pr.fb.numMels)
		b.ReportAllocs()
		for b.Loop() {
			pr.fromSpectrum(dst, spec)
		}
	})
}

// birdLikeBank builds a synthetic bank of mels rows of narrow triangles spread over
// the bins [lo, hi) of a numBins-bin spectrum, widths growing from 1 toward maxW.
func birdLikeBank(b *testing.B, mels, numBins, lo, hi, maxW int) *Filterbank {
	b.Helper()
	rows := make([][]float32, mels)
	for m := range rows {
		rows[m] = make([]float32, numBins)
		start := lo + m*(hi-lo-maxW)/mels
		w := 1 + m*(maxW-1)/mels
		for k := start; k < start+w && k < hi; k++ {
			rows[m][k] = 1 / float32(w)
		}
	}
	fb, err := FilterbankFromRows(rows)
	if err != nil {
		b.Fatal(err)
	}
	return fb
}

// BenchmarkComputeRealPowerLaw measures a whole 3 s, 48 kHz clip through the
// signed real-part, power-law path for two synthetic bank shapes: a 2048-point
// transform with rows over bins [1, 128) and a 1024-point one over [11, 320).
func BenchmarkComputeRealPowerLaw(b *testing.B) {
	shapes := []struct {
		name                 string
		n, hop, lo, hi, maxW int
	}{
		{"n2048h278", 2048, 278, 1, 128, 5},
		{"n1024h280", 1024, 280, 11, 320, 17},
	}
	sig := toneSig(3*48000, 48000, []float64{1200, 4300}, 0.4)
	for _, sh := range shapes {
		b.Run(sh.name, func(b *testing.B) {
			ex, err := New(Config{
				SampleRate: 48000, FrameSize: sh.n, HopSize: sh.hop,
				Filterbank: birdLikeBank(b, 96, sh.n/2+1, sh.lo, sh.hi, sh.maxW),
				Input:      InputReal, Log: PowerLaw, PowerExponent: 0.4,
			})
			if err != nil {
				b.Fatal(err)
			}
			m := Matrix{Data: make([]float32, ex.NumMels()*ex.NumFrames(len(sig), stft.NoPad))}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := ex.ComputeInto(&m, sig, stft.NoPad); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
