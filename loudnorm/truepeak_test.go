package loudnorm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The true peak of a pure sine equals its amplitude. A -6 dBFS, 1 kHz sine must
// therefore measure about -6.0 dBTP after oversampling.
func TestTruePeakSineAmplitude(t *testing.T) {
	m := NewMeter(48000, 1)
	m.AddFloat64(sineInterleaved(-6, 1000, 1, 48000, 1))
	got := m.TruePeakDBTP()
	if math.Abs(got-(-6.0)) > 0.2 {
		t.Errorf("true peak = %.3f dBTP, want -6.0 +/-0.2", got)
	}
}

// Silence has no peak.
func TestTruePeakSilence(t *testing.T) {
	m := NewMeter(48000, 1)
	m.AddFloat64(make([]float64, 48000))
	got := m.TruePeakDBTP()
	if !math.IsInf(got, -1) {
		t.Errorf("true peak = %.3f dBTP, want -Inf for silence", got)
	}
}

// TruePeakDBTP must be non-destructive: the group-delay drain runs on a copy of
// the filter state, so repeated calls return the same value.
func TestTruePeakIdempotent(t *testing.T) {
	m := NewMeter(48000, 1)
	m.AddFloat64(sineInterleaved(-6, 4000, 0.5, 48000, 1))
	first := m.TruePeakDBTP()
	second := m.TruePeakDBTP()
	if first != second {
		t.Errorf("TruePeakDBTP not idempotent: %.6f then %.6f", first, second)
	}
}

// Oversampled true peak must never under-report the raw sample peak.
func TestTruePeakAtLeastSamplePeak(t *testing.T) {
	pcm := sineInterleaved(-3, 5000, 0.5, 48000, 1)
	var samplePeak float64
	for _, s := range pcm {
		// The meter processes float32, so compare against the float32 sample peak.
		if a := math.Abs(float64(float32(s))); a > samplePeak {
			samplePeak = a
		}
	}
	m := NewMeter(48000, 1)
	m.AddFloat64(pcm)
	tp := math.Pow(10, m.TruePeakDBTP()/20)
	if tp < samplePeak-1e-9 {
		t.Errorf("true peak %.6f below sample peak %.6f", tp, samplePeak)
	}
}

// burstyBed is a quiet noise bed with loud bursts and near-Nyquist content, the
// shapes where the chunk skip fires and where an inter-sample overshoot could
// try to slip past it.
func burstyBed(n, channels int, seed uint64) []int16 {
	r := rand.New(rand.NewPCG(seed, seed+1))
	s := make([]int16, n*channels)
	for i := range n {
		amp := 400.0
		if (i/20000)%5 == 2 {
			amp = 20000
		}
		v := amp * (r.Float64()*2 - 1)
		if (i/7000)%3 == 1 { // alternating-sign run: peaks between samples
			v = amp * math.Pow(-1, float64(i)) * 0.7
		}
		for c := range channels {
			s[i*channels+c] = int16(v)
		}
	}
	return s
}

// TestTruePeakSkipIsExact pins that skipping chunks the filter-gain bound rules
// out leaves the measured peak bit-identical to convolving every chunk.
func TestTruePeakSkipIsExact(t *testing.T) {
	for _, ch := range []int{1, 2} {
		for seed := uint64(1); seed <= 4; seed++ {
			pcm := burstyBed(150000, ch, seed)
			skip := NewMeter(48000, ch)
			skip.AddInt16(pcm)

			saved := tpGain
			tpGain = float32(math.Inf(1)) // never skip
			full := NewMeter(48000, ch)
			full.AddInt16(pcm)
			tpGain = saved

			if a, b := skip.TruePeakDBTP(), full.TruePeakDBTP(); a != b {
				t.Errorf("ch %d seed %d: with skip %v, without %v", ch, seed, a, b)
			}
			if a, b := skip.IntegratedLoudness(), full.IntegratedLoudness(); a != b {
				t.Errorf("ch %d seed %d: loudness %v vs %v", ch, seed, a, b)
			}
		}
	}
}

// TestMeterChunkingIsExact pins that feeding in pieces that straddle sub-block
// boundaries gives the same measurement as one call.
func TestMeterChunkingIsExact(t *testing.T) {
	for _, ch := range []int{1, 2, 6} {
		pcm := burstyBed(100000, ch, 9)
		whole := NewMeter(48000, ch)
		whole.AddInt16(pcm)
		parts := NewMeter(48000, ch)
		for off := 0; off < len(pcm); {
			n := min(ch*(1+(off/ch)%4999), len(pcm)-off)
			parts.AddInt16(pcm[off : off+n])
			off += n
		}
		if a, b := whole.IntegratedLoudness(), parts.IntegratedLoudness(); a != b {
			t.Errorf("ch %d: loudness %v vs %v", ch, a, b)
		}
		if a, b := whole.TruePeakDBTP(), parts.TruePeakDBTP(); a != b {
			t.Errorf("ch %d: true peak %v vs %v", ch, a, b)
		}
	}
}
