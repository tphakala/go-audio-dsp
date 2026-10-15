package loudnorm

import (
	"math"
	"math/rand/v2"
	"slices"
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

// disableTruePeakSkip makes batch convolve every chunk until the test ends.
func disableTruePeakSkip(t *testing.T) {
	t.Helper()
	saved := tpGain
	tpGain = float32(math.Inf(1))
	t.Cleanup(func() { tpGain = saved })
}

// TestTruePeakSkipIsExact pins that skipping chunks the filter-gain bound rules
// out leaves the measured peak bit-identical to convolving every chunk.
func TestTruePeakSkipIsExact(t *testing.T) {
	type result struct{ peak, loudness float64 }
	measure := func(pcm []int16, ch int) result {
		m := NewMeter(48000, ch)
		m.AddInt16(pcm)
		return result{m.TruePeakDBTP(), m.IntegratedLoudness()}
	}
	type key struct {
		ch   int
		seed uint64
	}
	keys := make([]key, 0, 8)
	beds := map[key][]int16{}
	skipped := map[key]result{}
	for _, ch := range []int{1, 2} {
		for seed := uint64(1); seed <= 4; seed++ {
			k := key{ch, seed}
			keys = append(keys, k)
			beds[k] = burstyBed(150000, ch, seed)
			skipped[k] = measure(beds[k], ch)
		}
	}
	disableTruePeakSkip(t)
	for _, k := range keys {
		if got := measure(beds[k], k.ch); got != skipped[k] {
			t.Errorf("ch %d seed %d: with skip %+v, without %+v", k.ch, k.seed, skipped[k], got)
		}
	}
}

// TestTruePeakSkipKeepsWorstCaseOvershoot feeds the case the skip test must not
// skip: after a loud sample, a chunk whose own sample peak stays below it but
// whose samples follow the signs of the interpolation kernel, so the oversampled
// output reaches 2% above the earlier peak at the filter's full L1 gain. The
// peak must equal the unskipped measurement, which exceeds the earlier sample
// peak. A bound more than about 2% below the true L1 gain skips the chunk and
// misses the overshoot; a looser bound is not checked here (it only costs speed).
func TestTruePeakSkipKeepsWorstCaseOvershoot(t *testing.T) {
	phase, l1 := 0, 0.0
	for p := range oversample {
		var sum float64
		for k := range tapsPerPhase {
			sum += math.Abs(tpCoef[p][k])
		}
		if sum > l1 {
			phase, l1 = p, sum
		}
	}
	const spike = 24000
	if l1 <= 1.02 {
		t.Fatalf("kernel L1 gain %v leaves no room under the spike for this fixture", l1)
	}
	amp := spike / l1 * 1.02 // below the spike; amp*l1 is 2% above it
	pcm := make([]int16, 3*tpChunk)
	pcm[100] = spike
	for k := range tapsPerPhase { // the second chunk
		v := int16(math.Round(amp))
		if tpCoef[phase][k] < 0 {
			v = -v
		}
		pcm[tpChunk+1000-k] = v
	}
	measure := func() float64 {
		m := NewMeter(48000, 1)
		m.AddInt16(pcm)
		return m.TruePeakDBTP()
	}
	skipped := measure()
	disableTruePeakSkip(t)
	full := measure()
	if spikeDB := 20 * math.Log10(spike/32768.0); full < spikeDB+0.1 {
		t.Fatalf("fixture does not overshoot: unskipped true peak %.2f dBTP, earlier sample peak %.2f dBFS", full, spikeDB)
	}
	if skipped != full {
		t.Errorf("with skip %v dBTP, without %v dBTP: the chunk was skipped but raises the peak", skipped, full)
	}
}

// TestMeterChannelsMatchMono pins the channel indexing of every feed entry
// point against an independent oracle: with a different signal in each channel,
// a channel's sub-block energies and true peak equal those of a mono meter fed
// only that channel. The channel counts include odd ones (3 and 5).
func TestMeterChannelsMatchMono(t *testing.T) {
	for _, ch := range []int{2, 3, 5, 6} {
		const frames = 60000
		per := make([][]int16, ch)
		il := make([]int16, frames*ch)
		for c := range ch {
			per[c] = burstyBed(frames, 1, uint64(10+c))
			for i := range frames {
				il[i*ch+c] = per[c][i]
			}
		}
		feeds := []struct {
			name string
			add  func(m *Meter, s []int16)
		}{
			{"int16", func(m *Meter, s []int16) { m.AddInt16(s) }},
			{"float32", func(m *Meter, s []int16) {
				f := make([]float32, len(s))
				for i, v := range s {
					f[i] = float32(v) / 32768
				}
				m.AddFloat32(f)
			}},
			{"float64", func(m *Meter, s []int16) {
				f := make([]float64, len(s))
				for i, v := range s {
					f[i] = float64(v) / 32768
				}
				m.AddFloat64(f)
			}},
		}
		for _, feed := range feeds {
			multi := NewMeter(48000, ch)
			feed.add(multi, il)
			for c := range ch {
				mono := NewMeter(48000, 1)
				feed.add(mono, per[c])
				if !slices.Equal(multi.subEnergy[c], mono.subEnergy[0]) {
					t.Errorf("%s ch %d: channel %d sub-block energies differ from a mono meter", feed.name, ch, c)
				}
				if multi.tp[c].maxAbs != mono.tp[0].maxAbs {
					t.Errorf("%s ch %d: channel %d true peak %v, mono %v", feed.name, ch, c, multi.tp[c].maxAbs, mono.tp[0].maxAbs)
				}
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
