package gate

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
)

// TestGateAttenuatesNoiseKeepsTone pins the core gate mechanism: with a learned
// floor it reduces the noise-only regions by at least MaxAttenuationDB-4 while
// leaving a loud tone essentially untouched (within 1 dB), at the tuned Medium
// and Heavy params with frequency smoothing on. A broken sigmoid or power ratio
// fails one side or the other.
//
// The tone is 3 kHz at FrameSize 1024 and 48 kHz, which falls exactly on bin 64.
// That is the worst case for smoothing the gain across bins: the Hann-windowed
// tone leaves bins 62 and 66 with no tone energy, so a gain box mean pulls the
// open bins 63 to 65 down toward the closed neighbours (about 3 and 4 dB at
// Medium and Heavy). Smoothing the per-bin SNR instead keeps the tone. A tone
// half a bin off the grid loses well under 0.5 dB even with gain smoothing, so
// the on-bin frequency is what makes this test able to fail.
func TestGateAttenuatesNoiseKeepsTone(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	dur := 3 * sr
	noise := whiteNoise(dur, dbToLin(-40), 40)
	tn := tone(dur, sr, 3000, dbToLin(-20))
	clip := make([]float32, dur)
	for i := range clip {
		clip[i] = noise[i]
		if i >= sr && i < 2*sr { // tone present only in [1s, 2s]
			clip[i] += tn[i]
		}
	}
	noiseSpans := [][2]int{{sr / 4, 3 * sr / 4}, {2*sr + sr/4, 3*sr - sr/4}}
	toneSpans := [][2]int{{sr + sr/4, 2*sr - sr/4}} // interior of the tone region, clear of edges

	for _, strength := range []denoiser.Strength{denoiser.Medium, denoiser.Heavy} {
		t.Run(strength.String(), func(t *testing.T) {
			p := ParamsFor(strength)
			out, err := DenoiseWithNoise(clip, noise[:sr/2], Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != len(clip) {
				t.Fatalf("length %d != %d", len(out), len(clip))
			}
			// Guard before the dB metrics: spanRMSDB propagates NaN, and reduction < want
			// is false on a NaN operand, so a NaN-emitting regression would pass vacuously.
			assertFinite(t, "out", out)
			reduction := spanRMSDB(clip, noiseSpans) - spanRMSDB(out, noiseSpans)
			toneChange := spanRMSDB(out, toneSpans) - spanRMSDB(clip, toneSpans)

			t.Logf("noise reduction %.1f dB (floor %g dB), tone-region level change %.2f dB", reduction, p.MaxAttenuationDB, toneChange)
			if want := float64(p.MaxAttenuationDB) - 4; reduction < want {
				t.Errorf("noise reduction %.1f dB < %.1f (MaxAttenuationDB-4)", reduction, want)
			}
			if math.Abs(toneChange) > 1.0 {
				t.Errorf("tone-region level changed by %.2f dB (>1 dB); the gate is eating the tone", toneChange)
			}
		})
	}
}

// TestFreqSmoothingEngages pins the frequency-smoothing pass: with R>0 a spike in
// the per-bin SNR (the vector the pipeline smooths) spreads to its neighbours and
// the centre falls; with R==0 the pass is an exact identity. A broken shifted Add
// leaves neighbours unmoved. A second case drives computeMask itself: a 30 dB SNR
// spike over a 0 dB floor opens its +-1 neighbours with R=2 and leaves them near
// the 0 dB gain (just above gFloor) with R=0, which pins that the smoothing runs before the sigmoid.
func TestFreqSmoothingEngages(t *testing.T) {
	mkGate := func(fsb int) *Gate {
		p := ParamsFor(denoiser.Medium)
		p.FreqSmoothBins = fsb
		g, err := New(Config{SampleRate: 48000, FrameSize: 256, HopSize: 64, Params: &p})
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	spike := func(g *Gate) (dst, src []float32) {
		src = make([]float32, g.bins)
		for i := range src {
			src[i] = 0.2
		}
		src[g.bins/2] = 1.0
		return make([]float32, g.bins), src
	}

	g0 := mkGate(1) // R = 0: disabled
	d0, s0 := spike(g0)
	g0.freqSmooth(d0, s0)
	for k := range s0 {
		if d0[k] != s0[k] {
			t.Fatalf("FreqSmoothBins=1 (disabled): freqSmooth changed bin %d (%g -> %g)", k, s0[k], d0[k])
		}
	}

	g2 := mkGate(5) // R = 2
	d2, s2 := spike(g2)
	g2.freqSmooth(d2, s2)
	c := g2.bins / 2
	if !(d2[c-1] > s2[c-1] && d2[c+1] > s2[c+1]) {
		t.Errorf("FreqSmoothBins=5: the spike did not raise its neighbours (bin %d neighbours %g->%g, %g->%g)", c, s2[c-1], d2[c-1], s2[c+1], d2[c+1])
	}
	if d2[c] >= s2[c] {
		t.Errorf("FreqSmoothBins=5: the spike centre did not fall (%g -> %g)", s2[c], d2[c])
	}

	// Pipeline level: the smoothed quantity is the SNR, so the spike's neighbours open.
	neighbourGain := func(g *Gate) (lo, hi float32) {
		flat := slices.Repeat([]float32{1}, g.bins)
		if err := g.SetNoiseFloor(flat); err != nil {
			t.Fatal(err)
		}
		power := slices.Clone(flat)
		power[g.bins/2] = 1000 // 30 dB over the floor
		g.computeMask(power)
		return g.mask[g.bins/2-1], g.mask[g.bins/2+1]
	}
	if lo, hi := neighbourGain(g2); lo <= 0.5 || hi <= 0.5 {
		t.Errorf("R=2: the 30 dB spike left its neighbours closed (gain %g, %g), want > 0.5", lo, hi)
	}
	if lo, hi := neighbourGain(g0); lo > 0.35 || hi > 0.35 {
		t.Errorf("R=0: neighbours of the spike opened (gain %g, %g, gFloor %g), want < 0.35", lo, hi, g0.gFloor)
	}
}

// TestTimeSmoothingEngages pins the time-smoothing ring average: with L==0 the
// output gain is the current frame's mask, and with L==2 it is the mean of the
// 2L+1 mask rows. A missing ring average leaves a single spike unaveraged.
func TestTimeSmoothingEngages(t *testing.T) {
	mkGate := func(tsf int) *Gate {
		p := ParamsFor(denoiser.Medium)
		p.TimeSmoothFrames = tsf
		g, err := New(Config{SampleRate: 48000, FrameSize: 256, HopSize: 64, Params: &p})
		if err != nil {
			t.Fatal(err)
		}
		return g
	}

	g0 := mkGate(1) // L = 0
	f := int64(10)
	for k := range g0.bins {
		g0.maskRow(f)[k] = 0.5
	}
	dst0 := make([]float32, g0.bins)
	g0.smoothInto(dst0, f)
	for k := range dst0 {
		if math.Abs(float64(dst0[k]-0.5)) > 1e-6 {
			t.Fatalf("L=0: smoothInto did not return the current mask at bin %d: %g", k, dst0[k])
		}
	}

	g2 := mkGate(5) // L = 2
	for j := f - 4; j <= f; j++ {
		for k := range g2.bins {
			g2.maskRow(j)[k] = 0
		}
	}
	for k := range g2.bins {
		g2.maskRow(f)[k] = 1 // a single spike among the 5 rows
	}
	dst2 := make([]float32, g2.bins)
	g2.smoothInto(dst2, f)
	want := float32(1.0 / 5.0)
	for k := range dst2 {
		if math.Abs(float64(dst2[k]-want)) > 1e-5 {
			t.Fatalf("L=2: time average at bin %d = %g, want %g (5-frame mean of one spike)", k, dst2[k], want)
		}
	}
}

// scalarParityTol bounds the difference between the simd mask chain and a float64
// scalar reference. It is a few multiples of float32 epsilon and simd's sigmoid
// approximation, tight enough that any real wiring divergence (a wrong slope,
// offset, or floor) exceeds it by orders of magnitude.
const scalarParityTol = 2e-3

// TestScalarReferenceParity pins the simd mask chain (Div, SNR box mean,
// Log10Floored, Affine, Sigmoid, Affine) to an independent float64 scalar of the
// same math, on both the SIMD and pure-Go tiers (run under task test:noasm). It
// runs with frequency smoothing off (FreqSmoothBins=1, R=0) and at Medium's R=2,
// whose reference is the edge-aware mean of power/noise over [k-R, k+R] taken
// before the log. A divergence between the vectorized and scalar paths, a
// mis-wired coefficient, or smoothing applied after the knee turns it red.
func TestScalarReferenceParity(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	for _, fsb := range []int{1, ParamsFor(denoiser.Medium).FreqSmoothBins} {
		t.Run(fmt.Sprintf("freqSmoothBins=%d", fsb), func(t *testing.T) {
			p := ParamsFor(denoiser.Medium)
			p.FreqSmoothBins = fsb
			g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(7, 8))
			power := make([]float32, g.bins)
			noise := make([]float32, g.bins)
			for k := range power {
				power[k] = float32(math.Abs(rng.NormFloat64())) * float32(dbToLin(-30))
				noise[k] = float32(math.Abs(rng.NormFloat64())+0.1) * float32(dbToLin(-40))
			}
			dspshared.CopyFloor(g.noiseBuf, noise)
			g.noise = g.noiseBuf
			g.learned = true
			g.computeMask(power)

			ratio := make([]float64, g.bins)
			for k := range ratio {
				ratio[k] = float64(power[k]) / float64(g.noiseBuf[k])
			}
			for k := range power {
				lo, hi := max(0, k-g.freqR), min(g.bins-1, k+g.freqR)
				var sum float64
				for j := lo; j <= hi; j++ {
					sum += ratio[j]
				}
				smoothed := sum / float64(hi-lo+1)
				arg := float64(g.slope)*math.Log10(math.Max(smoothed, 1e-20)) + float64(g.offset)
				s := 1.0 / (1.0 + math.Exp(-arg))
				want := (1-float64(g.gFloor))*s + float64(g.gFloor)
				if d := math.Abs(float64(g.mask[k]) - want); d > scalarParityTol {
					t.Errorf("bin %d: simd mask %g, scalar reference %g (diff %g > %g)", k, g.mask[k], want, d, scalarParityTol)
				}
			}
		})
	}
}

// TestInfMaxAttenuationFullGating streams a +Inf MaxAttenuationDB gate (gFloor=0,
// full gating allowed): the output stays finite and the noise-only clip is
// reduced further than the 12 dB Medium floor could allow, proving gFloor really
// reaches 0. If gFloor were not 0 for +Inf, reduction would cap near the Medium
// floor.
func TestInfMaxAttenuationFullGating(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	noise := whiteNoise(2*sr, dbToLin(-40), 60)
	p := ParamsFor(denoiser.Medium)
	p.MaxAttenuationDB = float32(math.Inf(1))
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p}

	out, err := DenoiseWithNoise(noise, noise[:sr/2], cfg)
	if err != nil {
		t.Fatal(err)
	}
	assertFinite(t, "+Inf gate output", out)
	span := [][2]int{{sr / 4, 2*sr - sr/4}}
	reduction := spanRMSDB(noise, span) - spanRMSDB(out, span)
	medFloor := float64(ParamsFor(denoiser.Medium).MaxAttenuationDB) // 12
	t.Logf("+Inf full-gating reduction %.1f dB (Medium floor caps at %.0f dB)", reduction, medFloor)
	if reduction <= medFloor {
		t.Errorf("+Inf gate reduced noise by %.1f dB, want more than the %.0f dB Medium floor (gFloor did not reach 0)", reduction, medFloor)
	}
}
