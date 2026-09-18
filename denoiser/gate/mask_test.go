package gate

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestGateAttenuatesNoiseKeepsTone pins the core gate mechanism: with a learned
// floor it reduces the noise-only regions by at least MaxAttenuationDB-4 while
// leaving a loud tone essentially untouched (within 1 dB). A broken sigmoid or
// power ratio fails one side or the other. Frequency smoothing is disabled here so
// the test isolates the gate: a centered gain average across bins deliberately
// softens a single pure sine by a few dB (its worst case, since the tone's
// neighbour bins are noise-only), which is the smoothing tradeoff exercised by
// TestFreqSmoothingEngages, not a property of the gate itself.
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

	// Noise reduction with the tuned Medium params (frequency smoothing on).
	full := ParamsFor(denoiser.Medium)
	outFull, err := DenoiseWithNoise(clip, noise[:sr/2], Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &full})
	if err != nil {
		t.Fatal(err)
	}
	if len(outFull) != len(clip) {
		t.Fatalf("length %d != %d", len(outFull), len(clip))
	}
	// Guard before the dB metrics: spanRMSDB propagates NaN, and reduction < want
	// is false on a NaN operand, so a NaN-emitting regression would pass vacuously.
	assertFinite(t, "outFull", outFull)
	reduction := spanRMSDB(clip, noiseSpans) - spanRMSDB(outFull, noiseSpans)

	// Tone preservation with the gate isolated from frequency smoothing.
	iso := full
	iso.FreqSmoothBins = 0
	outIso, err := DenoiseWithNoise(clip, noise[:sr/2], Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &iso})
	if err != nil {
		t.Fatal(err)
	}
	assertFinite(t, "outIso", outIso)
	toneChange := spanRMSDB(outIso, toneSpans) - spanRMSDB(clip, toneSpans)

	t.Logf("noise reduction %.1f dB (floor %g dB), tone-region level change %.2f dB", reduction, full.MaxAttenuationDB, toneChange)
	if want := float64(full.MaxAttenuationDB) - 4; reduction < want {
		t.Errorf("noise reduction %.1f dB < %.1f (MaxAttenuationDB-4)", reduction, want)
	}
	if math.Abs(toneChange) > 1.0 {
		t.Errorf("tone-region level changed by %.2f dB (>1 dB); the gate is eating the tone", toneChange)
	}
}

// TestFreqSmoothingEngages pins the frequency-smoothing pass: with R>0 a spike in
// the gain spreads to its neighbours and the centre falls; with R==0 the pass is
// an exact identity. A broken shifted Add leaves neighbours unmoved.
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
}

// TestTimeSmoothingEngages pins the time-smoothing ring average: with L==0 the
// output gain is the current frame's mask, and with L==2 it is the mean of the
// 2L+1 mask rows. A missing ring average leaves a single spike unaveraged.
func TestTimeSmoothingEngages(t *testing.T) {
	mkGate := func(tsf int) *Gate {
		p := ParamsFor(denoiser.Medium)
		p.TimeSmoothFrames = tsf
		p.FreqSmoothBins = 1 // isolate time smoothing
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

// TestScalarReferenceParity pins the simd mask chain (Div, Log10Floored, Affine,
// Sigmoid, Affine) to an independent float64 scalar of the same math, on both the
// SIMD and pure-Go tiers (run under task test:noasm). A divergence between the
// vectorized and scalar paths, or a mis-wired coefficient, turns it red.
func TestScalarReferenceParity(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
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
	copyFloor(g.noiseBuf, noise)
	g.noise = g.noiseBuf
	g.learned = true
	g.computeMask(power)

	for k := range power {
		ratio := float64(power[k]) / float64(g.noiseBuf[k])
		arg := float64(g.slope)*math.Log10(math.Max(ratio, 1e-20)) + float64(g.offset)
		s := 1.0 / (1.0 + math.Exp(-arg))
		want := (1-float64(g.gFloor))*s + float64(g.gFloor)
		if d := math.Abs(float64(g.mask[k]) - want); d > scalarParityTol {
			t.Errorf("bin %d: simd mask %g, scalar reference %g (diff %g > %g)", k, g.mask[k], want, d, scalarParityTol)
		}
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
