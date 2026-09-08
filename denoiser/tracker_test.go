package denoiser

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// noisePower returns one frame of exponentially distributed bin powers with
// per-bin mean level[k] (the statistics of a Gaussian noise periodogram).
func noisePower(rng *rand.Rand, level, dst []float32) {
	for k := range dst {
		dst[k] = level[k] * float32(rng.ExpFloat64())
	}
}

func dbRatio(a, b float32) float64 { return 10 * math.Log10(float64(a)/float64(b)) }

func TestMCRAConvergesAndTracksSteps(t *testing.T) {
	const bins, window = 64, 200
	rng := rand.New(rand.NewPCG(7, 8))
	level := make([]float32, bins)
	for k := range level {
		level[k] = 1e-4 * (1 + float32(k)/16)
	}
	m := newMCRA(bins, window)
	p := make([]float32, bins)
	// A cold MCRA start needs about three windows to fully settle: the
	// two-buffer minimum takes up to two windows to flush the volatile first
	// frames out of smin before the presence gate opens reliably, then the
	// recursive average climbs to the true level. This matches the settling
	// budget the step phases below use.
	for range 3 * window {
		noisePower(rng, level, p)
		m.update(p)
	}
	checkWithin := func(what string, tolMedian, tolMax float64) {
		t.Helper()
		errs := make([]float64, bins)
		for k := range errs {
			errs[k] = math.Abs(dbRatio(m.noise[k], level[k]))
		}
		sorted := slices.Clone(errs)
		slices.Sort(sorted)
		if med := sorted[bins/2]; med > tolMedian {
			t.Errorf("%s: median |error| %.2f dB > %.1f", what, med, tolMedian)
		}
		if mx := sorted[bins-1]; mx > tolMax {
			t.Errorf("%s: max |error| %.2f dB > %.1f", what, mx, tolMax)
		}
	}
	checkWithin("stationary", 1.0, 3.0)

	// +10 dB step. The gate closes until the two-buffer minimum forgets the old
	// level, which takes up to 2 windows when the step lands mid-window, then
	// the recursive average settles in ~60 frames: allow 3 windows.
	for k := range level {
		level[k] *= 10
	}
	for range 3 * window {
		noisePower(rng, level, p)
		m.update(p)
	}
	checkWithin("after +10 dB step", 1.5, 3.5)

	// -10 dB step: tracks down quickly too.
	for k := range level {
		level[k] /= 10
	}
	for range 2 * window {
		noisePower(rng, level, p)
		m.update(p)
	}
	checkWithin("after -10 dB step", 1.5, 3.5)
}

func TestMCRAIgnoresToneBurst(t *testing.T) {
	const bins, window = 64, 200
	rng := rand.New(rand.NewPCG(9, 10))
	level := make([]float32, bins)
	for k := range level {
		level[k] = 1e-4
	}
	m := newMCRA(bins, window)
	p := make([]float32, bins)
	for range 2 * window {
		noisePower(rng, level, p)
		m.update(p)
	}
	before := m.noise[20]
	for range window / 2 { // a +30 dB burst in bin 20 for half a window
		noisePower(rng, level, p)
		p[20] += 1000 * level[20]
		m.update(p)
	}
	if d := math.Abs(dbRatio(m.noise[20], before)); d > 0.5 {
		t.Errorf("bin 20 noise estimate moved %.2f dB during a +30 dB burst, want < 0.5", d)
	}
	// Neighbouring bins unaffected as well.
	if d := math.Abs(dbRatio(m.noise[21], level[21])); d > 3 {
		t.Errorf("bin 21 drifted %.2f dB", d)
	}
}

func TestMCRAResetAndWindowFrames(t *testing.T) {
	m := newMCRA(4, 10)
	if !m.started {
		m.update([]float32{1, 1, 1, 1})
	}
	m.reset()
	if m.started || m.count != 0 {
		t.Error("reset did not clear state")
	}
	for _, v := range m.noise {
		if v != epsPower {
			t.Errorf("reset noise = %g, want epsPower", v)
		}
	}
	if f := trackWindowFrames(2, 48000, 256); f != 375 {
		t.Errorf("trackWindowFrames(2 s) = %d, want 375", f)
	}
	if f := trackWindowFrames(0.001, 48000, 256); f != 1 {
		t.Errorf("trackWindowFrames(tiny) = %d, want 1", f)
	}
	// A NaN frame must not poison the tracker: state stays finite and the
	// tracker keeps adapting afterwards.
	m = newMCRA(2, 10)
	m.update([]float32{1, 1})
	m.update([]float32{float32(math.NaN()), float32(math.Inf(1))})
	for range 50 {
		m.update([]float32{2, 2})
	}
	for k := range 2 {
		if !(m.s[k] < math.MaxFloat32) || !(m.smin[k] < math.MaxFloat32) || !(m.noise[k] < math.MaxFloat32) {
			t.Fatalf("bin %d: non-finite tracker state after a NaN frame", k)
		}
		if m.noise[k] < 1.5 {
			t.Errorf("bin %d: noise %g did not keep adapting after a NaN frame", k, m.noise[k])
		}
	}
}

func TestAdaptiveStreamReducesStationaryNoise(t *testing.T) {
	const sr, sigma = 48000, 0.01
	d, _ := New(Config{SampleRate: sr, Preset: Medium}) // no profile: adaptive
	x := whiteNoise(6*sr, sigma, 30)
	y := runStream(t, d, x, []int{4096})
	// After the tracker has converged (the last 2 s), Medium reduces by at
	// least 9 dB.
	if red := rmsDB(x[4*sr:]) - rmsDB(y[4*sr:]); red < 9 {
		t.Errorf("adaptive Medium reduced white noise by %.1f dB after convergence, want >= 9", red)
	}
}
