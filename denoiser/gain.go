package denoiser

import (
	"math"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
	simdf32 "github.com/tphakala/simd/f32"
)

// epsPower floors every noise power estimate so SNR ratios stay finite.
const epsPower = dspshared.EpsPower

// lsaVMin and lsaVUnity clamp the MMSE-LSA integrand argument v: below lsaVMin
// exp(0.5*E1(v)) grows without bound (and is then capped by the gain <= 1 clamp).
// From lsaVUnity upward the correction is exactly 1 in float32: E1(16) is about
// 7e-9, so 1+0.5*E1 rounds to 1.0 (float32 spacing at 1 is 1.2e-7) and the gain is
// just w. The bound is conservative: the correction already rounds to 1 from about
// v = 14. Clamping v to lsaVUnity therefore changes no output bit, and keeps the
// vectorized E1 away from arguments near 87 whose results are subnormal and slow
// on x86.
const (
	lsaVMin   = 1e-6
	lsaVUnity = 16
)

// gainState carries the per-bin gain parameters and the decision-directed
// state (the previous frame's estimated clean amplitude squared).
type gainState struct {
	alpha, beta   float32 // SNRSmoothing, OverSubtraction
	xiMin, gFloor float32 // linear MinPriorSNR and residual gain floor
	est           Estimator
	smooth        int // odd width > 1, or 0 for no frequency smoothing
	prevAmp2, tmp []float32
	lsaW, lsaV    []float32 // MMSE-LSA scratch: Wiener gain w and integrand argument v per bin
}

func newGainState(bins int, p Params) *gainState {
	g := &gainState{
		alpha:    p.SNRSmoothing,
		beta:     p.OverSubtraction,
		xiMin:    float32(math.Pow(10, float64(p.MinPriorSNRDB)/10)),
		gFloor:   float32(dsp.FactorFromDB(-float64(p.MaxAttenuationDB))),
		est:      p.Estimator,
		prevAmp2: make([]float32, bins),
		tmp:      make([]float32, bins),
	}
	if p.Estimator == MMSELSA {
		g.lsaW = make([]float32, bins)
		g.lsaV = make([]float32, bins)
	}
	if p.FreqSmoothBins > 1 {
		g.smooth = p.FreqSmoothBins | 1 // round even widths up to odd
	}
	return g
}

// reset clears the decision-directed state (stream start).
func (g *gainState) reset() { clear(g.prevAmp2) }

// compute writes the per-bin gain for one frame given its power spectrum and
// the current noise power, then updates the decision-directed state. All three
// slices have the same length.
func (g *gainState) compute(gain, power, noise []float32) {
	if g.est == MMSELSA {
		g.lsaGain(gain, power, noise)
	} else {
		g.ruleGain(gain, power, noise)
	}
	if g.smooth > 1 {
		smoothGain(gain, g.tmp, g.smooth)
	}
	for k := range gain {
		a := gain[k] * gain[k] * power[k]
		if !(a < math.MaxFloat32) { // NaN or +Inf
			a = 0
		}
		g.prevAmp2[k] = a
	}
}

// ruleGain writes the per-bin gain for the closed-form estimators (Wiener and
// spectral subtraction), clamped to [gFloor, 1].
func (g *gainState) ruleGain(gain, power, noise []float32) {
	for k := range gain {
		xi, gamma := g.snr(power[k], noise[k], g.prevAmp2[k])
		gain[k] = g.clampGain(estimatorGain(g.est, xi, gamma))
	}
}

// lsaGain writes the per-bin MMSE-LSA gain, w * exp(0.5*E1(v)) with w = xi/(1+xi)
// and v = w*gamma. The scalar pass computes w and v per bin, the transcendental
// part runs as three vector passes over the whole frame, and the last scalar pass
// applies the correction and the clamps.
func (g *gainState) lsaGain(gain, power, noise []float32) {
	w, v := g.lsaW[:len(gain)], g.lsaV[:len(gain)]
	for k := range gain {
		xi, gamma := g.snr(power[k], noise[k], g.prevAmp2[k])
		wk := xi / (1 + xi)
		w[k] = wk
		v[k] = min(max(wk*gamma, lsaVMin), lsaVUnity)
	}
	simdf32.ExpE1InPlace(v)
	simdf32.Scale(v, v, 0.5)
	simdf32.ExpInPlace(v)
	for k := range gain {
		gain[k] = g.clampGain(w[k] * v[k])
	}
}

// snr returns the a priori SNR xi (decision-directed, floored at xiMin) and the a
// posteriori SNR gamma of one bin.
func (g *gainState) snr(power, noise, prevAmp2 float32) (xi, gamma float32) {
	nz := g.beta * max(noise, epsPower)
	gamma = power / nz
	xi = g.alpha*prevAmp2/nz + (1-g.alpha)*max(gamma-1, 0)
	if !(xi >= g.xiMin) { // also catches NaN
		xi = g.xiMin
	}
	return xi, gamma
}

// clampGain limits a gain to [gFloor, 1]; NaN and values above 1 become 1.
func (g *gainState) clampGain(gn float32) float32 {
	if !(gn <= 1) { // NaN or > 1
		return 1
	}
	return max(gn, g.gFloor)
}

// estimatorGain returns the gain of a closed-form rule for a priori SNR xi and a
// posteriori SNR gamma, before clamping. MMSELSA is evaluated by lsaGain.
func estimatorGain(est Estimator, xi, gamma float32) float32 {
	if est == Subtraction {
		if gamma <= 1 {
			return 0
		}
		return float32(math.Sqrt(float64(1 - 1/gamma)))
	}
	return xi / (1 + xi) // Wiener
}

// smoothGain replaces gain with its centered moving average of the given odd
// width; the edge bins average over the part of the window that exists. tmp
// is scratch of the same length.
func smoothGain(gain, tmp []float32, width int) {
	r := width / 2
	copy(tmp, gain)
	for k := range gain {
		lo, hi := max(0, k-r), min(len(gain)-1, k+r)
		var s float32
		for j := lo; j <= hi; j++ {
			s += tmp[j]
		}
		gain[k] = s / float32(hi-lo+1)
	}
}
