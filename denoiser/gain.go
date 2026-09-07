package denoiser

import "math"

// epsPower floors every noise power estimate so SNR ratios stay finite.
// Inputs are normalized float32 audio, so real bin powers sit far above it.
const epsPower = 1e-12

// lsaVMin and lsaVMax clamp the MMSE-LSA integrand argument: below lsaVMin
// exp(0.5*E1(v)) grows without bound (and is then capped by the gain <= 1
// clamp); above lsaVMax E1 is zero in float32.
const (
	lsaVMin = 1e-6
	lsaVMax = 100
)

// gainState carries the per-bin gain parameters and the decision-directed
// state (the previous frame's estimated clean amplitude squared).
type gainState struct {
	alpha, beta   float32 // SNRSmoothing, OverSubtraction
	xiMin, gFloor float32 // linear MinPriorSNR and residual gain floor
	est           Estimator
	smooth        int // odd width > 1, or 0 for no frequency smoothing
	prevAmp2, tmp []float32
}

func newGainState(bins int, p Params) *gainState {
	g := &gainState{
		alpha:    p.SNRSmoothing,
		beta:     p.OverSubtraction,
		xiMin:    float32(math.Pow(10, float64(p.MinPriorSNRDB)/10)),
		gFloor:   float32(math.Pow(10, -float64(p.MaxAttenuationDB)/20)),
		est:      p.Estimator,
		prevAmp2: make([]float32, bins),
		tmp:      make([]float32, bins),
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
	for k := range gain {
		nz := g.beta * max(noise[k], epsPower)
		gamma := power[k] / nz // a posteriori SNR
		xi := g.alpha*g.prevAmp2[k]/nz + (1-g.alpha)*max(gamma-1, 0)
		if !(xi >= g.xiMin) { // also catches NaN
			xi = g.xiMin
		}
		gn := estimatorGain(g.est, xi, gamma)
		if !(gn <= 1) { // NaN or > 1
			gn = 1
		}
		if gn < g.gFloor {
			gn = g.gFloor
		}
		gain[k] = gn
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

// estimatorGain returns the gain for a priori SNR xi and a posteriori SNR
// gamma under the given rule, before clamping.
func estimatorGain(est Estimator, xi, gamma float32) float32 {
	switch est {
	case Wiener:
		return xi / (1 + xi)
	case Subtraction:
		if gamma <= 1 {
			return 0
		}
		return float32(math.Sqrt(float64(1 - 1/gamma)))
	default: // MMSELSA
		w := xi / (1 + xi)
		v := float64(w * gamma)
		v = min(max(v, lsaVMin), lsaVMax)
		return w * float32(math.Exp(0.5*expint1(v)))
	}
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
