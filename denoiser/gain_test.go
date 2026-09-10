package denoiser

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestEstimatorGainRules(t *testing.T) {
	if g := estimatorGain(Wiener, 1, 10); math.Abs(float64(g)-0.5) > 1e-6 {
		t.Errorf("Wiener(xi=1) = %g, want 0.5", g)
	}
	if g := estimatorGain(Subtraction, 1, 0.5); g != 0 {
		t.Errorf("Subtraction(gamma<=1) = %g, want 0", g)
	}
	if g := estimatorGain(Subtraction, 1, 4); math.Abs(float64(g)-math.Sqrt(0.75)) > 1e-6 {
		t.Errorf("Subtraction(gamma=4) = %g, want sqrt(0.75)", g)
	}
	// MMSE-LSA at xi = -15 dB, gamma = 1: 0.0316/1.0316 * exp(0.5*E1(0.0306)) ~ 0.133.
	if g := estimatorGain(MMSELSA, 0.0316, 1); math.Abs(float64(g)-0.133) > 0.005 {
		t.Errorf("LSA(xi=-15dB, gamma=1) = %g, want ~0.133", g)
	}
	// High SNR: all three tend to 1.
	for _, e := range []Estimator{MMSELSA, Wiener, Subtraction} {
		if g := estimatorGain(e, 1e6, 1e6); g < 0.999 {
			t.Errorf("%v at high SNR = %g, want ~1", e, g)
		}
	}
	// LSA >= Wiener for equal (xi, gamma); Wiener monotone in xi.
	prev := float32(0)
	for xi := float32(0.001); xi < 100; xi *= 1.5 {
		for gamma := float32(0.1); gamma < 100; gamma *= 2 {
			if l, w := estimatorGain(MMSELSA, xi, gamma), estimatorGain(Wiener, xi, gamma); l < w-1e-6 {
				t.Fatalf("LSA %g < Wiener %g at xi=%g gamma=%g", l, w, xi, gamma)
			}
		}
		if w := estimatorGain(Wiener, xi, 1); w < prev {
			t.Fatalf("Wiener not monotone at xi=%g", xi)
		}
		prev = estimatorGain(Wiener, xi, 1)
	}
}

func TestSmoothGain(t *testing.T) {
	g := []float32{1, 0, 0, 0, 1}
	smoothGain(g, make([]float32, 5), 3)
	want := []float32{0.5, 1.0 / 3, 0, 1.0 / 3, 0.5}
	for i := range g {
		if math.Abs(float64(g[i]-want[i])) > 1e-6 {
			t.Fatalf("smoothed[%d] = %g, want %g", i, g[i], want[i])
		}
	}
}

func TestGainStateFloorAndUnity(t *testing.T) {
	const bins = 8
	power := make([]float32, bins)
	noise := make([]float32, bins)
	gain := make([]float32, bins)
	for k := range power {
		power[k] = 1
		noise[k] = epsPower
	}
	// Negligible noise: every estimator passes the signal through.
	for _, e := range []Estimator{MMSELSA, Wiener, Subtraction} {
		p := ParamsFor(Medium)
		p.Estimator = e
		g := newGainState(bins, p)
		g.compute(gain, power, noise)
		for k, v := range gain {
			if v != 1 {
				t.Errorf("%v: gain[%d] = %g with negligible noise, want 1", e, k, v)
			}
		}
	}
	// Zero power: a posteriori SNR gamma is 0, so the MMSE-LSA term overshoots
	// (v is floored at lsaVMin), the G <= 1 clamp caps it at 1, and 1 is above
	// the floor, so the gain is 1 (per the spec: G = min(G,1) then max(G,Gfloor)).
	// The DD state is still 0 because it is gain^2 * power. The gain value here
	// is functionally moot: output and DD state multiply by zero power either way.
	g := newGainState(bins, ParamsFor(Medium))
	clear(power)
	for k := range noise {
		noise[k] = 1
	}
	g.compute(gain, power, noise)
	for k, v := range gain {
		if v != 1 || g.prevAmp2[k] != 0 {
			t.Errorf("gain[%d] = %g (prevAmp2 %g), want 1 and 0", k, v, g.prevAmp2[k])
		}
	}
	// MaxAttenuationDB = 0 is a hard unity gain even when power << noise.
	p := ParamsFor(Medium)
	p.MaxAttenuationDB = 0
	g = newGainState(bins, p)
	for k := range power {
		power[k] = 1e-6
	}
	g.compute(gain, power, noise)
	for k, v := range gain {
		if v != 1 {
			t.Errorf("MaxAttenuationDB=0: gain[%d] = %g, want 1", k, v)
		}
	}
}

func TestGainStateBoundsAndNaN(t *testing.T) {
	const bins = 64
	rng := rand.New(rand.NewPCG(3, 4))
	power := make([]float32, bins)
	noise := make([]float32, bins)
	gain := make([]float32, bins)
	for _, strength := range []Strength{Light, Medium, Heavy} {
		p := ParamsFor(strength)
		g := newGainState(bins, p)
		floor := float32(math.Pow(10, -float64(p.MaxAttenuationDB)/20))
		for frame := range 50 {
			for k := range power {
				power[k] = float32(rng.ExpFloat64()) * 1e-3
				noise[k] = 1e-3
			}
			g.compute(gain, power, noise)
			for k, v := range gain {
				if !(v >= floor-1e-6 && v <= 1) {
					t.Fatalf("%v frame %d: gain[%d] = %g outside [%g, 1]", strength, frame, k, v, floor)
				}
			}
		}
		// A NaN frame must not poison the next one.
		power[3] = float32(math.NaN())
		g.compute(gain, power, noise)
		if g.prevAmp2[3] != 0 {
			t.Errorf("%v: prevAmp2 after NaN = %g, want 0", strength, g.prevAmp2[3])
		}
		power[3] = 1e-3
		g.compute(gain, power, noise)
		if math.IsNaN(float64(gain[3])) {
			t.Errorf("%v: NaN leaked into the following frame", strength)
		}
		g.reset()
		for k := range g.prevAmp2 {
			if g.prevAmp2[k] != 0 {
				t.Fatalf("reset left prevAmp2[%d] = %g", k, g.prevAmp2[k])
			}
		}
	}
}
