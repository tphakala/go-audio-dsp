package audiotest

import (
	"math"
	"math/rand/v2"
)

// RMSDB returns 10*log10 of the mean square of x, i.e. its RMS level in dBFS
// (-200 for silence).
func RMSDB(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	if s == 0 || len(x) == 0 {
		return -200
	}
	return 10 * math.Log10(s/float64(len(x)))
}

// SynthClip is a 10 s synthetic test clip: stationary noise plus four tone or
// chirp bursts at known positions, with noise-only spans kept clear of the
// bursts for measurement. The separate noise track is kept alongside clean and
// mix so a test can build an oracle noise profile from it via
// DenoiseWithNoise(mix, noise, ...).
type SynthClip struct {
	SR                      int
	Clean, Noise, Mix       []float32
	SignalSpans, NoiseSpans [][2]int
}

// DBToLin converts decibels to a linear amplitude ratio.
func DBToLin(db float64) float64 { return math.Pow(10, db/20) }

// PinkFilter turns white noise into approximately 1/f noise (Kellet's
// three-pole economy filter).
func PinkFilter(x []float32) {
	var b0, b1, b2 float64
	for i, w := range x {
		wf := float64(w)
		b0 = 0.99765*b0 + wf*0.0990460
		b1 = 0.96300*b1 + wf*0.2965164
		b2 = 0.57000*b2 + wf*1.0526913
		x[i] = float32(b0 + b1 + b2 + wf*0.1848)
	}
}

// ScaleToRMS scales x in place to the given linear RMS.
func ScaleToRMS(x []float32, rms float64) {
	cur := DBToLin(RMSDB(x))
	g := float32(rms / cur)
	for i := range x {
		x[i] *= g
	}
}

// synthSeconds is the length of every SynthClip.
const synthSeconds = 10

// MakeSynthClip builds a SynthClip at sr Hz with noise at noiseDBFS and signal
// bursts at sigDBFS (RMS). pink selects 1/f rather than white noise; seed makes
// it reproducible.
func MakeSynthClip(sr int, noiseDBFS, sigDBFS float64, pink bool, seed uint64) SynthClip {
	n := synthSeconds * sr
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	noise := make([]float32, n)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64())
	}
	if pink {
		PinkFilter(noise)
	}
	ScaleToRMS(noise, DBToLin(noiseDBFS))
	return MixBursts(sr, noise, sigDBFS)
}

// MixBursts lays the four SynthClip signal bursts at sigDBFS (RMS) over the
// caller's noise, which must be exactly the SynthClip length (10 s at sr) and is
// used as is (no rescaling). It lets a test pair a recorded noise bed with
// synthetic signals of known position and level, so a clean reference exists for
// distance metrics on realistic noise.
func MixBursts(sr int, noise []float32, sigDBFS float64) SynthClip {
	n := synthSeconds * sr
	if len(noise) != n {
		panic("audiotest: MixBursts noise must be exactly 10 s")
	}
	clean := make([]float32, n)
	bursts := []struct{ t0, t1, f0, f1 float64 }{
		{2.0, 2.5, 4000, 4000}, {4.0, 4.6, 2000, 6000}, {6.0, 6.5, 6000, 6000}, {8.0, 8.4, 3000, 5000},
	}
	amp := DBToLin(sigDBFS) * math.Sqrt2
	c := SynthClip{SR: sr}
	for _, b := range bursts {
		i0, i1 := int(b.t0*float64(sr)), int(b.t1*float64(sr))
		dur := float64(i1-i0) / float64(sr)
		for i := i0; i < i1; i++ {
			tt := float64(i-i0) / float64(sr)
			ph := 2 * math.Pi * (b.f0*tt + (b.f1-b.f0)*tt*tt/(2*dur))
			env := math.Min(1, math.Min(tt, dur-tt)/0.01) // 10 ms fades
			clean[i] = float32(amp * env * math.Sin(ph))
		}
		c.SignalSpans = append(c.SignalSpans, [2]int{i0, i1})
	}
	for _, s := range [][2]float64{{0.2, 1.8}, {3.0, 3.8}, {7.0, 7.8}, {9.0, 9.8}} {
		c.NoiseSpans = append(c.NoiseSpans, [2]int{int(s[0] * float64(sr)), int(s[1] * float64(sr))})
	}
	mix := make([]float32, n)
	for i := range mix {
		mix[i] = clean[i] + noise[i]
	}
	c.Clean, c.Noise, c.Mix = clean, noise, mix
	return c
}
