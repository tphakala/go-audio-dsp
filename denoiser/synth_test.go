package denoiser

import (
	"math"
	"math/rand/v2"
)

// synthClip is a 10 s synthetic test clip: stationary noise plus four tone or
// chirp bursts at known positions, with noise-only spans kept clear of the
// bursts for measurement. The separate noise track is kept alongside clean and
// mix so a test can build an oracle noise profile from it via
// DenoiseWithNoise(mix, noise, ...).
type synthClip struct {
	sr                      int
	clean, noise, mix       []float32
	signalSpans, noiseSpans [][2]int
}

func dbToLin(db float64) float64 { return math.Pow(10, db/20) }

// pinkFilter turns white noise into approximately 1/f noise (Kellet's
// three-pole economy filter).
func pinkFilter(x []float32) {
	var b0, b1, b2 float64
	for i, w := range x {
		wf := float64(w)
		b0 = 0.99765*b0 + wf*0.0990460
		b1 = 0.96300*b1 + wf*0.2965164
		b2 = 0.57000*b2 + wf*1.0526913
		x[i] = float32(b0 + b1 + b2 + wf*0.1848)
	}
}

func scaleToRMS(x []float32, rms float64) {
	cur := dbToLin(rmsDB(x))
	g := float32(rms / cur)
	for i := range x {
		x[i] *= g
	}
}

func makeSynthClip(sr int, noiseDBFS, sigDBFS float64, pink bool, seed uint64) synthClip {
	const seconds = 10
	n := seconds * sr
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	noise := make([]float32, n)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64())
	}
	if pink {
		pinkFilter(noise)
	}
	scaleToRMS(noise, dbToLin(noiseDBFS))

	clean := make([]float32, n)
	bursts := []struct{ t0, t1, f0, f1 float64 }{
		{2.0, 2.5, 4000, 4000}, {4.0, 4.6, 2000, 6000}, {6.0, 6.5, 6000, 6000}, {8.0, 8.4, 3000, 5000},
	}
	amp := dbToLin(sigDBFS) * math.Sqrt2
	c := synthClip{sr: sr}
	for _, b := range bursts {
		i0, i1 := int(b.t0*float64(sr)), int(b.t1*float64(sr))
		dur := float64(i1-i0) / float64(sr)
		for i := i0; i < i1; i++ {
			tt := float64(i-i0) / float64(sr)
			ph := 2 * math.Pi * (b.f0*tt + (b.f1-b.f0)*tt*tt/(2*dur))
			env := math.Min(1, math.Min(tt, dur-tt)/0.01) // 10 ms fades
			clean[i] = float32(amp * env * math.Sin(ph))
		}
		c.signalSpans = append(c.signalSpans, [2]int{i0, i1})
	}
	for _, s := range [][2]float64{{0.2, 1.8}, {3.0, 3.8}, {7.0, 7.8}, {9.0, 9.8}} {
		c.noiseSpans = append(c.noiseSpans, [2]int{int(s[0] * float64(sr)), int(s[1] * float64(sr))})
	}
	mix := make([]float32, n)
	for i := range mix {
		mix[i] = clean[i] + noise[i]
	}
	c.clean, c.noise, c.mix = clean, noise, mix
	return c
}
