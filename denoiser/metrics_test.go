package denoiser

import (
	"math"
	"testing"

	"github.com/tphakala/simd/f32"
)

// spanRMSDB returns the RMS level (dBFS) pooled over the given sample spans.
func spanRMSDB(x []float32, spans [][2]int) float64 {
	var s float64
	var n int
	for _, sp := range spans {
		for _, v := range x[sp[0]:sp[1]] {
			s += float64(v) * float64(v)
		}
		n += sp[1] - sp[0]
	}
	if n == 0 || s == 0 {
		return -200
	}
	return 10 * math.Log10(s/float64(n))
}

// lsdDB is the log-spectral distance between a and b over the given spans:
// the mean over frames of the RMS across bins of the dB difference of their
// Hann power spectra (n-point, hop-spaced, no padding).
func lsdDB(a, b []float32, spans [][2]int, n, hop int) float64 {
	plan, _ := f32.NewSTFTPlan(n)
	win := hannPeriodic(n)
	bins := plan.NumBins()
	var total float64
	var frames int
	for _, sp := range spans {
		fa := plan.NumFrames(sp[1]-sp[0], hop, f32.NoPad)
		pa := make([]float32, fa*bins)
		pb := make([]float32, fa*bins)
		plan.STFTPowerInto(pa, a[sp[0]:sp[1]], win, hop, f32.NoPad)
		plan.STFTPowerInto(pb, b[sp[0]:sp[1]], win, hop, f32.NoPad)
		var peak float32
		for _, v := range pa {
			peak = max(peak, v)
		}
		floor := math.Max(float64(peak)*1e-9, 1e-20)
		for f := range fa {
			var acc float64
			for k := range bins {
				da := 10 * math.Log10(float64(pa[f*bins+k])+floor)
				db := 10 * math.Log10(float64(pb[f*bins+k])+floor)
				acc += (da - db) * (da - db)
			}
			total += math.Sqrt(acc / float64(bins))
			frames++
		}
	}
	if frames == 0 {
		return 0
	}
	return total / float64(frames)
}

// segSNRDB is the segmental SNR of test against clean over the spans, in
// segLen-sample segments, each clamped to [-10, 35] dB, averaged.
func segSNRDB(clean, test []float32, spans [][2]int, segLen int) float64 {
	var total float64
	var segs int
	for _, sp := range spans {
		for s := sp[0]; s+segLen <= sp[1]; s += segLen {
			var sig, errp float64
			for i := s; i < s+segLen; i++ {
				c := float64(clean[i])
				e := c - float64(test[i])
				sig += c * c
				errp += e * e
			}
			if sig == 0 {
				continue
			}
			snr := 10 * math.Log10(sig/math.Max(errp, 1e-20))
			total += math.Min(35, math.Max(-10, snr))
			segs++
		}
	}
	if segs == 0 {
		return 0
	}
	return total / float64(segs)
}

// bestLag returns the lag in [-maxLag, maxLag] maximizing the cross
// correlation of sig against ref over ref[from:to] (positive lag: sig is
// delayed relative to ref).
func bestLag(ref, sig []float32, from, to, maxLag int) int {
	best, bestV := 0, math.Inf(-1)
	for lag := -maxLag; lag <= maxLag; lag++ {
		var acc float64
		for i := from; i < to; i++ {
			j := i + lag
			if j < 0 || j >= len(sig) {
				continue
			}
			acc += float64(ref[i]) * float64(sig[j])
		}
		if acc > bestV {
			bestV, best = acc, lag
		}
	}
	return best
}

// shifted returns y with y[i] = x[i+lag] (zero beyond the ends), undoing a
// measured delay.
func shifted(x []float32, lag int) []float32 {
	y := make([]float32, len(x))
	for i := range y {
		if j := i + lag; j >= 0 && j < len(x) {
			y[i] = x[j]
		}
	}
	return y
}

// TestMetricInstruments calibrates the measurement helpers against known
// inputs. The quality and afftdn-oracle tests use these as their instruments,
// so a silent sign or scale error here would invert an assertion's meaning
// (e.g. segSNRDB's direction) and let a real regression pass.
func TestMetricInstruments(t *testing.T) {
	const sr = 48000
	const n = sr // 1 s
	full := [][2]int{{0, n}}

	// spanRMSDB: a full-span sine of amplitude A has mean-square A^2/2, so its
	// level is 20*log10(A) - 3.01 dBFS. Silence returns the -200 sentinel.
	sine := func(amp float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(amp * math.Sin(2*math.Pi*1000*float64(i)/float64(sr)))
		}
		return x
	}
	if got, want := spanRMSDB(sine(0.5), full), 20*math.Log10(0.5)-10*math.Log10(2); math.Abs(got-want) > 0.05 {
		t.Errorf("spanRMSDB(0.5 sine) = %.3f dBFS, want %.3f", got, want)
	}
	if got := spanRMSDB(make([]float32, n), full); got != -200 {
		t.Errorf("spanRMSDB(silence) = %.1f, want -200 sentinel", got)
	}

	x := whiteNoise(n, 0.2, 5)

	// segSNRDB: identical signals give infinite SNR, clamped to 35 dB; a zeroed
	// test signal gives 0 dB (error power equals signal power).
	if got := segSNRDB(x, x, full, 960); got != 35 {
		t.Errorf("segSNRDB(x, x) = %.2f, want 35 (clamped)", got)
	}
	if got := segSNRDB(x, make([]float32, n), full, 960); math.Abs(got) > 0.01 {
		t.Errorf("segSNRDB(x, zeros) = %.3f, want 0", got)
	}

	// bestLag recovers a known delay and shifted undoes it. delayed[i]=x[i-d].
	const d = 100
	delayed := shifted(x, -d)
	if got := bestLag(x, delayed, n/4, 3*n/4, 256); got != d {
		t.Errorf("bestLag recovered %d, want %d", got, d)
	}
	if undone := shifted(delayed, d); math.Abs(float64(undone[n/2]-x[n/2])) > 1e-6 {
		t.Errorf("shifted did not undo the delay at n/2: %g vs %g", undone[n/2], x[n/2])
	}

	// lsdDB: identical spectra distance is 0; halving the amplitude quarters
	// every bin's power. The shared spectral floor (derived from the reference
	// peak, added to both spectra) sits >70 dB below every bin for this
	// broadband signal, so it is negligible and each bin's dB difference is a
	// uniform 10*log10(4)=6.02 dB, giving a distance of ~6.02 dB.
	if got := lsdDB(x, x, full, 1024, 256); got != 0 {
		t.Errorf("lsdDB(x, x) = %g, want 0", got)
	}
	half := make([]float32, n)
	for i := range half {
		half[i] = x[i] * 0.5
	}
	if got := lsdDB(x, half, full, 1024, 256); math.Abs(got-10*math.Log10(4)) > 0.1 {
		t.Errorf("lsdDB(x, 0.5x) = %.2f dB, want ~%.2f", got, 10*math.Log10(4))
	}
}
