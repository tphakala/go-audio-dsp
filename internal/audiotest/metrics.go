package audiotest

import (
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
	"github.com/tphakala/simd/f32"
)

// SpanRMSDB returns the RMS level (dBFS) pooled over the given sample spans.
func SpanRMSDB(x []float32, spans [][2]int) float64 {
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

// LSDDB is the log-spectral distance between a and b over the given spans:
// the mean over frames of the RMS across bins of the dB difference of their
// Hann power spectra (n-point, hop-spaced, no padding).
func LSDDB(a, b []float32, spans [][2]int, n, hop int) float64 {
	plan, _ := f32.NewSTFTPlan(n)
	win := stft.GenerateWindow(stft.Hann, n)
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

// SegSNRDB is the segmental SNR of test against clean over the spans, in
// segLen-sample segments, each clamped to [-10, 35] dB, averaged.
func SegSNRDB(clean, test []float32, spans [][2]int, segLen int) float64 {
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

// BestLag returns the lag in [-maxLag, maxLag] maximizing the cross
// correlation of sig against ref over ref[from:to] (positive lag: sig is
// delayed relative to ref).
func BestLag(ref, sig []float32, from, to, maxLag int) int {
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

// Shifted returns y with y[i] = x[i+lag] (zero beyond the ends), undoing a
// measured delay.
func Shifted(x []float32, lag int) []float32 {
	y := make([]float32, len(x))
	for i := range y {
		if j := i + lag; j >= 0 && j < len(x) {
			y[i] = x[j]
		}
	}
	return y
}

// AssertFinite fails if x holds a NaN or Inf sample. A non-finite sample makes
// SpanRMSDB return NaN, and every ordered metric comparison against NaN is false,
// so the A/B bars would pass on invalid audio; reject it before measuring.
func AssertFinite(t *testing.T, what string, x []float32) {
	t.Helper()
	for i, v := range x {
		if f := float64(v); math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("%s has a non-finite sample at index %d (%v)", what, i, v)
		}
	}
}
