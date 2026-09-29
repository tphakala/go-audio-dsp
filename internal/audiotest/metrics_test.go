package audiotest

import (
	"math"
	"math/rand/v2"
	"testing"
)

// whiteNoise returns n samples of Gaussian noise at the given RMS.
func whiteNoise(n int, rms float64, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * rms)
	}
	return x
}

// TestMetricInstruments calibrates the measurement helpers against known
// inputs. The quality and afftdn-oracle tests use these as their instruments,
// so a silent sign or scale error here would invert an assertion's meaning
// (e.g. SegSNRDB's direction) and let a real regression pass.
func TestMetricInstruments(t *testing.T) {
	const sr = 48000
	const n = sr // 1 s
	full := [][2]int{{0, n}}

	// SpanRMSDB: a full-span sine of amplitude A has mean-square A^2/2, so its
	// level is 20*log10(A) - 3.01 dBFS. Silence returns the -200 sentinel.
	sine := func(amp float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(amp * math.Sin(2*math.Pi*1000*float64(i)/float64(sr)))
		}
		return x
	}
	if got, want := SpanRMSDB(sine(0.5), full), 20*math.Log10(0.5)-10*math.Log10(2); math.Abs(got-want) > 0.05 {
		t.Errorf("SpanRMSDB(0.5 sine) = %.3f dBFS, want %.3f", got, want)
	}
	if got := SpanRMSDB(make([]float32, n), full); got != -200 {
		t.Errorf("SpanRMSDB(silence) = %.1f, want -200 sentinel", got)
	}

	x := whiteNoise(n, 0.2, 5)

	// SegSNRDB: identical signals give infinite SNR, clamped to 35 dB; a zeroed
	// test signal gives 0 dB (error power equals signal power).
	if got := SegSNRDB(x, x, full, 960); got != 35 {
		t.Errorf("SegSNRDB(x, x) = %.2f, want 35 (clamped)", got)
	}
	if got := SegSNRDB(x, make([]float32, n), full, 960); math.Abs(got) > 0.01 {
		t.Errorf("SegSNRDB(x, zeros) = %.3f, want 0", got)
	}

	// BestLag recovers a known delay and Shifted undoes it. delayed[i]=x[i-d].
	const d = 100
	delayed := Shifted(x, -d)
	if got := BestLag(x, delayed, n/4, 3*n/4, 256); got != d {
		t.Errorf("BestLag recovered %d, want %d", got, d)
	}
	if undone := Shifted(delayed, d); math.Abs(float64(undone[n/2]-x[n/2])) > 1e-6 {
		t.Errorf("Shifted did not undo the delay at n/2: %g vs %g", undone[n/2], x[n/2])
	}

	// LSDDB: identical spectra give ~0 distance (bit-identical on amd64, a
	// sub-1e-15 residual on arm64's NEON path, so compare with a tolerance).
	// Halving the amplitude quarters every bin's power. The shared spectral
	// floor (derived from the reference peak, added to both spectra) sits >70 dB
	// below every bin for this broadband signal, so it is negligible and each
	// bin's dB difference is a uniform 10*log10(4)=6.02 dB, giving ~6.02 dB.
	if got := LSDDB(x, x, full, 1024, 256); math.Abs(got) > 1e-6 {
		t.Errorf("LSDDB(x, x) = %g, want ~0 (identical spectra)", got)
	}
	half := make([]float32, n)
	for i := range half {
		half[i] = x[i] * 0.5
	}
	if got := LSDDB(x, half, full, 1024, 256); math.Abs(got-10*math.Log10(4)) > 0.1 {
		t.Errorf("LSDDB(x, 0.5x) = %.2f dB, want ~%.2f", got, 10*math.Log10(4))
	}
}
