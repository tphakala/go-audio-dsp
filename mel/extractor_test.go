package mel

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// toneSig builds a mono signal of nSamples at sr Hz summing the given tones.
func toneSig(nSamples, sr int, freqs []float64, amp float64) []float32 {
	s := make([]float32, nSamples)
	for i := range s {
		var v float64
		for _, f := range freqs {
			v += amp * math.Sin(2*math.Pi*f*float64(i)/float64(sr))
		}
		s[i] = float32(v)
	}
	return s
}

// refSource returns the per-bin source one frame projects, in float64: the power
// |X|^2, its square root for InputMagnitude, or the signed real part Re(X) for
// InputReal. power and spec are the same frame; only the one cfg.Input needs is
// read.
func refSource(input Input, power []float32, spec []complex64) []float64 {
	switch input {
	case InputReal:
		src := make([]float64, len(spec))
		for k, v := range spec {
			src[k] = float64(real(v))
		}
		return src
	case InputMagnitude:
		src := make([]float64, len(power))
		for k, p := range power {
			src[k] = math.Sqrt(float64(p))
		}
		return src
	default:
	}
	src := make([]float64, len(power))
	for k, p := range power {
		src[k] = float64(p)
	}
	return src
}

// refMel projects one frame's per-bin source (see refSource) through the dense
// filterbank and applies the compression stage in float64, mirroring
// projector.fromPower and fromSpectrum. It returns the linear mel energies
// (pre-compression, signed for InputReal) and the final column, so a test can gate
// the comparison on the 40 dB-down floor.
func refMel(src []float64, dense [][]float32, cfg Config) (lin, out []float64) {
	offset, floor := float64(float32(cfg.LogOffset)), float64(float32(cfg.LogFloor))
	exponent := float64(float32(cfg.PowerExponent))
	lin = make([]float64, len(dense))
	out = make([]float64, len(dense))
	for m := range dense {
		var acc float64
		row := dense[m]
		for k := range src {
			acc += float64(row[k]) * src[k]
		}
		lin[m] = acc
		val := acc
		if cfg.Log != LogNone {
			if cfg.Log == PowerLaw {
				val = math.Abs(val)
			}
			if offset > 0 {
				val += offset
			}
			if floor > 0 {
				val = math.Max(val, floor)
			}
			switch cfg.Log {
			case Log10:
				val = math.Log10(val)
			case LogNatural:
				val = math.Log(val)
			case PowerLaw:
				val = math.Pow(val, exponent)
			case LogNone:
			}
		}
		out[m] = val
	}
	return lin, out
}

// customBank builds a small synthetic dense filterbank (overlapping triangles)
// for the FilterbankFromRows path, numMels rows over numBins bins.
func customBankRows(numMels, numBins int) [][]float32 {
	rows := make([][]float32, numMels)
	step := float64(numBins-1) / float64(numMels+1)
	for m := range rows {
		rows[m] = make([]float32, numBins)
		lo := step * float64(m)
		ctr := step * float64(m+1)
		hi := step * float64(m+2)
		for k := range numBins {
			f := float64(k)
			var w float64
			if f > lo && f < hi {
				if f <= ctr {
					w = (f - lo) / (ctr - lo)
				} else {
					w = (hi - f) / (hi - ctr)
				}
			}
			rows[m][k] = float32(w)
		}
	}
	return rows
}

func melTestConfigs(t *testing.T) map[string]Config {
	t.Helper()
	fb, err := FilterbankFromRows(customBankRows(24, 513))
	if err != nil {
		t.Fatal(err)
	}
	return map[string]Config{
		// Signed real-part input with a power law (rows ~40 bins wide, so the
		// DotProduct path) and a non-divisor hop, the BirdNET-like shape.
		"real-powerlaw-custom": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 280, Filterbank: fb,
			Input: InputReal, Log: PowerLaw, PowerExponent: 0.4,
		},
		"real-nolog-generated": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40,
			MinHz: 500, MaxHz: 20000, Input: InputReal, Log: LogNone,
		},
		"power-log10": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40,
			MinHz: 500, MaxHz: 20000, Input: InputPower, Log: Log10, LogOffset: 1e-6,
		},
		"htk-mag-natural": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40,
			MinHz: 500, MaxHz: 20000, Scale: HTK, Input: InputMagnitude,
			Log: LogNatural, LogFloor: 1e-10,
		},
		"linear-nolog": {
			SampleRate: 48000, FrameSize: 512, HopSize: 128, NumMels: 32,
			MinHz: 0, MaxHz: 0, Input: InputPower, Log: LogNone,
		},
		"nonenorm-offset-floor": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 300, NumMels: 40,
			MinHz: 500, MaxHz: 20000, Norm: NormNone, Input: InputPower,
			Log: Log10, LogOffset: 1e-6, LogFloor: 1e-9,
		},
		"custom-bank": {
			SampleRate: 48000, FrameSize: 1024, HopSize: 256, Filterbank: fb,
			Input: InputPower, Log: LogNatural, LogOffset: 1e-8,
		},
		"win768-center": {
			SampleRate: 16000, FrameSize: 1024, WindowLength: 768, WindowAlign: stft.AlignCenter,
			HopSize: 256, NumMels: 40, MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
		},
		// The BSG-BAT shape has most rows narrower than dotProductMinWidth, so the
		// projector routes them through the scalar multiply-accumulate loop rather than
		// f32.DotProduct; this drives that path against the float64 reference.
		"bsgbat-narrow": {
			SampleRate: 384000, FrameSize: 1024, HopSize: 256, NumMels: 128,
			MinHz: 9000, MaxHz: 150000, Input: InputPower, Log: Log10, LogOffset: 1e-6,
		},
	}
}

// TestExtractorAgainstPlanReference validates Compute against an independent
// stft.Plan.PowerInto power reference projected through the filterbank in float64,
// for every config, over all three pad modes. The Analyzer path (RFFT+AbsSq) and
// the Plan path (fused RealFFTPower) agree to tolerance, not bit for bit.
func TestExtractorAgainstPlanReference(t *testing.T) {
	sig := toneSig(6000, 48000, []float64{1200, 4300, 9100}, 0.4)
	for name, cfg := range melTestConfigs(t) {
		for _, pad := range []stft.PadMode{stft.NoPad, stft.PadZero, stft.PadReflect} {
			t.Run(name+"/"+padName(pad), func(t *testing.T) {
				runExtractorCase(t, cfg, sig, pad)
			})
		}
	}
}

// runExtractorCase compares Compute against an stft.Plan.PowerInto power reference
// projected through the filterbank in float64, for one config and pad mode.
func runExtractorCase(t *testing.T, cfg Config, sig []float32, pad stft.PadMode) {
	t.Helper()
	ex, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := stft.New(cfg.stftConfig())
	if err != nil {
		t.Fatal(err)
	}
	bins := plan.NumBins()
	frames := plan.NumFrames(len(sig), pad)
	if got := ex.NumFrames(len(sig), pad); got != frames {
		t.Fatalf("NumFrames = %d, want %d (stft)", got, frames)
	}
	powerRef := make([]float32, bins*frames)
	if got := plan.PowerInto(powerRef, sig, pad); got != frames {
		t.Fatalf("PowerInto wrote %d frames, want %d", got, frames)
	}
	specRef := make([][]complex64, frames) // only InputReal reads the spectrum
	if cfg.Input == InputReal {
		for f := range specRef {
			specRef[f] = make([]complex64, bins)
		}
		if got := plan.Spectrum(specRef, sig, pad); got != frames {
			t.Fatalf("Spectrum wrote %d frames, want %d", got, frames)
		}
	}
	dense := ex.Filterbank().Dense()
	m := ex.Compute(sig, pad)
	if m.Frames != frames || m.Mels != ex.NumMels() {
		t.Fatalf("matrix shape = %dx%d, want %dx%d", m.Mels, m.Frames, ex.NumMels(), frames)
	}
	for f := range frames {
		src := refSource(cfg.Input, powerRef[f*bins:(f+1)*bins], specRef[f])
		lin, out := refMel(src, dense, cfg)
		compareMelColumn(t, m.Column(f), lin, out, cfg, f)
	}
}

// compareMelColumn checks one produced column against the reference: linear values
// within a peak-relative tolerance, log values within an absolute tolerance for
// bands above 40 dB down (floor-dominated bands are only checked for finiteness).
// The tolerances are cross-tier headroom over the measured worst case: the
// Analyzer path (RFFT+AbsSq) and the Plan.PowerInto reference (fused RealFFTPower)
// differ only by rounding order, and the measured worst deviation over these
// configs is ~9e-8 linear-relative and ~8e-7 log-absolute across both the amd64
// AVX+FMA and the SIMD_DISABLE=all pure-Go tiers. These tolerances keep roughly
// 100x margin (ample for arm64 NEON too) while still catching a sub-1e-5
// projection regression; the within-tier batch-versus-stream test is the exact
// guard.
const (
	linTol = 1e-5 // peak-relative, linear mel energies
	logTol = 1e-4 // absolute, log units

	// InputReal sums signed terms, so the float32 projection cancels against a
	// float64 reference by more than the non-negative inputs do: the worst
	// peak-relative deviation is ~7.9e-6 on amd64 AVX+FMA and ~1.2e-6 on the
	// SIMD_DISABLE=all tier, versus ~9e-8 for power and magnitude. The analyzer and
	// stft.Plan.Spectrum spectra are bit-identical here, so this is accumulation
	// error in the projection, not a transform difference.
	realLinTol = 1e-4

	// Power-law values are compared relative to the reference where |lin| is at
	// least powLawFloor of the column peak (40 dB down); below that x^q amplifies the
	// projection error of a near-zero x. The worst measured relative deviation of the
	// compared values is ~1.3e-4 on amd64 AVX+FMA and ~7.3e-5 on the SIMD_DISABLE=all
	// tier (real-powerlaw-custom, from the real-part accumulation error above, not
	// from Pow, whose fast path is ~1.4e-5).
	powTol      = 1e-3
	powLawFloor = 1e-2
)

func compareMelColumn(t *testing.T, col []float32, lin, out []float64, cfg Config, f int) {
	t.Helper()
	log := cfg.Log
	tol := linTol
	if cfg.Input == InputReal {
		tol = realLinTol
	}
	peak := 0.0
	for _, v := range lin {
		peak = math.Max(peak, math.Abs(v)) // |v|: InputReal energies are signed
	}
	for mm := range col {
		got := float64(col[mm])
		if log == LogNone {
			if math.Abs(got-lin[mm]) > tol*peak+1e-9 {
				t.Fatalf("frame %d mel %d: linear got %g want %g (peak %g)", f, mm, got, lin[mm], peak)
			}
			continue
		}
		if math.IsInf(got, 0) || math.IsNaN(got) {
			t.Fatalf("frame %d mel %d: non-finite compressed value %g", f, mm, got)
		}
		if log == PowerLaw {
			// x^q amplifies the relative error of a near-zero x, so bands under 40 dB
			// below the peak are only checked for finiteness and sign.
			if got < 0 {
				t.Fatalf("frame %d mel %d: power-law value %g is negative", f, mm, got)
			}
			if math.Abs(lin[mm]) < peak*powLawFloor {
				continue
			}
			if math.Abs(got-out[mm]) > powTol*out[mm] {
				t.Fatalf("frame %d mel %d: power-law got %g want %g", f, mm, got, out[mm])
			}
			continue
		}
		if lin[mm] < peak*1e-4 {
			continue
		}
		if math.Abs(got-out[mm]) > logTol {
			t.Fatalf("frame %d mel %d: log got %g want %g", f, mm, got, out[mm])
		}
	}
}

func padName(p stft.PadMode) string {
	switch p {
	case stft.PadZero:
		return "zero"
	case stft.PadReflect:
		return "reflect"
	default:
		return "nopad"
	}
}

// TestExtractorNumFramesMatchesStft pins mel.NumFrames to stft.Plan.NumFrames for
// every pad mode over a range of lengths, including empty and shorter-than-frame.
func TestExtractorNumFramesMatchesStft(t *testing.T) {
	cfg := Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, MinHz: 0, MaxHz: 0, Input: InputPower}
	ex, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := stft.New(cfg.stftConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, pad := range []stft.PadMode{stft.NoPad, stft.PadZero, stft.PadReflect} {
		for _, n := range []int{0, 1, 100, 255, 256, 257, 1000, 4096} {
			if got, want := ex.NumFrames(n, pad), plan.NumFrames(n, pad); got != want {
				t.Errorf("NumFrames(%d, %s) = %d, want %d", n, padName(pad), got, want)
			}
		}
	}
}

func TestExtractorShortSignalZeroFrames(t *testing.T) {
	ex, err := New(Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, Input: InputPower})
	if err != nil {
		t.Fatal(err)
	}
	// NoPad shorter than a frame, and empty signal in every mode, yield 0 frames.
	if m := ex.Compute(make([]float32, 100), stft.NoPad); m.Frames != 0 || len(m.Data) != 0 {
		t.Fatalf("short NoPad: frames %d len %d, want 0", m.Frames, len(m.Data))
	}
	for _, pad := range []stft.PadMode{stft.NoPad, stft.PadZero, stft.PadReflect} {
		if m := ex.Compute(nil, pad); m.Frames != 0 {
			t.Fatalf("empty signal %s: frames %d, want 0", padName(pad), m.Frames)
		}
	}
}

func TestExtractorSilenceFloor(t *testing.T) {
	// offset only: silence maps to log10(offset) exactly, finite.
	ex, err := New(Config{SampleRate: 16000, FrameSize: 256, HopSize: 256, NumMels: 20, MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6})
	if err != nil {
		t.Fatal(err)
	}
	m := ex.Compute(make([]float32, 4*256), stft.NoPad)
	want := math.Log10(float64(float32(1e-6)))
	for i, v := range m.Data {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Fatalf("index %d = %v, want finite floor", i, v)
		}
		if math.Abs(float64(v)-want) > 1e-3 {
			t.Fatalf("index %d = %g, want %g", i, v, want)
		}
	}
	// floor only: silence maps to ln(floor), finite.
	exf, err := New(Config{SampleRate: 16000, FrameSize: 256, HopSize: 256, NumMels: 20, MinHz: 0, MaxHz: 0, Input: InputPower, Log: LogNatural, LogFloor: 1e-10})
	if err != nil {
		t.Fatal(err)
	}
	mf := exf.Compute(make([]float32, 4*256), stft.NoPad)
	wantF := math.Log(float64(float32(1e-10)))
	for i, v := range mf.Data {
		if math.Abs(float64(v)-wantF) > 1e-3 {
			t.Fatalf("floor index %d = %g, want %g", i, v, wantF)
		}
	}
}

func TestExtractorComputeIntoBufferTooSmall(t *testing.T) {
	ex, err := New(Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, Input: InputPower})
	if err != nil {
		t.Fatal(err)
	}
	sig := toneSig(2048, 16000, []float64{500}, 0.5)
	small := Matrix{Data: make([]float32, 4)}
	if _, err := ex.ComputeInto(&small, sig, stft.NoPad); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	need := ex.NumMels() * ex.NumFrames(len(sig), stft.NoPad)
	exact := Matrix{Data: make([]float32, need)}
	got, err := ex.ComputeInto(&exact, sig, stft.NoPad)
	if err != nil || got != ex.NumFrames(len(sig), stft.NoPad) {
		t.Fatalf("ComputeInto = (%d, %v), want (%d, nil)", got, err, ex.NumFrames(len(sig), stft.NoPad))
	}
	if exact.Mels != ex.NumMels() || exact.Frames != got {
		t.Fatalf("matrix dims = %dx%d, want %dx%d", exact.Mels, exact.Frames, ex.NumMels(), got)
	}
}

func TestExtractorAccessors(t *testing.T) {
	ex, err := New(Config{SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40, MinHz: 500, MaxHz: 20000, Input: InputPower})
	if err != nil {
		t.Fatal(err)
	}
	if ex.FrameSize() != 1024 || ex.HopSize() != 256 {
		t.Errorf("FrameSize/HopSize = %d/%d, want 1024/256", ex.FrameSize(), ex.HopSize())
	}
	if ex.NumMels() != 40 {
		t.Errorf("NumMels = %d, want 40", ex.NumMels())
	}
	if ex.NumBins() != 1024/2+1 {
		t.Errorf("NumBins = %d, want %d", ex.NumBins(), 1024/2+1)
	}
	if ex.Filterbank() == nil || ex.Filterbank().NumMels() != 40 {
		t.Errorf("Filterbank() = %v, want a 40-row bank", ex.Filterbank())
	}
}

func TestMatrixAccessors(t *testing.T) {
	ex, err := New(Config{SampleRate: 16000, FrameSize: 256, HopSize: 128, NumMels: 20, Input: InputPower})
	if err != nil {
		t.Fatal(err)
	}
	sig := toneSig(2048, 16000, []float64{1000}, 0.5)
	m := ex.Compute(sig, stft.NoPad)
	// At(mel, frame) equals Column(frame)[mel].
	for f := range m.Frames {
		col := m.Column(f)
		for mm := range m.Mels {
			if m.At(mm, f) != col[mm] {
				t.Fatalf("At(%d,%d) = %g, Column mismatch %g", mm, f, m.At(mm, f), col[mm])
			}
		}
	}
}

// TestReflectIndex pins the ported numpy "reflect" fold directly, since the
// whole-clip reference tests use a signal far longer than FrameSize/2 and so only
// exercise the far side of the fold.
func TestReflectIndex(t *testing.T) {
	// n == 1: every index maps to the single sample.
	for _, idx := range []int{-3, -1, 0, 1, 5} {
		if got := reflectIndex(idx, 1); got != 0 {
			t.Errorf("reflectIndex(%d, 1) = %d, want 0", idx, got)
		}
	}
	// n == 4 (period 6): numpy reflect of [0 1 2 3], edge not repeated.
	want4 := map[int]int{-3: 3, -2: 2, -1: 1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 0, 7: 1, 8: 2}
	for idx, want := range want4 {
		if got := reflectIndex(idx, 4); got != want {
			t.Errorf("reflectIndex(%d, 4) = %d, want %d", idx, got, want)
		}
	}
	// Over several periods in both directions the result stays a valid index and
	// folds periodically with period 2*(n-1); a short signal padded by FrameSize/2
	// (half > n) exercises exactly this.
	for _, n := range []int{2, 4, 7} {
		period := 2 * (n - 1)
		for idx := -3 * period; idx <= 3*period; idx++ {
			r := reflectIndex(idx, n)
			if r < 0 || r >= n {
				t.Fatalf("reflectIndex(%d, %d) = %d, out of [0, %d)", idx, n, r, n)
			}
			reduced := ((idx % period) + period) % period
			if got := reflectIndex(reduced, n); got != r {
				t.Fatalf("reflectIndex(%d, %d) = %d not periodic (reduced %d = %d)", idx, n, r, reduced, got)
			}
		}
	}
}

// TestExtractorShortSignalCentered verifies the centered pad path for signals
// shorter than the frame, where the FrameSize/2 padding folds the reflection over
// multiple periods. It reuses the stft.Plan.PowerInto reference comparison.
func TestExtractorShortSignalCentered(t *testing.T) {
	cfg := Config{
		SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20,
		MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	}
	for _, n := range []int{40, 100, 200, 256} { // 40, 100 are shorter than FrameSize/2 = 128
		sig := toneSig(n, 16000, []float64{900, 3100}, 0.6)
		for _, pad := range []stft.PadMode{stft.PadZero, stft.PadReflect} {
			t.Run(fmt.Sprintf("n%d/%s", n, padName(pad)), func(t *testing.T) {
				runExtractorCase(t, cfg, sig, pad)
			})
		}
	}
}

func TestExtractorConfigValidation(t *testing.T) {
	fbBad, _ := FilterbankFromRows(customBankRows(10, 100)) // 100 bins != 256/2+1
	base := Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, MinHz: 0, MaxHz: 0, Input: InputPower}
	mut := func(f func(*Config)) Config {
		c := base
		f(&c)
		return c
	}
	cases := []struct {
		name string
		cfg  Config
	}{
		{"sample rate", mut(func(c *Config) { c.SampleRate = 0 })},
		{"frame size", mut(func(c *Config) { c.FrameSize = 100 })},
		{"hop too big", mut(func(c *Config) { c.HopSize = 999 })},
		{"num mels", mut(func(c *Config) { c.NumMels = 0 })},
		{"min hz negative", mut(func(c *Config) { c.MinHz = -1 })},
		{"max hz above nyquist", mut(func(c *Config) { c.MaxHz = 9000 })},
		{"min ge max", mut(func(c *Config) { c.MinHz = 5000; c.MaxHz = 4000 })},
		{"scale", mut(func(c *Config) { c.Scale = MelScale(9) })},
		{"norm", mut(func(c *Config) { c.Norm = Norm(9) })},
		{"input", mut(func(c *Config) { c.Input = Input(9) })},
		{"log", mut(func(c *Config) { c.Log = Log(9) })},
		{"log without offset or floor", mut(func(c *Config) { c.Log = Log10 })},
		{"log offset negative", mut(func(c *Config) { c.Log = Log10; c.LogOffset = -1 })},
		{"log floor nan", mut(func(c *Config) { c.Log = Log10; c.LogFloor = math.NaN() })},
		{"log offset underflows float32", mut(func(c *Config) { c.Log = Log10; c.LogOffset = 1e-50 })},
		{"log floor underflows float32", mut(func(c *Config) { c.Log = Log10; c.LogFloor = 1e-50 })},
		{"log offset overflows float32", mut(func(c *Config) { c.Log = Log10; c.LogOffset = 1e40 })},
		{"log floor overflows float32", mut(func(c *Config) { c.Log = Log10; c.LogFloor = 1e40 })},
		{"powerlaw exponent zero", mut(func(c *Config) { c.Log = PowerLaw })},
		{"powerlaw exponent negative", mut(func(c *Config) { c.Log = PowerLaw; c.PowerExponent = -0.5 })},
		{"powerlaw exponent above one", mut(func(c *Config) { c.Log = PowerLaw; c.PowerExponent = 1.5 })},
		{"powerlaw exponent nan", mut(func(c *Config) { c.Log = PowerLaw; c.PowerExponent = math.NaN() })},
		{"powerlaw exponent inf", mut(func(c *Config) { c.Log = PowerLaw; c.PowerExponent = math.Inf(1) })},
		{"powerlaw exponent underflows float32", mut(func(c *Config) { c.Log = PowerLaw; c.PowerExponent = 1e-50 })},
		{"exponent with log10", mut(func(c *Config) { c.Log = Log10; c.LogOffset = 1e-6; c.PowerExponent = 0.5 })},
		{"exponent without compression", mut(func(c *Config) { c.PowerExponent = 0.5 })},
		{"real input with log10", mut(func(c *Config) { c.Input = InputReal; c.Log = Log10; c.LogOffset = 1e-6 })},
		{"real input with natural log", mut(func(c *Config) { c.Input = InputReal; c.Log = LogNatural; c.LogFloor = 1e-6 })},
		{"custom bank bins mismatch", mut(func(c *Config) { c.Filterbank = fbBad })},
		{"window length too big", mut(func(c *Config) { c.WindowLength = 9999 })},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(tc.cfg); !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("New err = %v, want ErrInvalidConfig", err)
			}
			if _, err := NewColumnSource(tc.cfg); !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("NewColumnSource err = %v, want ErrInvalidConfig", err)
			}
		})
	}
}

// TestExtractorConfigAccepted pins the boundaries the rejection table above sits
// next to: the exponent upper bound, the narrowed offset-or-floor rule, and
// signed input without compression.
func TestExtractorConfigAccepted(t *testing.T) {
	base := Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, MinHz: 0, MaxHz: 0}
	cases := map[string]func(*Config){
		"powerlaw exponent one":          func(c *Config) { c.Log = PowerLaw; c.PowerExponent = 1 },
		"powerlaw without offset":        func(c *Config) { c.Input = InputMagnitude; c.Log = PowerLaw; c.PowerExponent = 0.35 },
		"real input without compression": func(c *Config) { c.Input = InputReal },
		"real input powerlaw": func(c *Config) {
			c.Input, c.Log, c.PowerExponent, c.LogOffset, c.LogFloor = InputReal, PowerLaw, 0.8, 1e-6, 1e-9
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := base
			mutate(&cfg)
			if _, err := New(cfg); err != nil {
				t.Fatalf("New err = %v, want nil", err)
			}
			if _, err := NewColumnSource(cfg); err != nil {
				t.Fatalf("NewColumnSource err = %v, want nil", err)
			}
		})
	}
}

// TestExtractorRealPowerLawHandDerived checks the real-part projection and the
// power-law stage against values derived by hand. With a rectangular window of
// length 64 the DFT of cos(2*pi*k0*n/64 + phi) is 32*exp(i*phi) at bin k0, so
// x[n] = cos(2*pi*4n/64 + pi/3) - cos(2*pi*9n/64) + 0.5*cos(2*pi*20n/64) has
// Re X[4] = 16 (|X[4]| = 32), Re X[9] = -32 and Re X[20] = 16. The bank puts
// weight 1 on bin 4 (row 0), 1 on bin 4 and 0.25 on bin 9 (row 1, interior zeros),
// 1 on bin 9 (row 2), nothing (row 3), and 1 over bins 10..30 (row 4, 21 wide so
// it takes the DotProduct branch).
func TestExtractorRealPowerLawHandDerived(t *testing.T) {
	const n = 64
	sig := make([]float32, n)
	for i := range sig {
		x := float64(i)
		sig[i] = float32(math.Cos(2*math.Pi*4*x/n+math.Pi/3) - math.Cos(2*math.Pi*9*x/n) + 0.5*math.Cos(2*math.Pi*20*x/n))
	}
	rows := make([][]float32, 5)
	for m := range rows {
		rows[m] = make([]float32, n/2+1)
	}
	rows[0][4] = 1
	rows[1][4], rows[1][9] = 1, 0.25
	rows[2][9] = 1
	for k := 10; k <= 30; k++ {
		rows[4][k] = 1
	}
	fb, err := FilterbankFromRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	base := Config{SampleRate: n, FrameSize: n, HopSize: n, CustomWindow: ones, Filterbank: fb}
	cases := []struct {
		name  string
		input Input
		log   Log
		exp   float64
		off   float64
		want  []float64
		tol   float64 // relative to max(1, |want|)
	}{
		{"real linear is signed", InputReal, LogNone, 0, 0, []float64{16, 8, -32, 0, 16}, 1e-3},
		{"real powerlaw takes abs", InputReal, PowerLaw, 0.5, 0, []float64{4, math.Sqrt(8), math.Sqrt(32), 0, 4}, 1e-4},
		{"real powerlaw offset after abs", InputReal, PowerLaw, 0.5, 1, []float64{math.Sqrt(17), 3, math.Sqrt(33), 1, math.Sqrt(17)}, 1e-4},
		{"power powerlaw", InputPower, PowerLaw, 0.5, 0, []float64{32, math.Sqrt(1280), 32, 0, 16}, 1e-4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.Input, cfg.Log, cfg.PowerExponent, cfg.LogOffset = tc.input, tc.log, tc.exp, tc.off
			ex, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			m := ex.Compute(sig, stft.NoPad)
			if m.Frames != 1 || m.Mels != 5 {
				t.Fatalf("matrix %dx%d, want 5x1", m.Mels, m.Frames)
			}
			for mm, want := range tc.want {
				got := float64(m.At(mm, 0))
				if !(math.Abs(got-want) <= tc.tol*math.Max(1, math.Abs(want))) { // fail-closed: NaN fails
					t.Errorf("mel %d = %g, want %g", mm, got, want)
				}
			}
		})
	}
}
