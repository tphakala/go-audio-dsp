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

// refMel projects one frame's power (|X|^2 over NumBins) through the dense
// filterbank and applies the input and log stages in float64, mirroring
// projector.apply. It returns the linear mel energies (pre-log) and the final
// column, so a test can gate the log comparison on the 40 dB-down floor.
func refMel(power []float32, dense [][]float32, input Input, log Log, offset, floor float32) (lin, out []float64) {
	numBins := len(power)
	src := make([]float64, numBins)
	for k := range src {
		p := float64(power[k])
		if input == InputMagnitude {
			p = math.Sqrt(p)
		}
		src[k] = p
	}
	lin = make([]float64, len(dense))
	out = make([]float64, len(dense))
	for m := range dense {
		var acc float64
		row := dense[m]
		for k := range numBins {
			acc += float64(row[k]) * src[k]
		}
		lin[m] = acc
		val := acc
		if log != LogNone {
			if offset > 0 {
				val += float64(offset)
			}
			if floor > 0 {
				val = math.Max(val, float64(floor))
			}
			switch log {
			case Log10:
				val = math.Log10(val)
			case LogNatural:
				val = math.Log(val)
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
		// projector routes them through the scalar fused loop rather than f32.DotProduct;
		// this drives that path against the float64 reference.
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
	dense := ex.Filterbank().Dense()
	m := ex.Compute(sig, pad)
	if m.Frames != frames || m.Mels != ex.NumMels() {
		t.Fatalf("matrix shape = %dx%d, want %dx%d", m.Mels, m.Frames, ex.NumMels(), frames)
	}
	for f := range frames {
		lin, out := refMel(powerRef[f*bins:(f+1)*bins], dense, cfg.Input, cfg.Log, float32(cfg.LogOffset), float32(cfg.LogFloor))
		compareMelColumn(t, m.Column(f), lin, out, cfg.Log, f)
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
)

func compareMelColumn(t *testing.T, col []float32, lin, out []float64, log Log, f int) {
	t.Helper()
	peak := 0.0
	for _, v := range lin {
		peak = math.Max(peak, v)
	}
	for mm := range col {
		got := float64(col[mm])
		if log == LogNone {
			if math.Abs(got-lin[mm]) > linTol*peak+1e-9 {
				t.Fatalf("frame %d mel %d: linear got %g want %g (peak %g)", f, mm, got, lin[mm], peak)
			}
			continue
		}
		if math.IsInf(got, 0) || math.IsNaN(got) {
			t.Fatalf("frame %d mel %d: non-finite log value %g", f, mm, got)
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
