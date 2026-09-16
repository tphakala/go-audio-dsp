package spectrogram

import (
	"errors"
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// naiveColumn computes the reference column for frame f of signal under NoPad
// framing: a plain float64 DFT of the windowed frame, magnitude-squared, then
// window-energy normalized, for every bin b in [0, numBins). It is an
// independent oracle (no simd, no FFT) for the whole pipeline: framing, power,
// and normalization. The window is taken from stft.GenerateWindow so only the
// transform-and-scale math is under test, not window generation (stft tests
// that).
func naiveColumn(signal, w []float32, f, hop int) []float64 {
	n := len(w)
	numBins := n/2 + 1
	var sumSq float64
	for _, wi := range w {
		sumSq += float64(wi) * float64(wi)
	}
	norm := 1 / sumSq
	out := make([]float64, numBins)
	base := f * hop
	for b := range numBins {
		var re, im float64
		for k := range n {
			x := float64(signal[base+k]) * float64(w[k])
			ang := -2 * math.Pi * float64(b) * float64(k) / float64(n)
			re += x * math.Cos(ang)
			im += x * math.Sin(ang)
		}
		out[b] = (re*re + im*im) * norm
	}
	return out
}

func sine(n, sampleRate int, freq, amp float64) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(sampleRate)))
	}
	return s
}

// TestComputeAgainstNaiveDFT validates every scale against the independent
// naive-DFT reference over the full frequency range.
func TestComputeAgainstNaiveDFT(t *testing.T) {
	const sr, n, hop = 16000, 512, 128
	sig := sine(4096, sr, 1000, 0.8)
	// Add a second partial so the spectrum is not a single spike.
	for i, v := range sine(4096, sr, 3200, 0.3) {
		sig[i] += v
	}
	w := stft.GenerateWindow(stft.Hann, n)

	for _, sc := range []Scale{Magnitude, Power, DB} {
		t.Run(sc.String(), func(t *testing.T) {
			s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: sc})
			if err != nil {
				t.Fatal(err)
			}
			m := s.Compute(sig)
			if m.Frames != s.NumFrames(len(sig)) || m.Bins != n/2+1 {
				t.Fatalf("matrix shape = %dx%d, want %dx%d", m.Bins, m.Frames, n/2+1, s.NumFrames(len(sig)))
			}
			for f := range m.Frames {
				ref := naiveColumn(sig, w, f, hop)
				peakPow := 0.0
				for _, p := range ref {
					peakPow = math.Max(peakPow, p)
				}
				for b := range m.Bins {
					got := float64(m.At(b, f))
					switch sc {
					case Power:
						// Peak-relative tolerance: float32 FFT noise is a small
						// fraction of the column peak, so comparing per bin against
						// its own tiny value would fail at spectral nulls.
						if math.Abs(got-ref[b]) > 1e-3*peakPow {
							t.Fatalf("power frame %d bin %d: got %g, want %g (peak %g)", f, b, got, ref[b], peakPow)
						}
					case Magnitude:
						want := math.Sqrt(ref[b])
						if math.Abs(got-want) > 1e-3*math.Sqrt(peakPow) {
							t.Fatalf("magnitude frame %d bin %d: got %g, want %g", f, b, got, want)
						}
					case DB:
						// dB is only meaningful above the float32 FFT noise floor;
						// compare bins within 40 dB of the column peak.
						if ref[b] < peakPow*1e-4 {
							continue
						}
						want := 10 * math.Log10(math.Max(ref[b], dbFloorPower))
						if math.Abs(got-want) > 0.2 {
							t.Fatalf("db frame %d bin %d: got %g, want %g", f, b, got, want)
						}
					}
				}
			}
		})
	}
}

// TestSinePeakBin checks that a tone at a bin-center frequency peaks in that bin.
func TestSinePeakBin(t *testing.T) {
	const sr, n, hop = 16000, 1024, 256
	const bin = 40
	freq := float64(bin) * sr / n // exactly bin-centered
	sig := sine(8192, sr, freq, 0.9)

	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: Power})
	if err != nil {
		t.Fatal(err)
	}
	m := s.Compute(sig)
	col := m.Column(m.Frames / 2) // a steady-state frame
	peak, peakVal := 0, float32(-1)
	for b, v := range col {
		if v > peakVal {
			peak, peakVal = b, v
		}
	}
	if peak != bin {
		t.Errorf("peak at bin %d, want %d", peak, bin)
	}
	// Bin frequency reported by BinHz must match the tone.
	if got := s.BinHz(peak); math.Abs(got-freq) > 1e-6 {
		t.Errorf("BinHz(%d) = %g, want %g", peak, got, freq)
	}
}

// TestImpulseFlat checks that a single-sample impulse yields a near-flat
// magnitude spectrum (its DFT magnitude is constant across bins).
func TestImpulseFlat(t *testing.T) {
	const sr, n = 8000, 256
	sig := make([]float32, n)
	sig[10] = 1
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: n, Scale: Magnitude})
	if err != nil {
		t.Fatal(err)
	}
	m := s.Compute(sig)
	if m.Frames != 1 {
		t.Fatalf("frames = %d, want 1", m.Frames)
	}
	col := m.Column(0)
	lo, hi := col[0], col[0]
	for _, v := range col {
		lo, hi = min(lo, v), max(hi, v)
	}
	if lo <= 0 || hi/lo > 1.0001 {
		t.Errorf("impulse spectrum not flat: min %g max %g ratio %g", lo, hi, hi/lo)
	}
}

// TestDBFloorAndClamp checks silence maps to the floor and the clamp window
// bounds the DB values.
func TestDBFloorAndClamp(t *testing.T) {
	const sr, n = 16000, 256
	silent := make([]float32, 4*n)

	// No clamp: silence maps to the ~-200 dB floor plus gain.
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: n, Scale: DB, GainDB: 3})
	if err != nil {
		t.Fatal(err)
	}
	m := s.Compute(silent)
	wantFloor := 10*math.Log10(dbFloorPower) + 3
	for _, v := range m.Data {
		if math.Abs(float64(v)-wantFloor) > 1e-3 {
			t.Fatalf("silent DB = %g, want floor %g", v, wantFloor)
		}
	}

	// With a clamp, silence is pinned to the low end and a loud tone cannot
	// exceed the high end.
	sc, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: n, Scale: DB, GainDB: 3, DynamicRangeDB: 100})
	if err != nil {
		t.Fatal(err)
	}
	loud := sine(4*n, sr, 1000, 5) // clips well above 0 dBFS
	mc := sc.Compute(loud)
	for _, v := range mc.Data {
		if v < 3-100-1e-3 || v > 3+1e-3 {
			t.Fatalf("clamped DB = %g, want within [%g, %g]", v, 3.0-100, 3.0)
		}
	}
	ms := sc.Compute(silent)
	for _, v := range ms.Data {
		if math.Abs(float64(v)-(3-100)) > 1e-3 {
			t.Fatalf("clamped silent DB = %g, want %g", v, 3.0-100)
		}
	}
}

// TestDBZeroEnergyWindowFinite guards the DB fold against a degenerate zero-energy
// analysis window. HannSymmetric at FrameSize 2 is all zeros, so sum(w^2) == 0 and
// the window-energy norm is 0; the fold must degenerate to no normalization and
// still floor every bin to a finite value, not 10*log10(0) = -Inf.
func TestDBZeroEnergyWindowFinite(t *testing.T) {
	const sr, n = 16000, 2
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: n, Scale: DB, GainDB: 3, Window: stft.HannSymmetric})
	if err != nil {
		t.Fatal(err)
	}
	m := s.Compute(make([]float32, 32*n))
	wantFloor := 10*math.Log10(dbFloorPower) + 3
	for i, v := range m.Data {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Fatalf("bin %d = %v, want finite floor %g", i, v, wantFloor)
		}
		if math.Abs(float64(v)-wantFloor) > 1e-3 {
			t.Fatalf("bin %d = %g, want floor %g", i, v, wantFloor)
		}
	}
}

func TestSubRangeBins(t *testing.T) {
	const sr, n, hop = 48000, 1024, 256
	// binHz = 48000/1024 = 46.875. [1000, 5000] Hz -> bins ceil(1000/46.875)=22
	// .. floor(5000/46.875)=106, inclusive -> 22..106 -> 85 bins.
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, MinHz: 1000, MaxHz: 5000})
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Bins(); got != 85 {
		t.Errorf("Bins() = %d, want 85", got)
	}
	binHz := float64(sr) / n
	if got, want := s.BinHz(0), 22*binHz; math.Abs(got-want) > 1e-9 {
		t.Errorf("BinHz(0) = %g, want %g", got, want)
	}
	// MaxHz 0 means Nyquist: full range.
	full, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	if got := full.Bins(); got != n/2+1 {
		t.Errorf("full Bins() = %d, want %d", got, n/2+1)
	}
	// A positive MaxHz above Nyquist is accepted and clamped to Nyquist, so it
	// selects the full range too.
	above, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, MaxHz: 1e9})
	if err != nil {
		t.Fatalf("MaxHz above Nyquist: New err = %v, want nil", err)
	}
	if got := above.Bins(); got != n/2+1 {
		t.Errorf("MaxHz above Nyquist Bins() = %d, want %d", got, n/2+1)
	}
}

func TestComputeIntoBufferTooSmall(t *testing.T) {
	const sr, n, hop = 16000, 256, 64
	s, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: Power})
	if err != nil {
		t.Fatal(err)
	}
	sig := sine(2048, sr, 500, 0.5)
	small := Matrix{Data: make([]float32, 4)}
	if _, err := s.ComputeInto(&small, sig); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	// Exactly sized capacity works and reslices Data.
	need := s.Bins() * s.NumFrames(len(sig))
	exact := Matrix{Data: make([]float32, need)}
	if got, err := s.ComputeInto(&exact, sig); err != nil || got != s.NumFrames(len(sig)) {
		t.Fatalf("ComputeInto = (%d, %v), want (%d, nil)", got, err, s.NumFrames(len(sig)))
	}
}

func TestNumFramesAndHopForWidth(t *testing.T) {
	const n, hop = 512, 128
	s, err := New(Config{SampleRate: 16000, FrameSize: n, HopSize: hop, Scale: Power})
	if err != nil {
		t.Fatal(err)
	}
	if s.FrameSize() != n || s.HopSize() != hop {
		t.Errorf("FrameSize/HopSize = %d/%d, want %d/%d", s.FrameSize(), s.HopSize(), n, hop)
	}
	if got := s.NumFrames(n - 1); got != 0 {
		t.Errorf("NumFrames(short) = %d, want 0", got)
	}
	if got := s.NumFrames(n); got != 1 {
		t.Errorf("NumFrames(n) = %d, want 1", got)
	}
	if got := s.NumFrames(n + 3*hop); got != 4 {
		t.Errorf("NumFrames = %d, want 4", got)
	}
	// HopForWidth: a hop that spans ~columns frames.
	sigLen := 100000
	hw := HopForWidth(sigLen, n, 200)
	frames := 1 + (sigLen-n)/hw
	if frames < 199 || frames > 201 {
		t.Errorf("HopForWidth gives %d frames, want ~200", frames)
	}
	if got := HopForWidth(10, 512, 200); got != 512 {
		t.Errorf("HopForWidth(short) = %d, want 512", got)
	}
	if got := HopForWidth(100000, 512, 1); got != 512 {
		t.Errorf("HopForWidth(1 col) = %d, want 512", got)
	}
	// Far more columns than samples: the ideal hop rounds below 1 and clamps up.
	if got := HopForWidth(100000, 512, 500000); got != 1 {
		t.Errorf("HopForWidth(too many cols) = %d, want 1 (clamped)", got)
	}
	// A very long signal with few columns wants a hop above FrameSize, which is
	// clamped down to FrameSize (the max valid hop).
	if got := HopForWidth(1_000_000, 512, 2); got != 512 {
		t.Errorf("HopForWidth(long, 2 cols) = %d, want 512 (clamped)", got)
	}
}

func TestConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"sample rate", Config{SampleRate: 0, FrameSize: 256}},
		{"scale", Config{SampleRate: 16000, FrameSize: 256, Scale: Scale(99)}},
		{"dynamic range", Config{SampleRate: 16000, FrameSize: 256, Scale: DB, DynamicRangeDB: -1}},
		{"min hz negative", Config{SampleRate: 16000, FrameSize: 256, MinHz: -1}},
		{"gain nan", Config{SampleRate: 16000, FrameSize: 256, Scale: DB, GainDB: math.NaN()}},
		{"gain inf", Config{SampleRate: 16000, FrameSize: 256, Scale: DB, GainDB: math.Inf(1)}},
		{"dynamic range nan", Config{SampleRate: 16000, FrameSize: 256, Scale: DB, DynamicRangeDB: math.NaN()}},
		{"min hz nan", Config{SampleRate: 16000, FrameSize: 256, MinHz: math.NaN()}},
		{"max hz nan", Config{SampleRate: 16000, FrameSize: 256, MaxHz: math.NaN()}},
		{"max hz negative", Config{SampleRate: 16000, FrameSize: 256, MaxHz: -1}},
		{"min ge max", Config{SampleRate: 16000, FrameSize: 256, MinHz: 5000, MaxHz: 4000}},
		// binHz = 62.5; [7010, 7040] falls between bin centers 7000 and 7062.5, so
		// it selects no bin.
		{"empty bin range", Config{SampleRate: 16000, FrameSize: 256, MinHz: 7010, MaxHz: 7040}},
		{"frame size", Config{SampleRate: 16000, FrameSize: 100}}, // not power of two, stft rejects
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

func TestScaleString(t *testing.T) {
	for s, want := range map[Scale]string{Magnitude: "magnitude", Power: "power", DB: "db", Scale(9): "Scale(9)"} {
		if got := s.String(); got != want {
			t.Errorf("Scale(%d).String() = %q, want %q", int(s), got, want)
		}
	}
}
