package spectrogram

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/stft"

	simdf32 "github.com/tphakala/simd/f32"
)

// Scale selects how a frame's magnitude-squared power |X|^2 is mapped to an
// output value. The zero value is Magnitude.
type Scale int

const (
	// Magnitude is the window-normalized amplitude sqrt(|X|^2 * norm), the square
	// root of Power, so Magnitude and Power describe the same spectrum on
	// different axes.
	Magnitude Scale = iota
	// Power is the window-normalized power |X|^2 * norm, where norm = 1/sum(w^2)
	// removes the analysis window's energy so the value does not depend on the
	// window shape.
	Power
	// DB is the decibel power 10*log10(|X|^2 * norm) + GainDB, with a floor near
	// -200 dB so silence never yields negative infinity, optionally clamped to a
	// DynamicRangeDB window (see Config). Because the value is capped at GainDB
	// (0 dBFS shifted by the gain) when a dynamic range is set, a tone that falls
	// between two bins can reach the cap and be clipped there; that is expected.
	DB
)

// valid reports whether s is a defined scale.
func (s Scale) valid() bool { return s >= Magnitude && s <= DB }

// String returns the scale name.
func (s Scale) String() string {
	switch s {
	case Magnitude:
		return "magnitude"
	case Power:
		return "power"
	case DB:
		return "db"
	}
	return fmt.Sprintf("Scale(%d)", int(s))
}

// dbFloorPower is the floor applied to window-normalized power before the log in
// the DB scale, so an empty or near-silent bin maps to a finite ~-200 dB
// (10*log10(1e-20)) rather than negative infinity. It is far below any
// DynamicRangeDB a caller would set, so it never overrides the configured clamp.
const dbFloorPower = 1e-20

// Config configures a Spectrogram or ColumnSource: the short-time transform
// (shared with package stft), the value scale, and the frequency sub-range.
type Config struct {
	// SampleRate in Hz; required (> 0). Used only to map MinHz/MaxHz to bins and
	// to report BinHz; the transform itself is sample-rate agnostic.
	SampleRate int
	// FrameSize is the FFT length in samples, a power of two >= 2 (the stft
	// FrameSize). Named to match stft.Config.
	FrameSize int
	// HopSize is the frame advance in samples, in [1, FrameSize]. 0 selects
	// FrameSize/4 (75% overlap), matching stft.
	HopSize int
	// Window selects the analysis window; the zero value is Hann (periodic).
	Window stft.Window
	// Scale selects the output value mapping; the zero value is Magnitude.
	Scale Scale
	// GainDB is added to the DB scale (BirdNET-Go uses 3). It is the top of the
	// displayed dB range when DynamicRangeDB > 0. Ignored for Magnitude and Power.
	GainDB float64
	// DynamicRangeDB, when > 0, clamps the DB scale to [GainDB-DynamicRangeDB,
	// GainDB] (BirdNET-Go uses 100). 0 applies no clamp. Ignored for Magnitude
	// and Power. Must be >= 0.
	DynamicRangeDB float64
	// MinHz and MaxHz select the output frequency sub-range: rows cover the bins
	// whose center frequency lies in [MinHz, MaxHz]. MinHz 0 starts at DC. MaxHz
	// 0, or any value above Nyquist, is treated as Nyquist. MinHz must be < the
	// effective MaxHz.
	MinHz, MaxHz float64
}

// stftConfig projects the transform fields onto an stft.Config. FrameSize,
// HopSize and Window are validated by stft when the engine is built.
func (c Config) stftConfig() stft.Config {
	return stft.Config{FrameSize: c.FrameSize, HopSize: c.HopSize, Window: c.Window}
}

// validate checks the spectrogram-specific fields. FrameSize, HopSize and Window
// are left to stft.NewAnalyzer, which wraps the same ErrInvalidConfig.
func (c Config) validate() error {
	if c.SampleRate <= 0 {
		return fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, c.SampleRate)
	}
	if !c.Scale.valid() {
		return fmt.Errorf("%w: unknown Scale %d", ErrInvalidConfig, int(c.Scale))
	}
	if math.IsNaN(c.GainDB) || math.IsInf(c.GainDB, 0) {
		return fmt.Errorf("%w: GainDB must be finite, got %g", ErrInvalidConfig, c.GainDB)
	}
	if math.IsNaN(c.DynamicRangeDB) || math.IsInf(c.DynamicRangeDB, 0) || c.DynamicRangeDB < 0 {
		return fmt.Errorf("%w: DynamicRangeDB must be a finite value >= 0, got %g", ErrInvalidConfig, c.DynamicRangeDB)
	}
	if math.IsNaN(c.MinHz) || math.IsInf(c.MinHz, 0) || c.MinHz < 0 {
		return fmt.Errorf("%w: MinHz must be a finite value >= 0, got %g", ErrInvalidConfig, c.MinHz)
	}
	if math.IsNaN(c.MaxHz) {
		return fmt.Errorf("%w: MaxHz must not be NaN", ErrInvalidConfig)
	}
	return nil
}

// scaler maps a frame's magnitude-squared power (length NumBins, Analyzer-owned)
// to one output column over the selected bin sub-range [lo, hi), in the
// configured scale. It is shared by the whole-clip and streaming engines, so
// both produce identical columns.
type scaler struct {
	scale   Scale
	lo, hi  int     // output covers power[lo:hi]; hi-lo == bins
	binHz   float64 // frequency spacing of one bin, SampleRate/FrameSize
	norm    float32 // window-energy normalization, 1/sum(w^2)
	gainDB  float32
	clamp   bool
	clampLo float32
	clampHi float32
}

// bins returns the number of output rows (frequency bins) per column.
func (sc *scaler) bins() int { return sc.hi - sc.lo }

// newScaler resolves the bin sub-range and the window normalization for cfg
// against an already-built analyzer (its NumBins and resolved window).
func newScaler(cfg Config, a *stft.Analyzer) (scaler, error) {
	numBins := a.NumBins()
	binHz := float64(cfg.SampleRate) / float64(a.FrameSize())
	nyquist := float64(cfg.SampleRate) / 2

	maxHz := cfg.MaxHz
	if maxHz <= 0 || maxHz > nyquist {
		maxHz = nyquist
	}
	if cfg.MinHz >= maxHz {
		return scaler{}, fmt.Errorf("%w: MinHz %g must be < MaxHz %g (effective, clamped to Nyquist %g)", ErrInvalidConfig, cfg.MinHz, maxHz, nyquist)
	}

	// Include every bin whose center frequency lies in [MinHz, maxHz]. The small
	// epsilon absorbs floating-point rounding so an exact bin frequency is not
	// dropped by a hair.
	const eps = 1e-9
	lo := int(math.Ceil(cfg.MinHz/binHz - eps))
	hi := int(math.Floor(maxHz/binHz+eps)) + 1
	lo = max(lo, 0)
	hi = min(hi, numBins)
	if lo >= hi {
		return scaler{}, fmt.Errorf("%w: frequency range [%g, %g] Hz selects no bins at FrameSize %d, SampleRate %d", ErrInvalidConfig, cfg.MinHz, maxHz, a.FrameSize(), cfg.SampleRate)
	}

	var sumSq float64
	for _, w := range a.Window() {
		sumSq += float64(w) * float64(w)
	}
	norm := 0.0
	if sumSq > 0 {
		norm = 1 / sumSq
	}

	sc := scaler{
		scale:  cfg.Scale,
		lo:     lo,
		hi:     hi,
		binHz:  binHz,
		norm:   float32(norm),
		gainDB: float32(cfg.GainDB),
	}
	if cfg.Scale == DB && cfg.DynamicRangeDB > 0 {
		sc.clamp = true
		sc.clampHi = float32(cfg.GainDB)
		sc.clampLo = float32(cfg.GainDB - cfg.DynamicRangeDB)
	}
	return sc, nil
}

// buildEngine validates cfg and builds the pieces both producers share: a
// streaming stft.Analyzer and the resolved scaler. New and NewColumnSource call
// it so construction and validation live in one place.
func buildEngine(cfg Config) (*stft.Analyzer, scaler, error) {
	if err := cfg.validate(); err != nil {
		return nil, scaler{}, err
	}
	a, err := stft.NewAnalyzer(cfg.stftConfig())
	if err != nil {
		return nil, scaler{}, err
	}
	sc, err := newScaler(cfg, a)
	if err != nil {
		return nil, scaler{}, err
	}
	return a, sc, nil
}

// apply writes one output column into dst (length sc.bins()) from a frame's
// magnitude-squared power (length NumBins, so power[lo:hi] is the selected
// sub-range). dst and power must not alias. Every stage runs through simd, with
// scalar fallbacks under SIMD_DISABLE=all.
func (sc *scaler) apply(dst, power []float32) {
	src := power[sc.lo:sc.hi]
	switch sc.scale {
	case Power:
		simdf32.Scale(dst, src, sc.norm)
	case Magnitude:
		simdf32.Scale(dst, src, sc.norm)
		simdf32.Sqrt(dst, dst)
	case DB:
		simdf32.Scale(dst, src, sc.norm)                       // |X|^2 * norm
		simdf32.Clamp(dst, dst, dbFloorPower, math.MaxFloat32) // floor so log10 is finite
		simdf32.Log10(dst, dst)                                // log10(power)
		simdf32.Scale(dst, dst, 10)                            // 10*log10(power)
		simdf32.AddScalar(dst, dst, sc.gainDB)                 // + GainDB
		if sc.clamp {
			simdf32.Clamp(dst, dst, sc.clampLo, sc.clampHi)
		}
	}
}

// binHzOf returns the center frequency in Hz of output row i (0-based within the
// selected sub-range).
func (sc *scaler) binHzOf(i int) float64 { return float64(sc.lo+i) * sc.binHz }
