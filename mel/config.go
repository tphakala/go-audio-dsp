package mel

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/stft"

	simdf32 "github.com/tphakala/simd/f32"
)

// MelScale selects the Hz-to-mel mapping used to place the filters. The zero
// value is Slaney, librosa's default (htk=False).
type MelScale int

const (
	// Slaney is librosa's default mel scale (htk=False): linear below 1 kHz, log
	// above. See HzToMel for the exact breakpoints.
	Slaney MelScale = iota
	// HTK is the HTK mel scale, 2595 * log10(1 + f/700), log everywhere.
	HTK
)

// valid reports whether s is a defined scale.
func (s MelScale) valid() bool { return s >= Slaney && s <= HTK }

// String returns the scale name.
func (s MelScale) String() string {
	switch s {
	case Slaney:
		return "slaney"
	case HTK:
		return "htk"
	}
	return fmt.Sprintf("MelScale(%d)", int(s))
}

// Norm selects the filter weight normalization. The zero value is NormSlaney,
// librosa's default (norm="slaney").
type Norm int

const (
	// NormSlaney scales each triangle by 2/(f[m+2]-f[m]) so equal-energy input
	// across a band yields equal output regardless of the band's width in Hz
	// (librosa norm="slaney"). It applies to either MelScale.
	NormSlaney Norm = iota
	// NormNone leaves unit-peak triangles (librosa norm=None).
	NormNone
)

// valid reports whether n is a defined normalization.
func (n Norm) valid() bool { return n >= NormSlaney && n <= NormNone }

// String returns the normalization name.
func (n Norm) String() string {
	switch n {
	case NormSlaney:
		return "slaney"
	case NormNone:
		return "none"
	}
	return fmt.Sprintf("Norm(%d)", int(n))
}

// Input selects what the filterbank projects: the power spectrum |X|^2 or the
// magnitude spectrum |X|. The zero value is InputPower (librosa power=2.0).
type Input int

const (
	// InputPower projects the power spectrum |X|^2 (librosa power=2.0).
	InputPower Input = iota
	// InputMagnitude projects the magnitude spectrum |X| (librosa power=1.0).
	InputMagnitude
)

// valid reports whether i is a defined input.
func (i Input) valid() bool { return i >= InputPower && i <= InputMagnitude }

// String returns the input name.
func (i Input) String() string {
	switch i {
	case InputPower:
		return "power"
	case InputMagnitude:
		return "magnitude"
	}
	return fmt.Sprintf("Input(%d)", int(i))
}

// Log selects the compression applied to each mel value. The zero value is
// LogNone (linear mel energies), keeping the log stage opt-in and explicit.
type Log int

const (
	// LogNone leaves the mel energies linear.
	LogNone Log = iota
	// Log10 applies log10(max(x + LogOffset, LogFloor)).
	Log10
	// LogNatural applies ln(max(x + LogOffset, LogFloor)).
	LogNatural
)

// valid reports whether l is a defined log mode.
func (l Log) valid() bool { return l >= LogNone && l <= LogNatural }

// String returns the log-mode name.
func (l Log) String() string {
	switch l {
	case LogNone:
		return "none"
	case Log10:
		return "log10"
	case LogNatural:
		return "natural"
	}
	return fmt.Sprintf("Log(%d)", int(l))
}

// Config configures an Extractor or ColumnSource: the short-time transform
// (shared with package stft), the mel filterbank, the power/magnitude input, and
// an optional log stage. The zero values of Scale, Norm and Input reproduce
// librosa's melspectrogram defaults (Slaney scale, slaney norm, power input).
type Config struct {
	// SampleRate in Hz; required (> 0). Places the mel filters and validates
	// MinHz/MaxHz against Nyquist.
	SampleRate int

	// Transform fields, named to match stft.Config and validated by it.
	FrameSize    int              // FFT length, a power of two >= 2
	WindowLength int              // analysis window length in [1, FrameSize]; 0 means FrameSize
	WindowAlign  stft.WindowAlign // placement of a shorter window; zero value AlignCenter
	HopSize      int              // frame advance; 0 means FrameSize/4
	Window       stft.Window      // built-in window; zero value Hann (periodic)
	CustomWindow []float32        // exact window of length WindowLength; overrides Window

	// Filterbank generation. Ignored when Filterbank is non-nil.
	NumMels int     // number of mel bands, > 0
	MinHz   float64 // lowest filter edge, >= 0
	MaxHz   float64 // highest filter edge; 0 means Nyquist; must be <= Nyquist and > MinHz
	Scale   MelScale
	Norm    Norm

	// Filterbank, when non-nil, is a precomputed bank (FilterbankFromRows or
	// NewFilterbank) used instead of generating one. Its NumBins must equal
	// FrameSize/2 + 1. It is read-only and may be shared between instances.
	Filterbank *Filterbank

	// Input selects power or magnitude projection.
	Input Input

	// Log stage. When Log != LogNone, at least one of LogOffset and LogFloor must
	// be > 0 so silence never yields -Inf: y = log(max(x + LogOffset, LogFloor)).
	Log       Log
	LogOffset float64 // added before the log (BSG-BAT: 1e-6)
	LogFloor  float64 // lower clamp before the log (0 means none)
}

// stftConfig projects the transform fields onto an stft.Config; stft validates
// them when the engine is built.
func (c Config) stftConfig() stft.Config {
	return stft.Config{
		FrameSize:    c.FrameSize,
		HopSize:      c.HopSize,
		Window:       c.Window,
		CustomWindow: c.CustomWindow,
		WindowLength: c.WindowLength,
		WindowAlign:  c.WindowAlign,
	}
}

// validate checks the mel-specific scalar fields, wrapping ErrInvalidConfig with
// the first offending field. The transform fields are left to stft.NewAnalyzer;
// the band fields (NumMels, MinHz, MaxHz, Scale, Norm) are validated by
// NewFilterbank in buildEngine when generating a bank, and a custom Filterbank's
// bin count is checked in buildEngine against the analyzer.
func (c Config) validate() error {
	if c.SampleRate <= 0 {
		return fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, c.SampleRate)
	}
	if !c.Input.valid() {
		return fmt.Errorf("%w: unknown Input %d", ErrInvalidConfig, int(c.Input))
	}
	if !c.Log.valid() {
		return fmt.Errorf("%w: unknown Log %d", ErrInvalidConfig, int(c.Log))
	}
	if math.IsNaN(c.LogOffset) || math.IsInf(c.LogOffset, 0) || c.LogOffset < 0 {
		return fmt.Errorf("%w: LogOffset must be a finite value >= 0, got %g", ErrInvalidConfig, c.LogOffset)
	}
	if math.IsNaN(c.LogFloor) || math.IsInf(c.LogFloor, 0) || c.LogFloor < 0 {
		return fmt.Errorf("%w: LogFloor must be a finite value >= 0, got %g", ErrInvalidConfig, c.LogFloor)
	}
	// The projector applies the offset and floor as float32 (buildEngine casts them
	// once). A finite positive float64 that overflows to +Inf or underflows to 0 in
	// float32 would defeat the guard and let apply emit a non-finite column, so
	// validate the converted values the projector actually uses, not just the float64
	// inputs.
	offset32, floor32 := float32(c.LogOffset), float32(c.LogFloor)
	if math.IsInf(float64(offset32), 0) {
		return fmt.Errorf("%w: LogOffset %g overflows to non-finite in float32", ErrInvalidConfig, c.LogOffset)
	}
	if math.IsInf(float64(floor32), 0) {
		return fmt.Errorf("%w: LogFloor %g overflows to non-finite in float32", ErrInvalidConfig, c.LogFloor)
	}
	if c.Log != LogNone && offset32 <= 0 && floor32 <= 0 {
		return fmt.Errorf("%w: Log requires LogOffset or LogFloor to stay > 0 in float32 so silence never yields -Inf, got LogOffset %g, LogFloor %g", ErrInvalidConfig, c.LogOffset, c.LogFloor)
	}
	return nil
}

// projector maps one frame's power (len NumBins, Analyzer-owned) to one mel
// column (len NumMels). Shared by Extractor and ColumnSource so both produce
// identical columns.
type projector struct {
	fb     *Filterbank
	input  Input
	mag    []float32 // sqrt scratch, len NumBins, only for InputMagnitude
	log    Log
	offset float32 // LogOffset cast once
	floor  float32 // LogFloor cast once
}

// apply writes one mel column into dst (len NumMels) from a frame's power (len
// NumBins, Analyzer-owned). dst and power must not alias. Every arithmetic pass
// runs through simd with scalar fallbacks under SIMD_DISABLE=all. Allocation-free.
func (p *projector) apply(dst, power []float32) {
	src := power
	if p.input == InputMagnitude {
		simdf32.Sqrt(p.mag, power) // |X| = sqrt(|X|^2)
		src = p.mag
	}
	// Sparse per-row dot product: each row multiplies only its contiguous nonzero
	// weights against the matching bins. An empty row (n == 0) reads empty spans
	// and DotProduct returns 0.
	for m := range p.fb.rows {
		r := p.fb.rows[m]
		dst[m] = simdf32.DotProduct(p.fb.weights[r.off:r.off+r.n], src[r.start:r.start+r.n])
	}
	if p.log == LogNone {
		return
	}
	// y = log(max(x + offset, floor)). The mel energy x is >= 0 (power and
	// magnitude are >= 0, and every filterbank weight is non-negative: generated
	// banks build non-negative triangles and FilterbankFromRows rejects negative
	// weights), and validation guarantees offset > 0 or floor > 0, so the argument
	// to log is always > 0. Either guard may be absent.
	if p.offset > 0 {
		simdf32.AddScalar(dst, dst, p.offset)
	}
	if p.floor > 0 {
		simdf32.Clamp(dst, dst, p.floor, math.MaxFloat32)
	}
	switch p.log {
	case Log10:
		simdf32.Log10(dst, dst)
	case LogNatural:
		simdf32.Log(dst, dst)
	case LogNone:
		// unreachable: returned above
	}
}

// buildEngine validates cfg and builds the pieces both producers share: a
// streaming stft.Analyzer and the resolved projector (with its filterbank). New
// and NewColumnSource call it so construction and validation live in one place.
func buildEngine(cfg Config) (*stft.Analyzer, projector, error) {
	if err := cfg.validate(); err != nil {
		return nil, projector{}, err
	}
	a, err := stft.NewAnalyzer(cfg.stftConfig())
	if err != nil {
		return nil, projector{}, err
	}
	fb := cfg.Filterbank
	switch {
	case fb == nil:
		fb, err = NewFilterbank(FilterbankConfig{
			SampleRate: cfg.SampleRate,
			FrameSize:  cfg.FrameSize,
			NumMels:    cfg.NumMels,
			MinHz:      cfg.MinHz,
			MaxHz:      cfg.MaxHz,
			Scale:      cfg.Scale,
			Norm:       cfg.Norm,
		})
		if err != nil {
			return nil, projector{}, err
		}
	case fb.NumBins() != a.NumBins():
		return nil, projector{}, fmt.Errorf("%w: Filterbank.NumBins %d must equal FrameSize/2+1 = %d", ErrInvalidConfig, fb.NumBins(), a.NumBins())
	}
	pr := projector{
		fb:     fb,
		input:  cfg.Input,
		log:    cfg.Log,
		offset: float32(cfg.LogOffset),
		floor:  float32(cfg.LogFloor),
	}
	if cfg.Input == InputMagnitude {
		pr.mag = make([]float32, a.NumBins())
	}
	return a, pr, nil
}
