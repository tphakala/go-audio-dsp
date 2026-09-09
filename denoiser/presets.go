package denoiser

import (
	"fmt"
	"math"
)

// Estimator selects the per-bin gain rule. MMSELSA is the default and the
// recommended choice; the others exist for comparison and tuning.
type Estimator int

const (
	// MMSELSA is the Ephraim-Malah minimum mean-square error log-spectral
	// amplitude estimator with decision-directed a priori SNR: the smoothest
	// of the three, with the least musical noise.
	MMSELSA Estimator = iota
	// Wiener is the Wiener filter gain xi/(1+xi) on the decision-directed a
	// priori SNR.
	Wiener
	// Subtraction is plain power spectral subtraction, sqrt(max(1-1/gamma, 0)),
	// the classic baseline; it is the most aggressive and the most prone to
	// musical noise.
	Subtraction
)

// String returns the estimator name.
func (e Estimator) String() string {
	switch e {
	case MMSELSA:
		return "mmse-lsa"
	case Wiener:
		return "wiener"
	case Subtraction:
		return "subtraction"
	}
	return fmt.Sprintf("Estimator(%d)", int(e))
}

func (e Estimator) valid() bool { return e >= MMSELSA && e <= Subtraction }

// Preset names a tuned Params set. The zero value is Medium.
type Preset int

const (
	// Medium is the default: up to 12 dB of reduction.
	Medium Preset = iota
	// Light reduces noise by at most 6 dB and preserves the most detail.
	Light
	// Heavy reduces noise by up to 20 dB with aggressive over-subtraction; the
	// most reduction, best on steady broadband noise.
	Heavy
)

// String returns the preset name.
func (p Preset) String() string {
	switch p {
	case Medium:
		return "medium"
	case Light:
		return "light"
	case Heavy:
		return "heavy"
	}
	return fmt.Sprintf("Preset(%d)", int(p))
}

func (p Preset) valid() bool { return p >= Medium && p <= Heavy }

// Params holds every denoiser knob. Presets are named Params values; set
// Config.Params to override a preset entirely.
type Params struct {
	// MaxAttenuationDB is the residual gain floor in dB: no bin is ever
	// attenuated by more than this, so nothing gates to silence. It is the
	// analog of ffmpeg afftdn's "nr". 0 disables the denoiser (unity gain).
	MaxAttenuationDB float32
	// OverSubtraction multiplies the estimated noise power before the SNR
	// estimates (1 = none; >1 is more aggressive).
	OverSubtraction float32
	// SNRSmoothing is the decision-directed a priori SNR weight alpha in
	// [0,1): higher is smoother in time and less musical noise, lower tracks
	// transients faster.
	SNRSmoothing float32
	// MinPriorSNRDB floors the a priori SNR (dB); it bounds how aggressive the
	// gain can get in noise-only bins before the MaxAttenuationDB floor.
	MinPriorSNRDB float32
	// FreqSmoothBins is the width (odd; even values are rounded up) of a
	// moving average applied to the gain across bins; 0 or 1 disables it.
	FreqSmoothBins int
	// Estimator selects the gain rule.
	Estimator Estimator
	// TrackWindowSec is the adaptive tracker's minimum-search window in
	// seconds; sustained tones shorter than this are not absorbed into the
	// noise estimate.
	TrackWindowSec float32
}

// validate reports the first out-of-range field wrapped in ErrInvalidConfig.
func (p Params) validate() error {
	switch {
	case !(p.MaxAttenuationDB >= 0): // rejects NaN and negatives; +Inf is allowed (a bottomless floor, i.e. full gating)
		return fmt.Errorf("%w: MaxAttenuationDB must be >= 0, got %g", ErrInvalidConfig, p.MaxAttenuationDB)
	case math.IsNaN(float64(p.MinPriorSNRDB)) || math.IsInf(float64(p.MinPriorSNRDB), 0):
		return fmt.Errorf("%w: MinPriorSNRDB must be finite, got %g", ErrInvalidConfig, p.MinPriorSNRDB)
	case !(p.OverSubtraction > 0):
		return fmt.Errorf("%w: OverSubtraction must be > 0, got %g", ErrInvalidConfig, p.OverSubtraction)
	case !(p.SNRSmoothing >= 0 && p.SNRSmoothing < 1):
		return fmt.Errorf("%w: SNRSmoothing must be in [0,1), got %g", ErrInvalidConfig, p.SNRSmoothing)
	case p.FreqSmoothBins < 0:
		return fmt.Errorf("%w: FreqSmoothBins must be >= 0, got %d", ErrInvalidConfig, p.FreqSmoothBins)
	case !p.Estimator.valid():
		return fmt.Errorf("%w: unknown Estimator %d", ErrInvalidConfig, int(p.Estimator))
	case !(p.TrackWindowSec > 0):
		return fmt.Errorf("%w: TrackWindowSec must be > 0, got %g", ErrInvalidConfig, p.TrackWindowSec)
	}
	return nil
}

// presetParams are tuned against the synthetic quality harness (see
// quality_test.go); refinement against a real bird-clip corpus is future work.
var presetParams = [...]Params{
	Medium: {MaxAttenuationDB: 12, OverSubtraction: 1.2, SNRSmoothing: 0.96, MinPriorSNRDB: -18, FreqSmoothBins: 0, Estimator: MMSELSA, TrackWindowSec: 2},
	Light:  {MaxAttenuationDB: 6, OverSubtraction: 1.0, SNRSmoothing: 0.95, MinPriorSNRDB: -15, FreqSmoothBins: 0, Estimator: MMSELSA, TrackWindowSec: 2},
	Heavy:  {MaxAttenuationDB: 20, OverSubtraction: 2.0, SNRSmoothing: 0.97, MinPriorSNRDB: -22, FreqSmoothBins: 0, Estimator: MMSELSA, TrackWindowSec: 2},
}

// Params returns the preset's knob values (Medium's for an unknown preset).
func (p Preset) Params() Params {
	if !p.valid() {
		return presetParams[Medium]
	}
	return presetParams[p]
}
