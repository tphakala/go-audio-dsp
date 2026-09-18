package gate

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// Params holds every knob of the soft-gate method. A Strength selects a tuned
// Params set (see ParamsFor); set Config.Params to override those knobs entirely.
type Params struct {
	// MaxAttenuationDB is the residual gain floor in dB: no bin is attenuated by
	// more than this, so nothing gates to digital silence. 0 disables the gate
	// (unity gain), +Inf allows full gating (floor 0). NaN and negatives are
	// rejected. It is the analog of ffmpeg afftdn's "nr" and follows the flagship's
	// 6/12/20 per strength.
	MaxAttenuationDB float32
	// ThresholdDB is the gate threshold above the per-bin noise floor, in dB: the
	// k-sigma margin at which a bin is half open. It must be finite.
	ThresholdDB float32
	// TransitionDB is the soft-knee width in dB over which a bin goes from mostly
	// closed to mostly open. It must be finite and > 0.
	TransitionDB float32
	// FreqSmoothBins is the width (odd; an even value rounds up to the next odd
	// width, since the half-width is FreqSmoothBins/2 and the box spans
	// 2*half+1) of a centered moving average of the gain across bins; 0 or 1
	// disables it. Smoothing across frequency suppresses isolated musical-noise
	// bins.
	FreqSmoothBins int
	// TimeSmoothFrames is the width 2L+1 of a centered moving average of the gain
	// across frames (odd; an even value rounds up to the next odd effective width,
	// since L = TimeSmoothFrames/2); 0 or 1 disables it. L frames of lookahead are
	// added to Latency(), because the memoryless sigmoid would otherwise flutter
	// frame to frame.
	TimeSmoothFrames int
	// FloorWindowSec is the blind rolling-median window in seconds, used only
	// while no floor is learned. It must be finite and > 0.
	FloorWindowSec float32
}

// validate reports the first out-of-range field wrapped in ErrInvalidConfig.
func (p Params) validate() error {
	switch {
	case !(p.MaxAttenuationDB >= 0): // rejects NaN and negatives; +Inf is allowed (full gating)
		return fmt.Errorf("%w: MaxAttenuationDB must be >= 0, got %g", ErrInvalidConfig, p.MaxAttenuationDB)
	case math.IsNaN(float64(p.ThresholdDB)) || math.IsInf(float64(p.ThresholdDB), 0):
		return fmt.Errorf("%w: ThresholdDB must be finite, got %g", ErrInvalidConfig, p.ThresholdDB)
	case !(p.TransitionDB > 0) || math.IsInf(float64(p.TransitionDB), 1): // rejects NaN, <= 0, +Inf
		return fmt.Errorf("%w: TransitionDB must be finite and > 0, got %g", ErrInvalidConfig, p.TransitionDB)
	case p.FreqSmoothBins < 0:
		return fmt.Errorf("%w: FreqSmoothBins must be >= 0, got %d", ErrInvalidConfig, p.FreqSmoothBins)
	case p.TimeSmoothFrames < 0:
		return fmt.Errorf("%w: TimeSmoothFrames must be >= 0, got %d", ErrInvalidConfig, p.TimeSmoothFrames)
	case !(p.FloorWindowSec > 0) || math.IsInf(float64(p.FloorWindowSec), 1): // rejects NaN, <= 0, +Inf
		return fmt.Errorf("%w: FloorWindowSec must be finite and > 0, got %g", ErrInvalidConfig, p.FloorWindowSec)
	}
	return nil
}

// strengthParams are the soft-gate method's tuned knob sets per Strength. The
// TimeSmoothFrames column is the full odd width 2L+1 (Light and Medium give L=1,
// Heavy L=2). MaxAttenuationDB follows the flagship's 6/12/20 so the shared
// synthetic quality bar (MaxAttenuationDB-4) carries over. These are reasoned
// starting values, to be tuned on the synthetic harness and then a real corpus.
var strengthParams = [...]Params{
	denoiser.Medium: {MaxAttenuationDB: 12, ThresholdDB: 5, TransitionDB: 6, FreqSmoothBins: 5, TimeSmoothFrames: 3, FloorWindowSec: 0.75},
	denoiser.Light:  {MaxAttenuationDB: 6, ThresholdDB: 3, TransitionDB: 6, FreqSmoothBins: 3, TimeSmoothFrames: 3, FloorWindowSec: 0.75},
	denoiser.Heavy:  {MaxAttenuationDB: 20, ThresholdDB: 8, TransitionDB: 5, FreqSmoothBins: 5, TimeSmoothFrames: 5, FloorWindowSec: 0.75},
}

// ParamsFor returns the soft-gate method's tuned Params for a Strength (Medium's
// for an unknown value). Start from it to override individual knobs through
// Config.Params.
func ParamsFor(s denoiser.Strength) Params {
	if !s.Valid() {
		return strengthParams[denoiser.Medium]
	}
	return strengthParams[s]
}
