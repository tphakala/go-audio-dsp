// Package dspshared holds the small helpers the flagship denoiser and the
// denoiser/gate method both need: the noise-power floor guard, the analysis
// configuration, frame-size and geometry defaults, the tracker window
// conversion, and the streaming emission arithmetic. They live here so a fix
// lands once; each caller keeps its own config type.
package dspshared

import (
	"fmt"
	"math"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/stft"
)

// EpsPower floors every noise power estimate so SNR ratios stay finite. Inputs
// are normalized float32 audio, so real bin powers sit far above it.
const EpsPower = 1e-12

const (
	// MinFrameSize is the smallest supported FFT length.
	MinFrameSize     = 64
	autoFrameSeconds = 0.0213 // audio duration the auto frame size targets
	defaultOverlap   = 4      // frame/hop ratio used when HopSize is 0
)

// CopyFloor copies src into dst, replacing each NaN, infinite, or value at or
// below EpsPower (and exactly math.MaxFloat32) with EpsPower, so the floor is always finite and strictly positive
// and a finite power divided by it stays finite. dst may alias src.
func CopyFloor(dst, src []float32) {
	for k, v := range src {
		if !(v > EpsPower) || !(v < math.MaxFloat32) { // NaN, <= floor, or +Inf
			v = EpsPower
		}
		dst[k] = v
	}
}

// TrackWindowFrames converts a tracker window from seconds to frames, never
// below one.
func TrackWindowFrames(sec float32, sampleRate, hop int) int {
	return max(1, int(math.Round(float64(sec)*float64(sampleRate)/float64(hop))))
}

// AutoFrameSize returns the power of two nearest to autoFrameSeconds of audio,
// never below MinFrameSize: 1024 at 44.1 and 48 kHz, 512 at 22.05-32 kHz, 256 at
// 16 kHz.
func AutoFrameSize(sampleRate int) int {
	e := int(math.Round(math.Log2(float64(sampleRate) * autoFrameSeconds)))
	return max(MinFrameSize, 1<<max(e, 0))
}

// ResolveFrame validates sampleRate and fills the frame and hop defaults (0
// selects AutoFrameSize and frame/defaultOverlap). It returns the effective
// frame and hop, or an error wrapping dsp.ErrInvalidConfig (the sentinel both
// methods alias) that names the first invalid field.
func ResolveFrame(sampleRate, frameSize, hopSize int) (frame, hop int, err error) {
	if sampleRate <= 0 {
		return 0, 0, fmt.Errorf("%w: SampleRate must be > 0, got %d", dsp.ErrInvalidConfig, sampleRate)
	}
	if frameSize == 0 {
		frameSize = AutoFrameSize(sampleRate)
	}
	if frameSize < MinFrameSize || frameSize&(frameSize-1) != 0 {
		return 0, 0, fmt.Errorf("%w: FrameSize must be a power of two >= %d, got %d", dsp.ErrInvalidConfig, MinFrameSize, frameSize)
	}
	if hopSize == 0 {
		hopSize = frameSize / defaultOverlap
	}
	if hopSize < 1 || hopSize >= frameSize || frameSize%hopSize != 0 {
		return 0, 0, fmt.Errorf("%w: HopSize must be a divisor of FrameSize (%d) smaller than it, got %d", dsp.ErrInvalidConfig, frameSize, hopSize)
	}
	return frameSize, hopSize, nil
}

// STFTConfig is the analysis configuration both methods use: a periodic Hann
// window at the given frame and hop. The shared window keeps a noise floor or
// profile measured by one method in the same |RFFT(hann*x)|^2 scale as the other's
// stream power.
func STFTConfig(frame, hop int) stft.Config {
	return stft.Config{FrameSize: frame, HopSize: hop, Window: stft.Hann}
}

// NoEmit is the no-op frame callback used to feed an analyzer's leading-zero
// preroll, during which no frame completes.
func NoEmit(spec []complex64, power []float32) {}

// PendingOutput reports how many output samples feeding extra more input
// samples will emit. inFill is the analyzer's buffered sample count, frames the
// frames completed so far, and warm the number of leading frames whose output
// block is discarded. Emission count after F frames is max(0, F-warm).
func PendingOutput(inFill, extra, n, hop int, frames, warm int64) int {
	total := inFill + extra
	if total < n {
		return 0
	}
	completing := int64((total-n)/hop) + 1 // frames that will complete
	cnt := frames + completing - max(frames, warm)
	if cnt <= 0 {
		return 0
	}
	return int(cnt) * hop
}
