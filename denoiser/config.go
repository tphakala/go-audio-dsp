package denoiser

import (
	"fmt"
	"math"
)

// Config configures a Denoiser. Only SampleRate is required.
type Config struct {
	// SampleRate in Hz; required.
	SampleRate int
	// FrameSize is the FFT length in samples, a power of two >= 64. 0 selects
	// the power of two nearest to 21.3 ms of audio: 1024 at 44.1 and 48 kHz,
	// 512 at 22.05-32 kHz, 256 at 16 kHz.
	FrameSize int
	// HopSize is the frame advance in samples; it must be a divisor of FrameSize
	// smaller than FrameSize, giving at least 2x overlap (a periodic Hann window
	// needs overlap to reconstruct without gaps). 0 selects FrameSize/4 (75%
	// overlap).
	HopSize int
	// Preset selects the tuned knob set; the zero value is Medium.
	Preset Preset
	// Params, when non-nil, replaces the preset's knobs entirely.
	Params *Params
}

const (
	minFrameSize     = 64
	autoFrameSeconds = 0.0213
	defaultOverlap   = 4 // hop = frame/4
)

// autoFrameSize returns the power of two nearest to autoFrameSeconds of audio,
// never below minFrameSize.
func autoFrameSize(sampleRate int) int {
	e := int(math.Round(math.Log2(float64(sampleRate) * autoFrameSeconds)))
	return max(minFrameSize, 1<<max(e, 0))
}

// resolve fills defaults and validates, returning the effective Config and the
// Params in force.
func (c Config) resolve() (Config, Params, error) {
	if c.SampleRate <= 0 {
		return c, Params{}, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, c.SampleRate)
	}
	if c.FrameSize == 0 {
		c.FrameSize = autoFrameSize(c.SampleRate)
	}
	if c.FrameSize < minFrameSize || c.FrameSize&(c.FrameSize-1) != 0 {
		return c, Params{}, fmt.Errorf("%w: FrameSize must be a power of two >= %d, got %d", ErrInvalidConfig, minFrameSize, c.FrameSize)
	}
	if c.HopSize == 0 {
		c.HopSize = c.FrameSize / defaultOverlap
	}
	if c.HopSize < 1 || c.HopSize >= c.FrameSize || c.FrameSize%c.HopSize != 0 {
		return c, Params{}, fmt.Errorf("%w: HopSize must be a divisor of FrameSize (%d) smaller than it, got %d", ErrInvalidConfig, c.FrameSize, c.HopSize)
	}
	var p Params
	if c.Params != nil {
		p = *c.Params
	} else {
		if !c.Preset.valid() {
			return c, Params{}, fmt.Errorf("%w: unknown Preset %d", ErrInvalidConfig, int(c.Preset))
		}
		p = c.Preset.Params()
	}
	if err := p.validate(); err != nil {
		return c, Params{}, err
	}
	return c, p, nil
}
