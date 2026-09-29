package gate

import (
	"fmt"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
)

// Config configures a Gate. Only SampleRate is required.
type Config struct {
	// SampleRate in Hz; required.
	SampleRate int
	// FrameSize is the FFT length in samples, a power of two >= 64. 0 selects the
	// power of two nearest to 21.3 ms of audio: 1024 at 44.1 and 48 kHz, 512 at
	// 22.05-32 kHz, 256 at 16 kHz. Both methods share one auto rule, so they
	// frame a given rate identically.
	FrameSize int
	// HopSize is the frame advance in samples; it must be a divisor of FrameSize
	// smaller than FrameSize, giving at least 2x overlap. 0 selects FrameSize/4
	// (75% overlap).
	HopSize int
	// Strength selects the tuned knob set; the zero value is denoiser.Medium.
	Strength denoiser.Strength
	// Params, when non-nil, replaces the strength's knobs entirely.
	Params *Params
}

// resolve fills defaults and validates, returning the effective Config and the
// Params in force.
func (c Config) resolve() (Config, Params, error) {
	var err error
	c.FrameSize, c.HopSize, err = dspshared.ResolveFrame(c.SampleRate, c.FrameSize, c.HopSize)
	if err != nil {
		return c, Params{}, err
	}
	var p Params
	if c.Params != nil {
		p = *c.Params
	} else {
		if !c.Strength.Valid() {
			return c, Params{}, fmt.Errorf("%w: unknown Strength %d", ErrInvalidConfig, int(c.Strength))
		}
		p = ParamsFor(c.Strength)
	}
	if err := p.validate(); err != nil {
		return c, Params{}, err
	}
	return c, p, nil
}
