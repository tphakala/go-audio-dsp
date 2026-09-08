package denoiser

import "errors"

// Sentinel errors. Wrapped errors can be tested with errors.Is.
var (
	// ErrInvalidConfig reports a Config that New cannot honour; the wrapped
	// message names the offending field.
	ErrInvalidConfig = errors.New("denoiser: invalid config")
	// ErrBufferTooSmall reports an output slice shorter than the samples a
	// ProcessInto or FlushInto call would emit; nothing is consumed.
	ErrBufferTooSmall = errors.New("denoiser: output buffer too small")
	// ErrNoQuietRegion reports that EstimateNoiseProfile found no window
	// distinctly quieter than the rest of the clip.
	ErrNoQuietRegion = errors.New("denoiser: no distinct quiet region found")
	// ErrProfileTooShort reports a noise sample shorter than one frame.
	ErrProfileTooShort = errors.New("denoiser: noise sample shorter than one frame")
	// ErrProfileMismatch reports a NoiseProfile built for another FrameSize.
	ErrProfileMismatch = errors.New("denoiser: noise profile built for a different frame size")
)
