package gate

import (
	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/denoiser"
)

// Sentinel errors. Wrapped errors can be tested with errors.Is. The gate reuses
// the shared dsp sentinels and the parent denoiser sentinels so a consumer
// chaining or switching denoise methods tests one error value per condition,
// regardless of which method produced it.
var (
	// ErrInvalidConfig reports a Config that New cannot honour; the wrapped
	// message names the offending field. It is the shared dsp.ErrInvalidConfig.
	ErrInvalidConfig = dsp.ErrInvalidConfig
	// ErrBufferTooSmall reports an output slice shorter than the samples a
	// ProcessInto or FlushInto call would emit; nothing is consumed. It is the
	// shared dsp.ErrBufferTooSmall.
	ErrBufferTooSmall = dsp.ErrBufferTooSmall
	// ErrNoiseTooShort reports a LearnNoise excerpt shorter than one frame. It is
	// the parent denoiser.ErrProfileTooShort, so one errors.Is covers a
	// too-short noise excerpt for either denoise method.
	ErrNoiseTooShort = denoiser.ErrProfileTooShort
	// ErrFloorMismatch reports a SetNoiseFloor spectrum whose length is not
	// FrameSize/2+1. It is the parent denoiser.ErrProfileMismatch.
	ErrFloorMismatch = denoiser.ErrProfileMismatch
)
