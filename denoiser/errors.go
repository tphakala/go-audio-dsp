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
)
