package dsp

import "errors"

// Shared sentinel errors. A block reports these conditions by returning or
// wrapping one of these values, so a consumer that chains blocks can test for
// them uniformly with errors.Is instead of matching a per-package error.
var (
	// ErrBufferTooSmall reports that a ProcessInto or FlushInto output slice was
	// shorter than the samples the call would emit. The call consumes nothing
	// and leaves the block unchanged, so the caller can retry with a larger
	// buffer.
	ErrBufferTooSmall = errors.New("dsp: output buffer too small")
	// ErrInvalidConfig reports a configuration a block's constructor cannot
	// honour. A block wraps it with the offending field, so errors.Is(err,
	// ErrInvalidConfig) holds for any block's construction failure.
	ErrInvalidConfig = errors.New("dsp: invalid config")
)
