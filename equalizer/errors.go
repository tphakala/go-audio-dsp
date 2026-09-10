package equalizer

import dsp "github.com/tphakala/go-audio-dsp"

// Sentinel errors. Wrapped errors can be tested with errors.Is.
var (
	// ErrInvalidConfig reports a Config that New cannot honour; the wrapped
	// message names the offending band and field. It is the shared
	// dsp.ErrInvalidConfig so a caller can test any block's construction failure
	// with a single errors.Is.
	ErrInvalidConfig = dsp.ErrInvalidConfig
	// ErrBufferTooSmall reports a ProcessInto output slice shorter than the
	// input; nothing is consumed. It is the shared dsp.ErrBufferTooSmall so a
	// caller chaining blocks can test every block's undersize error uniformly.
	ErrBufferTooSmall = dsp.ErrBufferTooSmall
)
