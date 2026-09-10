package gain

import (
	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/pcm"
)

// Sentinel errors. Wrapped errors can be tested with errors.Is.
var (
	// ErrInvalidConfig reports a gain New cannot honour (a non-finite dB). It is
	// the shared dsp.ErrInvalidConfig so a caller can test any block's
	// construction failure with a single errors.Is.
	ErrInvalidConfig = dsp.ErrInvalidConfig
	// ErrBufferTooSmall reports a ProcessInto output slice shorter than the
	// input; nothing is written. It is the shared dsp.ErrBufferTooSmall so a
	// caller chaining blocks can test every block's undersize error uniformly.
	ErrBufferTooSmall = dsp.ErrBufferTooSmall
	// ErrOddByteLength reports a byte slice whose length is not a multiple of
	// two, so it cannot hold whole little-endian int16 samples. It is the shared
	// pcm.ErrOddByteLength.
	ErrOddByteLength = pcm.ErrOddByteLength
)
