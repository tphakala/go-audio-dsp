package mel

import dsp "github.com/tphakala/go-audio-dsp"

// Sentinel errors. Wrapped errors can be tested with errors.Is.
var (
	// ErrInvalidConfig reports a Config, FilterbankConfig, or FilterbankFromRows
	// input that a constructor cannot honour; the wrapped message names the
	// offending field. It is the shared dsp.ErrInvalidConfig so a caller can test
	// any block's construction failure with a single errors.Is.
	ErrInvalidConfig = dsp.ErrInvalidConfig
	// ErrBufferTooSmall reports a ComputeInto destination whose Data slice has too
	// little capacity for the output matrix; nothing is written. It is the shared
	// dsp.ErrBufferTooSmall so a caller can test every block's undersize error
	// uniformly.
	ErrBufferTooSmall = dsp.ErrBufferTooSmall
)
