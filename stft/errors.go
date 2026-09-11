package stft

import dsp "github.com/tphakala/go-audio-dsp"

// Sentinel errors. Wrapped errors can be tested with errors.Is.
var (
	// ErrInvalidConfig reports a Config that New or NewAnalyzer cannot honour; the
	// wrapped message names the offending field. It is the shared
	// dsp.ErrInvalidConfig so a caller can test any block's construction failure
	// with a single errors.Is.
	ErrInvalidConfig = dsp.ErrInvalidConfig
)
