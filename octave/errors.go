package octave

import dsp "github.com/tphakala/go-audio-dsp"

// ErrInvalidConfig reports a Config that New cannot honour; the wrapped message
// names the offending field. It is the shared dsp.ErrInvalidConfig so a caller
// can test any block's construction failure with a single errors.Is.
var ErrInvalidConfig = dsp.ErrInvalidConfig
