package equalizer

import dsp "github.com/tphakala/go-audio-dsp"

// An Equalizer is a streaming dsp.Processor with zero latency and no tail (a
// biquad's output sample aligns with its input sample), so it does not implement
// dsp.Flusher. This assertion fails to compile if the contract drifts.
var _ dsp.Processor = (*Equalizer)(nil)
