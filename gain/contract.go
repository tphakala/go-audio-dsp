package gain

import dsp "github.com/tphakala/go-audio-dsp"

// A Gain is a memoryless streaming dsp.Processor: zero latency and no tail, so
// it does not implement dsp.Flusher. This assertion fails to compile if the
// contract drifts.
var _ dsp.Processor = (*Gain)(nil)
