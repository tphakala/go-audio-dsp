package gate

import (
	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/denoiser"
)

// A Gate is a streaming dsp.Processor with a tail, so it also satisfies
// dsp.Flusher, and it can learn its noise floor from a noise-only excerpt, so it
// satisfies the parent denoiser.NoiseLearner capability. These assertions fail to
// compile if the contract drifts.
var (
	_ dsp.Processor         = (*Gate)(nil)
	_ dsp.Flusher           = (*Gate)(nil)
	_ denoiser.NoiseLearner = (*Gate)(nil)
)

// FrameSize is the resolved FFT length in samples (the Config.FrameSize in force,
// with the auto value substituted when Config.FrameSize was 0).
func (g *Gate) FrameSize() int { return g.n }

// HopSize is the resolved frame advance in samples (the Config.HopSize in force,
// with the FrameSize/4 default substituted when Config.HopSize was 0).
func (g *Gate) HopSize() int { return g.hop }

// MaxOutputLen returns an output length that always holds ProcessInto's output
// for an input of inputLen samples: inputLen + HopSize(). This is a safe upper
// bound, not the exact count (output is emitted in whole HopSize blocks, and the
// time-smoothing lookahead delays emission, so a small input can produce fewer
// samples, even zero). Sizing an output buffer with it guarantees ProcessInto
// never returns ErrBufferTooSmall.
func (g *Gate) MaxOutputLen(inputLen int) int { return inputLen + g.hop }
