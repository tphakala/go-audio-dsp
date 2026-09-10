package denoiser

import dsp "github.com/tphakala/go-audio-dsp"

// A Denoiser is a streaming dsp.Processor with a tail, so it also satisfies
// dsp.Flusher. These assertions fail to compile if the contract drifts.
var (
	_ dsp.Processor = (*Denoiser)(nil)
	_ dsp.Flusher   = (*Denoiser)(nil)
)

// FrameSize is the resolved FFT length in samples (the Config.FrameSize in
// force, with the auto value substituted when Config.FrameSize was 0).
func (d *Denoiser) FrameSize() int { return d.n }

// HopSize is the resolved frame advance in samples (the Config.HopSize in
// force, with the FrameSize/4 default substituted when Config.HopSize was 0).
func (d *Denoiser) HopSize() int { return d.hop }

// MaxOutputLen returns an output length that always holds ProcessInto's output
// for an input of inputLen samples: inputLen + HopSize(). This is a safe upper
// bound, not the exact count (output is emitted in whole HopSize blocks, so a
// small input can produce fewer samples, even zero). Sizing an output buffer
// with it guarantees ProcessInto never returns ErrBufferTooSmall.
func (d *Denoiser) MaxOutputLen(inputLen int) int { return inputLen + d.hop }
