package dspshared

import (
	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/stft"
)

// FrameFunc is a denoise method's per-frame step. f is the frame index in the
// stream (0 for the first frame after the leading-zero preroll); spec and power
// are the Analyzer-owned half-spectrum and |X|^2, valid only during the call
// (spec may be modified in place); flushing is true while FlushInto drains the
// tail, when adaptive estimates freeze. It returns the gained spectrum to
// overlap-add for the output frame f-delay, or nil while fewer than delay
// frames have completed (the method is still priming).
type FrameFunc func(f int64, spec []complex64, power []float32, flushing bool) []complex64

// Stream drives a method's FrameFunc over the stft.Analyzer and
// stft.Synthesizer: the leading-zero preroll, the frame and sample counters,
// the per-frame overlap-add and the Process, Flush and Reset arithmetic shared
// by the denoiser and the gate.
//
// The stream is modelled as the padded sequence p = (n-hop zeros) ++ input ++
// (flush zeros). Frame f is p[f*hop : f*hop+n]. Input sample t is p[t+n-hop],
// so output is aligned with input. The output for frame f-delay is produced
// when frame f completes; the first warm = n/hop-1+delay blocks are discarded
// (leading zeros, plus the method's priming delay), and the steady-state lag is
// (n-hop)+delay*hop samples. Framing, windowing, the RFFT and |X|^2 live in the
// stft.Analyzer and the overlap-add in the stft.Synthesizer.
//
// Initialize with Init before use. A Stream must not be copied after Init: its
// hooks are bound to the owning block.
type Stream struct {
	an     *stft.Analyzer
	syn    *stft.Synthesizer
	n, hop int
	delay  int   // frames the method holds a frame back (0 flagship, L gate)
	warm   int64 // n/hop-1+delay: leading frames whose output block is discarded

	prezeros []float32 // len n-hop: the leading-zero preroll fed on Reset
	zeros    []float32 // len hop, fed by FlushInto

	frame      FrameFunc
	resetState func() // resets the method's own stream state; called by Reset

	frames            int64 // frames processed in this stream
	totalIn, totalOut int64
}

// Init builds the synthesizer over an and stores the method hooks. delay is the
// number of frames the method holds a frame back (0 emits each frame as it
// completes). frame is called once per completed frame; resetState is called
// by Reset after the driver's own state is cleared. Init does not call Reset:
// the caller finishes allocating whatever resetState touches, then calls Reset.
// It returns the synthesizer's error, which wraps dsp.ErrInvalidConfig.
func (s *Stream) Init(an *stft.Analyzer, delay int, frame FrameFunc, resetState func()) error {
	syn, err := stft.NewSynthesizer(an)
	if err != nil {
		return err
	}
	s.an = an
	s.syn = syn
	s.n, s.hop = an.FrameSize(), an.HopSize()
	s.delay = delay
	s.warm = int64(s.n/s.hop - 1 + delay)
	s.prezeros = make([]float32, s.n-s.hop)
	s.zeros = make([]float32, s.hop)
	s.frame = frame
	s.resetState = resetState
	return nil
}

// Reset clears the analysis buffer, the overlap-add accumulator and the
// counters, re-primes the analyzer with the leading zeros, then runs the
// method's reset hook.
func (s *Stream) Reset() {
	// The stream is modelled as n-hop leading zeros ++ input. Prime the analyzer
	// with that preroll so the first output sample aligns with the first input
	// sample; n-hop < n, so no frame completes and noEmit is never called.
	s.an.Reset()
	s.an.Feed(s.prezeros, noEmit)
	s.syn.Reset()
	s.frames = 0
	s.totalIn, s.totalOut = 0, 0
	s.resetState()
}

// Latency returns how many samples the output lags the input in steady state:
// (n-hop)+delay*hop. Output is emitted in whole hop blocks, so between block
// boundaries the lag is up to hop-1 samples more.
func (s *Stream) Latency() int { return (s.n - s.hop) + s.delay*s.hop }

// MaxOutputLen returns an output length that always holds ProcessInto's output
// for an input of inputLen samples: inputLen + hop.
func (s *Stream) MaxOutputLen(inputLen int) int { return inputLen + s.hop }

// pending reports how many output samples feeding extra more input samples
// will emit, given the current stream state.
func (s *Stream) pending(extra int) int {
	return pendingOutput(s.an.InFill(), extra, s.n, s.hop, s.frames, s.warm)
}

// feed pushes in through the analyzer and, for each frame that completes, runs
// the method's frame hook and overlap-adds the spectrum it returns, writing the
// finalized output blocks into out, clipped to len(out) (ProcessInto sizes out
// exactly; FlushInto clips the final block). Returns samples written.
func (s *Stream) feed(in, out []float32, flushing bool) int {
	written := 0
	s.an.Feed(in, func(spec []complex64, power []float32) {
		f := s.frames
		if row := s.frame(f, spec, power, flushing); row != nil {
			s.syn.Add(row)
			if f >= s.warm {
				written += s.syn.Finish(out[written:])
			} else {
				s.syn.Discard()
			}
		}
		s.frames++
	})
	return written
}

// Process feeds the next chunk of the stream and returns the output samples
// that became final, in a freshly allocated slice (possibly empty).
func (s *Stream) Process(in []float32) ([]float32, error) {
	out := make([]float32, s.pending(len(in)))
	n, err := s.ProcessInto(in, out)
	return out[:n], err
}

// ProcessInto is Process without allocation. The count is known before any work
// is done; if len(out) is smaller, dsp.ErrBufferTooSmall is returned and
// nothing is consumed.
func (s *Stream) ProcessInto(in, out []float32) (int, error) {
	need := s.pending(len(in))
	if len(out) < need {
		return 0, dsp.ErrBufferTooSmall
	}
	n := s.feed(in, out[:need], false)
	s.totalIn += int64(len(in))
	s.totalOut += int64(n)
	return n, nil
}

// Flush ends the stream: it completes the frames overlapping the last input
// samples, returns the remaining output so the total output length equals the
// total input length, and resets the stream.
func (s *Stream) Flush() ([]float32, error) {
	out := make([]float32, int(s.totalIn-s.totalOut))
	n, err := s.FlushInto(out)
	return out[:n], err
}

// FlushInto is Flush without allocation. It returns dsp.ErrBufferTooSmall (and
// does nothing) if len(out) is smaller than the remaining output;
// len(out) >= Latency()+hop always suffices.
func (s *Stream) FlushInto(out []float32) (int, error) {
	need := int(s.totalIn - s.totalOut)
	if len(out) < need {
		return 0, dsp.ErrBufferTooSmall
	}
	written := 0
	for written < need {
		written += s.feed(s.zeros, out[written:need], true)
	}
	s.Reset()
	return written, nil
}

// ProcessAll runs x through the stream and flushes into one exact-length
// result.
func (s *Stream) ProcessAll(x []float32) ([]float32, error) {
	out := make([]float32, len(x))
	n, err := s.ProcessInto(x, out)
	if err != nil {
		return nil, err
	}
	m, err := s.FlushInto(out[n:])
	if err != nil {
		return nil, err
	}
	return out[:n+m], nil
}
