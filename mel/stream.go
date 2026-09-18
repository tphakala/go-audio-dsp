package mel

import "github.com/tphakala/go-audio-dsp/stft"

// ColumnSource is a streaming producer for a fixed Config: it feeds arbitrary
// chunks of a live stream through a resident stft.Analyzer and emits one finished
// mel column per completed frame. It carries one frame of state, so memory is
// bounded regardless of stream length. It shares its projection stage with
// Extractor, so a streamed column equals the whole-clip column (NoPad) at the same
// frame offset bit for bit. Not safe for concurrent use; build one per stream and
// reuse it with Reset.
type ColumnSource struct {
	an     *stft.Analyzer
	pr     projector
	n, hop int
	col    []float32 // reusable output column handed to the callback, len NumMels
	frames int64     // frames emitted since construction or the last Reset
}

// NewColumnSource builds a ColumnSource for cfg, returning an error wrapping
// ErrInvalidConfig for the first invalid field.
func NewColumnSource(cfg Config) (*ColumnSource, error) {
	a, pr, err := buildEngine(cfg)
	if err != nil {
		return nil, err
	}
	return &ColumnSource{
		an:  a,
		pr:  pr,
		n:   a.FrameSize(),
		hop: a.HopSize(),
		col: make([]float32, pr.fb.numMels),
	}, nil
}

// FrameSize returns the transform size in samples.
func (s *ColumnSource) FrameSize() int { return s.n }

// HopSize returns the frame advance in samples.
func (s *ColumnSource) HopSize() int { return s.hop }

// NumMels returns the number of mel rows in each emitted column.
func (s *ColumnSource) NumMels() int { return s.pr.fb.numMels }

// NumBins returns the transform's Hermitian half-spectrum length, FrameSize/2 + 1.
func (s *ColumnSource) NumBins() int { return s.an.NumBins() }

// Filterbank returns the immutable mel filterbank in use. Do not mutate it.
func (s *ColumnSource) Filterbank() *Filterbank { return s.pr.fb }

// Feed pushes src through the analyzer and calls onColumn once per completed
// frame, in order, with that frame's mel column and the center-sample index of the
// frame within the stream (frame f under NoPad spans [f*HopSize, f*HopSize+FrameSize),
// so its center is f*HopSize + FrameSize/2). col has length NumMels() and is
// source-owned: it is valid only until onColumn returns and is overwritten on the
// next frame, so a caller that keeps a column copies it. A Feed that does not
// complete a frame calls onColumn zero times. onColumn must not be nil.
// Allocation-free in the steady state when onColumn does not allocate or let col
// escape.
func (s *ColumnSource) Feed(src []float32, onColumn func(col []float32, centerSample int64)) {
	s.an.Feed(src, func(_ []complex64, power []float32) {
		s.pr.apply(s.col, power)
		center := s.frames*int64(s.hop) + int64(s.n/2)
		onColumn(s.col, center)
		s.frames++
	})
}

// Reset clears the analyzer buffer and the frame counter so the next Feed starts a
// new stream at center-sample 0. The configuration is kept.
func (s *ColumnSource) Reset() {
	s.an.Reset()
	s.frames = 0
}
