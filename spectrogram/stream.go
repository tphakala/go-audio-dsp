package spectrogram

import "github.com/tphakala/go-audio-dsp/stft"

// ColumnSource is a streaming producer for a fixed Config: it feeds arbitrary
// chunks of a live stream through a resident stft.Analyzer and emits one
// finished spectrogram column per completed frame. It carries one frame of
// state, so memory is bounded regardless of stream length. It shares its scale
// stage with Spectrogram, so a streamed column equals the whole-clip column at
// the same frame offset bit for bit, for every scale. Not safe for concurrent
// use; build one per stream and reuse it with Reset.
type ColumnSource struct {
	an     *stft.Analyzer
	sc     scaler
	n, hop int
	col    []float32 // reusable output column handed to the callback, len sc.bins()
	frames int64     // frames emitted since construction or the last Reset
}

// NewColumnSource builds a ColumnSource for cfg. It validates cfg and returns an
// error wrapping ErrInvalidConfig for the first invalid field.
func NewColumnSource(cfg Config) (*ColumnSource, error) {
	a, sc, err := buildEngine(cfg)
	if err != nil {
		return nil, err
	}
	return &ColumnSource{
		an:  a,
		sc:  sc,
		n:   a.FrameSize(),
		hop: a.HopSize(),
		col: make([]float32, sc.bins()),
	}, nil
}

// FrameSize returns the transform size in samples.
func (s *ColumnSource) FrameSize() int { return s.n }

// HopSize returns the frame advance in samples.
func (s *ColumnSource) HopSize() int { return s.hop }

// Bins returns the number of frequency rows in each emitted column.
func (s *ColumnSource) Bins() int { return s.sc.bins() }

// BinHz returns the center frequency in Hz of row bin (0-based) in an emitted
// column, matching Spectrogram.BinHz for the same Config.
func (s *ColumnSource) BinHz(bin int) float64 { return s.sc.binHzOf(bin) }

// Feed pushes src through the analyzer and calls onColumn once per completed
// frame, in order, with that frame's scaled column and the center-sample index
// of the frame within the stream (frame f under NoPad spans [f*HopSize,
// f*HopSize+FrameSize), so its center is f*HopSize + FrameSize/2). col has length
// Bins() and is source-owned: it is valid only until onColumn returns and is
// overwritten on the next frame, so a caller that keeps a column copies it. A
// Feed that does not complete a frame calls onColumn zero times. onColumn must
// not be nil. Allocation-free in the steady state when onColumn does not allocate
// or let col escape.
func (s *ColumnSource) Feed(src []float32, onColumn func(col []float32, centerSample int64)) {
	s.an.Feed(src, func(_ []complex64, power []float32) {
		s.sc.apply(s.col, power)
		center := s.frames*int64(s.hop) + int64(s.n/2)
		onColumn(s.col, center)
		s.frames++
	})
}

// Reset clears the analyzer buffer and the frame counter so the next Feed starts
// a new stream at center-sample 0. The configuration is kept.
func (s *ColumnSource) Reset() {
	s.an.Reset()
	s.frames = 0
}
