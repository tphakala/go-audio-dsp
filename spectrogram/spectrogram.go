package spectrogram

import (
	"math"

	"github.com/tphakala/go-audio-dsp/stft"
)

// Spectrogram is a whole-clip producer for a fixed Config: it runs a resident
// stft.Analyzer over a signal and writes one scaled column per completed frame
// into a Matrix. Reuse one across many clips to stay allocation-free; each
// ComputeInto resets the analyzer so clips are independent. Not safe for
// concurrent use.
type Spectrogram struct {
	an  *stft.Analyzer
	sc  scaler
	n   int // FrameSize
	hop int // HopSize
}

// New builds a Spectrogram for cfg. It validates cfg and returns an error
// wrapping ErrInvalidConfig for the first invalid field.
func New(cfg Config) (*Spectrogram, error) {
	a, sc, err := buildEngine(cfg)
	if err != nil {
		return nil, err
	}
	return &Spectrogram{an: a, sc: sc, n: a.FrameSize(), hop: a.HopSize()}, nil
}

// FrameSize returns the transform size in samples.
func (s *Spectrogram) FrameSize() int { return s.n }

// HopSize returns the frame advance in samples.
func (s *Spectrogram) HopSize() int { return s.hop }

// Bins returns the number of output frequency rows per column: the count of
// FFT bins whose center frequency lies in the configured [MinHz, MaxHz] range.
func (s *Spectrogram) Bins() int { return s.sc.bins() }

// BinHz returns the center frequency in Hz of output row bin (0-based within the
// selected sub-range), so BinHz(0) is the lowest emitted frequency. A renderer
// uses it to place a linear or log frequency axis.
func (s *Spectrogram) BinHz(bin int) float64 { return s.sc.binHzOf(bin) }

// NumFrames reports how many columns ComputeInto writes for a signal of
// signalLen samples under the NoPad framing: 1 + (signalLen-FrameSize)/HopSize,
// or 0 when the signal is shorter than one frame. A caller uses it to size a
// Matrix.
func (s *Spectrogram) NumFrames(signalLen int) int {
	if signalLen < s.n {
		return 0
	}
	return 1 + (signalLen-s.n)/s.hop
}

// Matrix holds a spectrogram as frame-contiguous float32 values: column
// (frame) f occupies Data[f*Bins : (f+1)*Bins], each entry a frequency row in
// the configured scale. This is the layout the streaming source emits one
// column at a time, so a whole-clip Matrix and a stream agree column for column.
type Matrix struct {
	Bins   int
	Frames int
	Data   []float32
}

// Column returns column frame's Bins values (a view into Data, not a copy). A
// renderer drawing time on the x axis iterates frames and reads each column.
func (m *Matrix) Column(frame int) []float32 {
	return m.Data[frame*m.Bins : (frame+1)*m.Bins]
}

// At returns the value at frequency row bin of column frame.
func (m *Matrix) At(bin, frame int) float32 {
	return m.Data[frame*m.Bins+bin]
}

// Compute returns a freshly allocated Matrix for signal. It is the allocating
// convenience over ComputeInto; reuse ComputeInto with a retained Matrix on a
// hot path.
func (s *Spectrogram) Compute(signal []float32) Matrix {
	m := Matrix{Data: make([]float32, s.sc.bins()*s.NumFrames(len(signal)))}
	// ComputeInto cannot fail here: Data is sized exactly to the requirement.
	_, _ = s.ComputeInto(&m, signal)
	return m
}

// ComputeInto writes signal's spectrogram into dst and returns the number of
// columns written. dst.Data must have capacity for Bins()*NumFrames(len(signal))
// values; a smaller capacity returns ErrBufferTooSmall and writes nothing.
// dst.Data is resliced to the exact length and dst.Bins and dst.Frames are set.
// Allocation-free when dst.Data already has the capacity. The analyzer is reset
// first, so repeated calls are independent.
func (s *Spectrogram) ComputeInto(dst *Matrix, signal []float32) (int, error) {
	bins := s.sc.bins()
	frames := s.NumFrames(len(signal))
	need := bins * frames
	if cap(dst.Data) < need {
		return 0, ErrBufferTooSmall
	}
	dst.Data = dst.Data[:need]
	dst.Bins = bins
	dst.Frames = frames

	s.an.Reset()
	frame := 0
	s.an.Feed(signal, func(_ []complex64, power []float32) {
		s.sc.apply(dst.Data[frame*bins:(frame+1)*bins], power)
		frame++
	})
	return frame, nil
}

// HopForWidth returns a hop size that makes a signal of signalLen samples span
// close to columns frames under NoPad framing, so a caller can fit a clip to a
// target image width. It rounds the ideal hop (signalLen-frameSize)/(columns-1)
// to the nearest sample and clamps it to [1, frameSize]. Because the hop is an
// integer, the realized column count (see NumFrames) only approximates columns:
// the approximation is close when the ideal hop is comfortably above 1, and
// degrades as it rounds toward 1, since a hop of 1 is the minimum and a clip
// with fewer samples than the requested columns then yields more columns than
// asked. For columns <= 1 or a signal no longer than one frame it returns
// frameSize (a single column).
func HopForWidth(signalLen, frameSize, columns int) int {
	if columns <= 1 || signalLen <= frameSize {
		return frameSize
	}
	hop := int(math.Round(float64(signalLen-frameSize) / float64(columns-1)))
	if hop < 1 {
		return 1
	}
	if hop > frameSize {
		return frameSize
	}
	return hop
}
