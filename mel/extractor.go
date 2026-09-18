package mel

import "github.com/tphakala/go-audio-dsp/stft"

// Matrix holds frame-contiguous mel columns: frame f occupies
// Data[f*Mels : (f+1)*Mels]. Same layout as spectrogram.Matrix with Mels in place
// of Bins, so a stream and a whole-clip matrix agree column for column.
type Matrix struct {
	Mels   int
	Frames int
	Data   []float32
}

// Column returns column frame's Mels values (a view into Data, not a copy).
func (m *Matrix) Column(frame int) []float32 {
	return m.Data[frame*m.Mels : (frame+1)*m.Mels]
}

// At returns the value at mel row mel of column frame.
func (m *Matrix) At(mel, frame int) float32 {
	return m.Data[frame*m.Mels+mel]
}

// Extractor is a whole-clip producer for a fixed Config: it runs a resident
// stft.Analyzer over a signal and writes one mel column per completed frame into
// a Matrix. Reuse one across many clips to stay allocation-free; each ComputeInto
// resets the analyzer so clips are independent. Not safe for concurrent use.
type Extractor struct {
	an       *stft.Analyzer
	pr       projector
	n, hop   int
	pre, suf []float32 // pad scratch, each len FrameSize/2, for centered modes
}

// New builds an Extractor for cfg, returning an error wrapping ErrInvalidConfig
// for the first invalid field.
func New(cfg Config) (*Extractor, error) {
	a, pr, err := buildEngine(cfg)
	if err != nil {
		return nil, err
	}
	n := a.FrameSize()
	return &Extractor{
		an:  a,
		pr:  pr,
		n:   n,
		hop: a.HopSize(),
		pre: make([]float32, n/2),
		suf: make([]float32, n/2),
	}, nil
}

// FrameSize returns the transform size in samples.
func (e *Extractor) FrameSize() int { return e.n }

// HopSize returns the frame advance in samples.
func (e *Extractor) HopSize() int { return e.hop }

// NumMels returns the number of mel rows per column.
func (e *Extractor) NumMels() int { return e.pr.fb.numMels }

// NumBins returns the transform's Hermitian half-spectrum length, FrameSize/2 + 1.
func (e *Extractor) NumBins() int { return e.an.NumBins() }

// Filterbank returns the immutable mel filterbank in use. Do not mutate it.
func (e *Extractor) Filterbank() *Filterbank { return e.pr.fb }

// NumFrames reports how many columns ComputeInto writes for a signal of signalLen
// samples under pad, so a caller can size a Matrix. It matches
// stft.Plan.NumFrames: NoPad is 1 + (signalLen-FrameSize)/HopSize (0 when shorter
// than a frame); the centered modes are 1 + signalLen/HopSize (0 when signalLen
// is 0).
func (e *Extractor) NumFrames(signalLen int, pad stft.PadMode) int {
	return e.an.NumFrames(signalLen, pad)
}

// Compute returns a freshly allocated Matrix for signal under pad. It is the
// allocating convenience over ComputeInto; reuse ComputeInto with a retained
// Matrix on a hot path.
func (e *Extractor) Compute(signal []float32, pad stft.PadMode) Matrix {
	m := Matrix{Data: make([]float32, e.pr.fb.numMels*e.NumFrames(len(signal), pad))}
	// ComputeInto cannot fail here: Data is sized exactly to the requirement.
	_, _ = e.ComputeInto(&m, signal, pad)
	return m
}

// ComputeInto writes signal's mel matrix into dst under pad and returns the number
// of columns written. dst.Data must have capacity for NumMels()*NumFrames(...); a
// smaller capacity returns ErrBufferTooSmall and writes nothing. dst.Data is
// resliced to the exact length and dst.Mels and dst.Frames are set. Allocation-free
// when dst.Data already has the capacity. The analyzer is reset first, so repeated
// calls are independent.
//
// Framing follows pad: NoPad frames signal directly (frame f spans [f*hop,
// f*hop+FrameSize)); PadZero and PadReflect center each frame on f*hop by feeding
// FrameSize/2 of padding (zeros, or a numpy-style reflection) before and after the
// signal through the NoPad analyzer, so the result matches stft.Plan under the same
// pad without an allocation.
func (e *Extractor) ComputeInto(dst *Matrix, signal []float32, pad stft.PadMode) (int, error) {
	mels := e.pr.fb.numMels
	frames := e.NumFrames(len(signal), pad)
	need := mels * frames
	if cap(dst.Data) < need {
		return 0, ErrBufferTooSmall
	}
	dst.Data = dst.Data[:need]
	dst.Mels = mels
	dst.Frames = frames
	// Reset unconditionally (even for a 0-frame signal) so the analyzer buffer is
	// always clean afterwards and repeated calls stay independent.
	e.an.Reset()
	if frames == 0 {
		return 0, nil
	}

	frame := 0
	onFrame := func(_ []complex64, power []float32) {
		e.pr.apply(dst.Data[frame*mels:(frame+1)*mels], power)
		frame++
	}
	if centered(pad) {
		half := e.n / 2
		e.fillPad(signal, pad == stft.PadReflect)
		e.an.Feed(e.pre[:half], onFrame)
		e.an.Feed(signal, onFrame)
		e.an.Feed(e.suf[:half], onFrame)
	} else {
		e.an.Feed(signal, onFrame)
	}
	return frame, nil
}

// fillPad fills the FrameSize/2 prefix and suffix scratch for a centered frame.
// PadZero clears them to zeros; PadReflect (reflect == true) fills them with the
// numpy-style reflection of signal, matching simd's centered framing so the padded
// NoPad analyzer reproduces stft.Plan under PadReflect. The signal is never empty
// here (frames == 0 returns earlier), so reflectIndex is well defined.
func (e *Extractor) fillPad(signal []float32, reflect bool) {
	half := e.n / 2
	if !reflect {
		clear(e.pre[:half])
		clear(e.suf[:half])
		return
	}
	n := len(signal)
	for k := range half {
		// Prefix sample k maps to original index k-half (before the signal start).
		e.pre[k] = signal[reflectIndex(k-half, n)]
	}
	for j := range half {
		// Suffix sample j maps to original index n+j (past the signal end).
		e.suf[j] = signal[reflectIndex(n+j, n)]
	}
}

// centered reports whether pad adds FrameSize/2 padding per side (PadZero or
// PadReflect). Any other value, including NoPad, frames without padding, matching
// stft.PadMode's fallback.
func centered(pad stft.PadMode) bool {
	return pad == stft.PadZero || pad == stft.PadReflect
}

// reflectIndex maps an out-of-range index into [0, n) by numpy "reflect" folding
// (period 2*(n-1), edge sample not repeated), the same fold simd applies for
// PadReflect. A ported copy so mel does not depend on unexported simd internals.
func reflectIndex(idx, n int) int {
	if n == 1 {
		return 0
	}
	period := (n - 1) << 1
	m := idx % period
	if m < 0 {
		m += period
	}
	if m < n {
		return m
	}
	return period - m
}
