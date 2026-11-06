package stft

import (
	"fmt"

	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// Analyzer is a streaming short-time analyzer for a fixed Config. It owns the
// analysis buffer and a resident simd plan, buffers fed samples into overlapping
// frames of FrameSize advancing by HopSize, and transforms each frame that
// completes. Framing is NoPad: frame f is the (n-hop)-overlapped window ending at
// the newest hop of input, with no implicit padding; a consumer that wants a
// centered or delayed start feeds its own leading zeros. Not safe for concurrent
// use; build one per stream and reuse it with Reset.
type Analyzer struct {
	p      *f32.STFTPlan
	window []float32
	n, hop int

	buf   []float32   // analysis buffer, len n; the next frame once fill == n
	fill  int         // samples currently in buf
	spec  []complex64 // per-frame Hermitian half-spectrum, reused
	power []float32   // per-frame magnitude-squared, reused
}

// NewAnalyzer builds a streaming Analyzer for cfg. It validates cfg and returns
// an error wrapping ErrInvalidConfig for the first invalid field. The buffer
// starts empty; the first frame completes once FrameSize samples have been fed.
func NewAnalyzer(cfg Config) (*Analyzer, error) {
	rc, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	p, err := f32.NewSTFTPlan(rc.FrameSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	bins := p.NumBins()
	return &Analyzer{
		p:      p,
		window: rc.buildWindow(),
		n:      rc.FrameSize,
		hop:    rc.HopSize,
		buf:    make([]float32, rc.FrameSize),
		spec:   make([]complex64, bins),
		power:  make([]float32, bins),
	}, nil
}

// FrameSize returns the transform size in samples.
func (a *Analyzer) FrameSize() int { return a.n }

// HopSize returns the frame advance in samples.
func (a *Analyzer) HopSize() int { return a.hop }

// NumBins returns FrameSize/2 + 1, the Hermitian half-spectrum length.
func (a *Analyzer) NumBins() int { return a.p.NumBins() }

// NumFrames reports how many frames a whole-clip pass over a signal of signalLen
// samples completes under pad, so a streaming caller can size a matrix the same
// way a Plan caller does. For NoPad (the streaming framing) it is
// 1 + (signalLen-FrameSize)/HopSize (0 when shorter than a frame); for the
// centered modes it is 1 + signalLen/HopSize (0 when signalLen <= 0). It matches
// Plan.NumFrames, so streaming and whole-clip callers agree bin for bin, and it
// is the single frame-count formula shared with the mel and spectrogram producers.
func (a *Analyzer) NumFrames(signalLen int, pad PadMode) int {
	return a.p.NumFrames(signalLen, a.hop, pad.simd())
}

// Window returns the resolved analysis window. Do not mutate it.
func (a *Analyzer) Window() []float32 { return a.window }

// InFill returns how many samples are buffered toward the next frame. Between
// Feed calls it is in [0, FrameSize); it briefly equals FrameSize inside an
// onFrame callback, before the post-callback slide by HopSize. A frame completes
// and is emitted when a Feed fills this to FrameSize. A caller predicting how many
// frames a Feed will emit uses this: with InFill()+len(src) >= FrameSize, the
// count is 1 + (InFill()+len(src)-FrameSize)/HopSize.
func (a *Analyzer) InFill() int { return a.fill }

// Feed pushes src into the analysis buffer and, for every frame that completes,
// runs the windowed real-input transform and its magnitude-squared power, then
// calls onFrame with that frame's Hermitian half-spectrum (len NumBins) and power
// (len NumBins). Both slices are Analyzer-owned and valid only until onFrame
// returns; onFrame may modify spec in place (for example to apply a per-bin gain
// before Inverse), and nothing reads spec or power after it returns. Frames
// complete on hop boundaries; a Feed that does not complete a frame calls onFrame
// zero times. onFrame must not be nil. Allocation-free when onFrame does not
// allocate or let its arguments escape. A consumer that does not read power should
// call FeedSpectrum, which skips the power pass.
func (a *Analyzer) Feed(src []float32, onFrame func(spec []complex64, power []float32)) {
	for {
		var ok bool
		if src, ok = a.next(src); !ok {
			return
		}
		c64.AbsSq(a.power, a.spec)
		onFrame(a.spec, a.power)
		a.slide()
	}
}

// FeedSpectrum is Feed without the magnitude-squared pass: it buffers, frames and
// transforms identically, and calls onFrame with the Hermitian half-spectrum (len
// NumBins) of every frame that completes. The ownership, ordering, in-place
// modification, InFill and allocation rules are those of Feed. It suits consumers
// that read the spectrum itself (for example its real part) and never the power.
// Feed and FeedSpectrum share the framing state, so they may be mixed on one
// stream.
func (a *Analyzer) FeedSpectrum(src []float32, onFrame func(spec []complex64)) {
	for {
		var ok bool
		if src, ok = a.next(src); !ok {
			return
		}
		onFrame(a.spec)
		a.slide()
	}
}

// next copies src into the analysis buffer. When that completes a frame it
// transforms it into a.spec and returns the unconsumed rest of src with ok true;
// otherwise it returns ok false and the buffer holds all of src. It is a method
// rather than a closure-taking helper so the caller's onFrame stays a direct,
// non-escaping call.
func (a *Analyzer) next(src []float32) (rest []float32, ok bool) {
	if len(src) == 0 {
		return src, false
	}
	took := copy(a.buf[a.fill:], src) // copies min(a.n-a.fill, len(src))
	a.fill += took
	if a.fill < a.n {
		return nil, false
	}
	a.p.RFFT(a.spec, a.buf, a.window)
	return src[took:], true
}

// slide advances the analysis buffer by one hop after a frame was emitted.
func (a *Analyzer) slide() {
	copy(a.buf, a.buf[a.hop:])
	a.fill -= a.hop
}

// Inverse transforms a Hermitian half-spectrum spec back to at most FrameSize
// real time samples, writing into dst and returning the count. It is the inverse
// of the transform Feed runs (spec scaled by 1/FrameSize), so a consumer can
// modify the spec handed to onFrame and resynthesize the frame. Allocation-free;
// it reuses the plan scratch, so it must not run concurrently with Feed on the
// same Analyzer.
func (a *Analyzer) Inverse(dst []float32, spec []complex64) int {
	return a.p.IRFFT(dst, spec)
}

// Reset clears the analysis buffer so the next Feed starts a new stream. The
// configured window is kept.
func (a *Analyzer) Reset() {
	clear(a.buf)
	a.fill = 0
}
