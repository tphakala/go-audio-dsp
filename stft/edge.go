package stft

import (
	"fmt"
	"math"

	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// edgeCacheWindows caps how many edge windows each cache slab (start-edge and
// drain) keeps. A frame needing a window past the cap is evaluated into a working
// buffer for that frame, so the output never depends on the cap; only time does.
// It bounds the analyzer's window memory at (2*edgeCacheWindows + 2) * FrameSize
// float32 values whatever the hop or the input length.
const edgeCacheWindows = 32

// EdgeConfig configures an EdgeAnalyzer.
type EdgeConfig struct {
	// FrameSize is the FFT length in samples, a power of two >= 2 (at most 32768
	// for ShapeDolphChebyshev).
	FrameSize int
	// Shape is the window family and its parameter. The zero value is ShapeHann.
	Shape WindowShape
}

// EdgeAnalyzer is a streaming analyzer whose frame schedule and edge handling
// follow the sox 14.4.2 spectrogram effect, the behavioural reference. The
// reference is named for behaviour only; this is an independent implementation
// of the model below.
//
// With hop S and frame size N the analyzed stream is ceil((N-S)/2) lead zeros, the
// signal, then drain zeros added by Finish: floor((N-S)/2), plus S-r more when
// r = L mod S is at least floor(S/2), for L signal samples. Frame k is
// stream[k*S : k*S+N], so frames are centred on hop blocks and the frame count is
// floor(L/S), plus one when r >= floor(S/2) (see NumFrames).
//
// A frame that includes padding does not use the full window. It uses the
// symmetric window of the shorter length that fits the real samples, so the
// window ends at the signal edge instead of cutting through a full-length one:
//
//   - A frame with drain zeros (d >= 1 of them) uses the m = N-d+1 point window
//     at slot 0, zero after, whatever lead zeros it still holds. A clip shorter
//     than a frame therefore has drain-rule frames only.
//   - Any other frame with e lead zeros uses the m = N-e+1 point window with its
//     first m-1 points at slots [e, N); with e = 0 that is the full window.
//
// Each window is scaled by 2/sum * ((m-1)/N)^2, where sum adds the points the
// frame uses (m-1 for a lead frame, m for a drain frame). A full-scale sinusoid
// centred on a bin therefore reads power 1 (0 dBFS) in a frame with no padding, up
// to the leakage of its negative-frequency image. That leakage grows toward DC and
// Nyquist and can be larger at low attenuation, and it adds to or subtracts from
// the reading depending on the sinusoid's phase in the frame. Measured for a cosine
// whose image adds to the bin: below 0.01 dB at bin N/8 for Hann and for
// Dolph-Chebyshev at 86.4 dB and above (N from 256 to 2048); +1.05 dB at bin 1 of
// 256 for Dolph-Chebyshev at 86.4 dB; +0.21 dB at bin 32 of 256 for 26.1 dB; +6.02
// dB at DC and Nyquist, where the image lands on the bin.
// This is an amplitude normalization, unlike the energy normalization of the
// spectrogram package's Spectrogram.
//
// The Dolph-Chebyshev shape is normalized so its centre sample is 1, not its peak
// (DolphChebyshevSymmetric normalizes the peak); the two differ only where the window
// peaks away from its centre (below about 50 dB at 1025 points and 40 dB at 257; the
// threshold rises with the frame size), and FullWindowSum depends on the choice.
//
// Edge windows depend only on the hop, the shape and, for drain frames, on the
// drain geometry, so the analyzer caches them across Reset calls: the start-edge
// windows are rebuilt only when the hop changes, and the drain windows only when
// the hop, the first drain length or the drain frame count changes (for all but
// very short clips, when L mod S changes). A repeat run on the same geometry
// builds no window. At most 32 windows are cached per kind; a frame needing one
// beyond that evaluates it into a working buffer. Cache state never changes the
// output.
//
// Reset(hop) starts a stream; Feed pushes samples; Finish appends the drain and
// ends the stream, after which Feed and Finish do nothing until the next Reset.
// Memory is bounded by (2*32+2)*FrameSize float32 window values plus fixed
// scratch, independent of the input length. Feed allocates nothing when onFrame
// does not; Finish and Reset allocate only when a window slab must grow, the first
// time a geometry needs more cached windows than any earlier run. Not safe for
// concurrent use.
type EdgeAnalyzer struct {
	p     *f32.STFTPlan
	n     int
	shape WindowShape

	fullSum    float64 // sum of the first N points of shape_(N+1), before the factor
	defaultHop int     // round(fullSum) clamped to [1, N]

	hop   int // current S
	z0    int // lead zeros, ceil((N-S)/2)
	buf   []float32
	fill  int // samples in buf, lead zeros included
	spec  []complex64
	power []float32

	full []float32 // full window with the factor; built once at construction
	work []float32 // window for a frame past a cache slab
	win  []float32 // window of the frame in flight; read by tests via the callback

	shape64 []float64 // shape_m scratch, N+1
	dolphP  []float64 // Dolph scratch, N/2+1; nil for Hann
	dolphT  []float64 // Dolph scratch, 2(N+1); nil for Hann

	lead      []float32 // leadCount windows of N, window k has e = z0-k*hop lead zeros
	leadHop   int
	leadValid bool
	leadCount int

	drain      []float32 // drainCount windows of N, window j has d = drainD0+j*hop drain zeros
	drainHop   int
	drainD0    int
	drainValid bool
	drainCount int

	consumed int64 // samples fed since Reset
	frame    int64 // frames emitted since Reset
	finished bool

	windowBuilds uint64 // shape evaluations, for the cache tests
}

// NewEdgeAnalyzer builds an EdgeAnalyzer for cfg, ready to stream with the
// default hop (see Reset). It returns an error wrapping ErrInvalidConfig for an
// invalid FrameSize or Shape.
func NewEdgeAnalyzer(cfg EdgeConfig) (*EdgeAnalyzer, error) {
	if err := checkFrameSize(cfg.FrameSize); err != nil {
		return nil, err
	}
	if err := cfg.Shape.validate(cfg.FrameSize); err != nil {
		return nil, err
	}
	n := cfg.FrameSize
	p, err := f32.NewSTFTPlan(n)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	bins := p.NumBins()
	a := &EdgeAnalyzer{
		p:       p,
		n:       n,
		shape:   cfg.Shape,
		buf:     make([]float32, n),
		spec:    make([]complex64, bins),
		power:   make([]float32, bins),
		full:    make([]float32, n),
		work:    make([]float32, n),
		shape64: make([]float64, n+1),
	}
	if cfg.Shape.Family == ShapeDolphChebyshev {
		a.dolphP = make([]float64, n/2+1)
		a.dolphT = make([]float64, 2*(n+1))
	}
	// The full window is shape_(N+1) without its last point. Building it here
	// also proves the shape is computable at the largest length the analyzer uses.
	sum, ok := a.buildWindow(a.full, n+1, n, 0)
	if !ok {
		return nil, fmt.Errorf("%w: window evaluation lost precision (FrameSize %d, attenuationDB %g)", ErrInvalidConfig, n, cfg.Shape.AttenuationDB)
	}
	a.fullSum = sum
	a.defaultHop = min(max(int(math.Floor(sum+0.5)), 1), n)
	if err := a.Reset(0); err != nil {
		return nil, err
	}
	return a, nil
}

// FrameSize returns the transform size in samples.
func (a *EdgeAnalyzer) FrameSize() int { return a.n }

// NumBins returns FrameSize/2 + 1, the Hermitian half-spectrum length and the
// length of the power slice handed to onFrame.
func (a *EdgeAnalyzer) NumBins() int { return a.p.NumBins() }

// HopSize returns the current frame advance in samples (the value Reset chose).
func (a *EdgeAnalyzer) HopSize() int { return a.hop }

// FullWindowSum returns the float64 sum of the full window, the first FrameSize
// points of the symmetric (FrameSize+1)-point shape, before the amplitude factor.
// Its rounded value is the default hop.
func (a *EdgeAnalyzer) FullWindowSum() float64 { return a.fullSum }

// Reset starts a new stream with frame advance hop in [1, FrameSize]. A hop of 0
// selects round(FullWindowSum) clamped to [1, FrameSize]. It keeps the
// configuration, and the window cache when the geometry allows (see the type
// documentation). On an invalid hop it returns an error wrapping
// ErrInvalidConfig and leaves the current stream untouched.
func (a *EdgeAnalyzer) Reset(hop int) error {
	if hop == 0 {
		hop = a.defaultHop
	}
	if hop < 1 || hop > a.n {
		return fmt.Errorf("%w: hop must be in [1, FrameSize=%d] (0 selects the default), got %d", ErrInvalidConfig, a.n, hop)
	}
	a.hop = hop
	a.z0 = (a.n - hop + 1) / 2
	a.prepareLead()
	clear(a.buf)
	a.fill = a.z0
	a.consumed, a.frame = 0, 0
	a.finished = false
	return nil
}

// prepareLead makes the start-edge slab hold the windows for the current hop.
func (a *EdgeAnalyzer) prepareLead() {
	a.leadCount = min((a.z0+a.hop-1)/a.hop, edgeCacheWindows)
	if a.leadValid && a.leadHop == a.hop {
		return
	}
	a.lead = growWindows(a.lead, a.leadCount*a.n)
	for k := range a.leadCount {
		a.buildFlow(a.lead[k*a.n:(k+1)*a.n], a.z0-k*a.hop)
	}
	a.leadHop, a.leadValid = a.hop, true
}

// NumFrames returns how many frames a stream of signalLen samples emits at the
// current hop: floor(L/S), plus one when L mod S >= floor(S/2). It is 0 for
// L < floor(S/2) and L+1 when S = 1. A negative length gives 0.
func (a *EdgeAnalyzer) NumFrames(signalLen int64) int64 {
	if signalLen < 0 {
		return 0
	}
	hop := int64(a.hop)
	f := signalLen / hop
	if signalLen%hop >= hop/2 {
		f++
	}
	return f
}

// drainLen returns the drain zero count for a stream of signalLen samples at the
// current hop.
func (a *EdgeAnalyzer) drainLen(signalLen int64) int {
	dz := (a.n - a.hop) / 2
	if r := int(signalLen % int64(a.hop)); r >= a.hop/2 {
		dz += a.hop - r
	}
	return dz
}

// Feed pushes src into the stream and calls onFrame with the power spectrum
// (length NumBins) of every frame that completes. The slice is analyzer-owned and
// valid only until onFrame returns. A Feed that completes no frame calls onFrame
// zero times; chunk sizes do not change the result. onFrame must not be nil.
// Allocation-free when onFrame does not allocate or let its argument escape. After
// Finish it does nothing until Reset.
func (a *EdgeAnalyzer) Feed(src []float32, onFrame func(power []float32)) {
	if a.finished {
		return
	}
	a.consumed += int64(len(src))
	for len(src) > 0 {
		took := copy(a.buf[a.fill:], src) // copies min(a.n-a.fill, len(src))
		a.fill += took
		src = src[took:]
		if a.fill < a.n {
			return
		}
		a.emit(a.flowWindow(), onFrame)
		a.frame++
		copy(a.buf, a.buf[a.hop:])
		a.fill -= a.hop
	}
}

// Finish appends the drain zeros, emits the frames they complete with the drain
// window rule, and ends the stream: Feed and Finish do nothing until Reset. The
// frames emitted over the whole stream number NumFrames(total samples fed).
func (a *EdgeAnalyzer) Finish(onFrame func(power []float32)) {
	if a.finished {
		return
	}
	a.finished = true
	// The first drain frame fires once the buffer holds d0 zeros; every later one
	// holds hop more, until the drain zeros run out.
	d0 := a.n - a.fill
	dz := a.drainLen(a.consumed)
	nDrain := 0
	if dz >= d0 {
		nDrain = (dz-d0)/a.hop + 1
	}
	a.prepareDrain(d0, nDrain)

	clear(a.buf[a.fill:])
	for j := range nDrain {
		var w []float32
		if j < a.drainCount {
			w = a.drain[j*a.n : (j+1)*a.n]
		} else {
			a.buildDrain(a.work, d0+j*a.hop)
			w = a.work
		}
		a.emit(w, onFrame)
		a.frame++
		copy(a.buf, a.buf[a.hop:])
		clear(a.buf[a.n-a.hop:])
	}
}

// prepareDrain makes the drain slab hold the windows for a drain of nDrain frames
// whose first holds d0 zeros. Window j depends on the hop and d0+j*hop alone, so
// the key is the hop, d0 and how many windows the slab holds.
func (a *EdgeAnalyzer) prepareDrain(d0, nDrain int) {
	if nDrain == 0 {
		return // no drain frame; keep the slab for the next stream
	}
	count := min(nDrain, edgeCacheWindows)
	if a.drainValid && a.drainHop == a.hop && a.drainD0 == d0 && a.drainCount == count {
		return
	}
	a.drain = growWindows(a.drain, count*a.n)
	for j := range count {
		a.buildDrain(a.drain[j*a.n:(j+1)*a.n], d0+j*a.hop)
	}
	a.drainHop, a.drainD0, a.drainCount, a.drainValid = a.hop, d0, count, true
}

// flowWindow returns the window of the next frame completed from signal samples
// alone: the full window, a cached start-edge window, or one built for this frame.
func (a *EdgeAnalyzer) flowWindow() []float32 {
	e := int64(a.z0) - a.frame*int64(a.hop)
	switch {
	case e <= 0:
		return a.full
	case a.frame < int64(a.leadCount):
		return a.lead[int(a.frame)*a.n : (int(a.frame)+1)*a.n]
	}
	a.buildFlow(a.work, int(e))
	return a.work
}

// emit transforms the frame in buf with window w and calls onFrame with its power.
// The simd plan accepts a different window on every call (simd v1.12.0
// f32/stft.go:628-640, RFFT).
func (a *EdgeAnalyzer) emit(w []float32, onFrame func(power []float32)) {
	a.win = w
	a.p.RFFT(a.spec, a.buf, w)
	c64.AbsSq(a.power, a.spec)
	onFrame(a.power)
}

// buildFlow builds the window of a frame with e > 0 lead zeros and no drain.
func (a *EdgeAnalyzer) buildFlow(dst []float32, e int) {
	m := a.n - e + 1
	_, _ = a.buildWindow(dst, m, m-1, e)
}

// buildDrain builds the window of a frame holding d >= 1 drain zeros.
func (a *EdgeAnalyzer) buildDrain(dst []float32, d int) {
	m := a.n - d + 1
	_, _ = a.buildWindow(dst, m, m, 0)
}

// buildWindow writes into dst (length N) the m-point shape's first points samples
// at slots [at, at+points), zero elsewhere, scaled by 2/sum * ((m-1)/N)^2 with sum
// over those points in float64. It returns that sum and whether the shape
// evaluated. A non-positive or non-finite sum gives an all-zero window, so no NaN
// or Inf reaches the transform; a shape that cannot be evaluated does the same
// (shapes accepted by NewEdgeAnalyzer are evaluable at its full-window length).
func (a *EdgeAnalyzer) buildWindow(dst []float32, m, points, at int) (sum float64, ok bool) {
	clear(dst)
	if !a.evalShape(m) {
		return 0, false
	}
	sum = sumFloat64(a.shape64[:points])
	if !(sum > 0) || math.IsInf(sum, 1) {
		return sum, true
	}
	ratio := float64(m-1) / float64(a.n)
	amp := 2 / sum * (ratio * ratio)
	if math.IsInf(amp, 0) || math.IsNaN(amp) {
		return sum, true
	}
	for i, v := range a.shape64[:points] {
		dst[at+i] = float32(v * amp)
	}
	return sum, true
}

// evalShape evaluates the symmetric m-point shape into a.shape64[:m] and reports
// whether the evaluation stayed finite.
func (a *EdgeAnalyzer) evalShape(m int) bool {
	a.windowBuilds++
	w := a.shape64[:m]
	switch a.shape.Family {
	case ShapeHann:
		hannSymmetric64(w)
		return true
	case ShapeDolphChebyshev:
		return dolphCentreInto(w, a.dolphP, a.dolphT, a.shape.AttenuationDB) == nil
	}
	return false
}

// sumFloat64 adds v in order. The sum stays scalar on purpose: it sets the
// default hop through rounding, which must not depend on the simd kernel tier.
func sumFloat64(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s
}

// growWindows returns s resized to n values, reusing its capacity when it can.
func growWindows(s []float32, n int) []float32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}
