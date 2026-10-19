package stft

import (
	"fmt"
	"math"

	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
	"github.com/tphakala/simd/f64"
)

// edgeCacheWindows caps how many edge windows an analyzer references per kind
// (start-edge and drain). A frame needing a window past the cap is evaluated into
// a working buffer for that frame, so the output never depends on the cap; only
// time does. The windows an analyzer references are pinned even if the shared
// cache evicts them, which bounds that memory at (2*edgeCacheWindows + 2) *
// FrameSize float32 values whatever the hop or the input length (the full window,
// the working buffer and the two kinds).
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
//     at slot 0, zero after, whatever lead zeros it still holds.
//   - Any other frame with e lead zeros uses the m = N-e+1 point window with its
//     first m-1 points at slots [e, N); with e = 0 that is the full window.
//
// Each window is scaled by 2/sum * ((m-1)/N)^2, where sum adds the points the
// frame uses (m-1 for a lead frame, m for a drain frame). A full-scale sinusoid
// centred on a bin therefore reads close to power 1 (0 dBFS) in a frame with no
// padding. The reading is off by the leakage of the sinusoid's negative-frequency
// image, which depends on its phase and grows toward DC and Nyquist and at low
// attenuation; at DC and Nyquist the image lands on the bin itself.
// This is an amplitude normalization, unlike the energy normalization of the
// spectrogram package's Spectrogram.
//
// The Dolph-Chebyshev shape is normalized so its centre sample is 1, not its peak
// (DolphChebyshevSymmetric normalizes the peak); the two differ at low attenuation,
// where the window peaks at its endpoints, and FullWindowSum depends on the choice.
//
// An edge window depends only on the frame size, the shape, its length m and
// whether it is a lead or a drain window, not on the hop or the clip length. Every
// EdgeAnalyzer in the process takes its windows from one shared cache keyed that
// way (bounded at 16 MiB of window data, least recently used evicted), so an
// analyzer builds none of the windows still cached from an earlier build, and two
// hops that need a window of the same length share it. The analyzer also
// remembers the windows of its current geometry, so a repeat run does no lookup.
// At most 32 windows are referenced per kind; a frame needing one beyond that
// evaluates it into a working buffer. Cache state never changes the output.
//
// Reset(hop) starts a stream; Feed pushes samples; Finish appends the drain and
// ends the stream, after which Feed and Finish do nothing until the next Reset.
// Pinned window memory is bounded by (2*32+2)*FrameSize float32 values plus fixed
// scratch, independent of the input length. Feed allocates nothing when onFrame
// does not; Reset and Finish allocate only when a window they need is not in the
// shared cache (first use in the process, or after eviction). A Dolph-Chebyshev
// analyzer allocates its evaluation plan and scratch once, at its first window
// build or, when Reset sees a geometry that can need a window past the cap
// (FrameSize/2 > 32*hop), in that Reset, so Feed never allocates it. One analyzer
// is not safe for concurrent use; distinct analyzers may run in different
// goroutines, because the shared cache is synchronized and its windows are
// read-only.
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

	cache *edgeWindowCache // where edge windows are shared; the package-wide one outside tests

	full []float32 // full window with the factor; shared and read-only
	work []float32 // private window for a frame past the per-kind window cap
	win  []float32 // window of the frame in flight; read by tests via the callback

	shape64   []float64    // shape_m scratch, N+1
	dolphRe   []float64    // Dolph DFT scratch, N+1; nil until ensureDolph
	dolphIm   []float64    // Dolph DFT scratch, N+1; nil until ensureDolph
	dolphPlan *f64.DFTPlan // Dolph DFT plan of capacity N+1; nil until ensureDolph

	lead      [][]float32 // leadCount shared windows of N, window k has e = z0-k*hop lead zeros
	leadHop   int
	leadValid bool
	leadCount int

	drain      [][]float32 // drainCount shared windows of N, window j has d = drainD0+j*hop drain zeros
	drainHop   int
	drainD0    int
	drainValid bool
	drainCount int

	consumed int64 // samples fed since Reset
	frame    int64 // frames emitted since Reset
	finished bool

	windowBuilds uint64 // shape evaluations this analyzer ran, for the cache tests
}

// NewEdgeAnalyzer builds an EdgeAnalyzer for cfg, ready to stream with the
// default hop (see Reset). It returns an error wrapping ErrInvalidConfig for an
// invalid FrameSize or Shape. Windows already in the shared cache are reused, so
// construction evaluates the full window only when it is not cached: the first
// time a configuration is seen in the process, or after eviction.
func NewEdgeAnalyzer(cfg EdgeConfig) (*EdgeAnalyzer, error) {
	return newEdgeAnalyzer(cfg, sharedEdgeWindows)
}

// newEdgeAnalyzer is NewEdgeAnalyzer over an explicit window cache, so tests can
// count builds against a private one.
func newEdgeAnalyzer(cfg EdgeConfig, cache *edgeWindowCache) (*EdgeAnalyzer, error) {
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
		cache:   cache,
		buf:     make([]float32, n),
		spec:    make([]complex64, bins),
		power:   make([]float32, bins),
		work:    make([]float32, n),
		shape64: make([]float64, n+1),
		lead:    make([][]float32, 0, edgeCacheWindows),
		drain:   make([][]float32, 0, edgeCacheWindows),
	}
	// The full window is shape_(N+1) without its last point. Building it here
	// also proves the shape is computable at the largest length the analyzer uses.
	fw := a.window(n+1, false)
	if !fw.ok {
		return nil, fmt.Errorf("%w: window evaluation lost precision (FrameSize %d, attenuationDB %g)", ErrInvalidConfig, n, cfg.Shape.AttenuationDB)
	}
	a.full, a.fullSum = fw.w, fw.sum
	a.defaultHop = min(max(int(math.Floor(fw.sum+0.5)), 1), n)
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
// configuration and reuses the windows of the current geometry; any others come
// from the shared cache, or are built when not cached (see the type
// documentation). On an invalid hop it
// returns an error wrapping ErrInvalidConfig and leaves the current stream
// untouched.
func (a *EdgeAnalyzer) Reset(hop int) error {
	if hop == 0 {
		hop = a.defaultHop
	}
	if hop < 1 || hop > a.n {
		return fmt.Errorf("%w: hop must be in [1, FrameSize=%d] (0 selects the default), got %d", ErrInvalidConfig, a.n, hop)
	}
	// A frame past the per-kind window cap evaluates its window inside Feed or
	// Finish, where nothing may allocate, so the Dolph plan has to exist by then
	// even when every cacheable window comes from the shared cache.
	if a.n/2 > edgeCacheWindows*hop {
		if err := a.ensureDolph(); err != nil {
			return err
		}
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

// ensureDolph allocates the Dolph DFT plan and scratch on first use; it does
// nothing for other shapes or once they exist. The plan has capacity N+1, the
// longest shape the analyzer evaluates, and serves every shorter length without
// allocating (simd f64/dft.go:26-28, 96 at v1.14.0). Allocating is allowed only
// where a window can be missing from the shared cache (the constructor, Reset,
// Finish) or where Reset has run it ahead of a past-cap frame.
//
// Reset runs it ahead when N/2 > 32*hop, a bound that covers both kinds of
// past-cap window. Lead: ceil(z0/hop) windows are referenced and z0 =
// (N-hop+1)/2 <= N/2, so more than 32 means z0 > 32*hop. Drain: the drain zeros
// number floor((N-hop)/2) plus at most ceil(hop/2), which is at most N/2 for even
// N, and the first drain frame needs d0 >= 1 of them (Finish), so there are at
// most (N/2-1)/hop + 1 drain frames, more than 32 only if N/2 > 32*hop.
func (a *EdgeAnalyzer) ensureDolph() error {
	if a.shape.Family != ShapeDolphChebyshev || a.dolphPlan != nil {
		return nil
	}
	plan, err := f64.NewDFTPlan(a.n + 1)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	a.dolphPlan = plan
	a.dolphRe = make([]float64, a.n+1)
	a.dolphIm = make([]float64, a.n+1)
	return nil
}

// prepareLead points a.lead at the shared start-edge windows for the current hop.
func (a *EdgeAnalyzer) prepareLead() {
	a.leadCount = min((a.z0+a.hop-1)/a.hop, edgeCacheWindows)
	if a.leadValid && a.leadHop == a.hop {
		return
	}
	a.lead = a.lead[:a.leadCount]
	for k := range a.leadCount {
		a.lead[k] = a.window(a.n-(a.z0-k*a.hop)+1, false).w
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
			w = a.drain[j]
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

// prepareDrain points a.drain at the shared windows for a drain of nDrain frames
// whose first holds d0 zeros. Window j depends on d0+j*hop alone, so the
// analyzer's own key is the hop, d0 and how many windows it references.
func (a *EdgeAnalyzer) prepareDrain(d0, nDrain int) {
	if nDrain == 0 {
		return // no drain frame; keep the references for the next stream
	}
	count := min(nDrain, edgeCacheWindows)
	if a.drainValid && a.drainHop == a.hop && a.drainD0 == d0 && a.drainCount == count {
		return
	}
	a.drain = a.drain[:count]
	for j := range count {
		a.drain[j] = a.window(a.n-(d0+j*a.hop)+1, true).w
	}
	a.drainHop, a.drainD0, a.drainCount, a.drainValid = a.hop, d0, count, true
}

// flowWindow returns the window of the next frame completed from signal samples
// alone: the full window, a shared start-edge window, or one built for this frame.
func (a *EdgeAnalyzer) flowWindow() []float32 {
	e := int64(a.z0) - a.frame*int64(a.hop)
	switch {
	case e <= 0:
		return a.full
	case a.frame < int64(a.leadCount):
		return a.lead[a.frame]
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

// windowLayout returns how many of the m-point shape's leading points a window
// uses and the slot they start at: m points at slot 0 for a drain window, m-1
// points ending at the frame's last slot for a lead window. The shared and the
// working-buffer paths both build through it, so the cache key and the content
// cannot disagree.
func (a *EdgeAnalyzer) windowLayout(m int, drain bool) (points, at int) {
	if drain {
		return m, 0
	}
	return m - 1, a.n - m + 1
}

// buildFlow builds the window of a frame with e > 0 lead zeros and no drain.
func (a *EdgeAnalyzer) buildFlow(dst []float32, e int) {
	a.buildWork(dst, a.n-e+1, false)
}

// buildDrain builds the window of a frame holding d >= 1 drain zeros.
func (a *EdgeAnalyzer) buildDrain(dst []float32, d int) {
	a.buildWork(dst, a.n-d+1, true)
}

// buildWork builds the m-point lead or drain window into dst, the reused working
// buffer, clearing first because dst still holds the previous past-cap window.
func (a *EdgeAnalyzer) buildWork(dst []float32, m int, drain bool) {
	clear(dst)
	points, at := a.windowLayout(m, drain)
	_, _ = a.buildWindow(dst, m, points, at)
}

// window returns the shared entry for the m-point lead or drain window, building
// it when no analyzer has. The returned entry is read-only. The full window is
// the lead rule at m = N+1.
func (a *EdgeAnalyzer) window(m int, drain bool) *edgeWindow {
	k := edgeKey{n: a.n, shape: a.shape, m: m, drain: drain}
	for {
		e, build := a.cache.acquire(k)
		if build {
			a.buildEntry(e, m, drain)
			return e
		}
		<-e.ready
		if e.done {
			return e
		}
		// The builder abandoned the entry; acquire again, possibly as the builder.
	}
}

// buildEntry fills and publishes e, which the caller acquired with build set. If
// the build panics, the deferred abandon removes the entry and wakes its waiters
// so they retry instead of blocking forever; the panic itself propagates.
func (a *EdgeAnalyzer) buildEntry(e *edgeWindow, m int, drain bool) {
	published := false
	defer func() {
		if !published {
			a.cache.abandon(e)
		}
	}()
	w := make([]float32, a.n) // fresh, so all zero as buildWindow requires
	points, at := a.windowLayout(m, drain)
	e.sum, e.ok = a.buildWindow(w, m, points, at)
	e.w, e.done = w, true
	published = true
	close(e.ready)
}

// buildWindow writes into dst (length N, all zero on entry) the m-point shape's
// first points samples at slots [at, at+points), scaled by 2/sum * ((m-1)/N)^2
// with sum over those points in float64. It returns that sum and whether the
// shape evaluated. A non-positive or non-finite sum leaves dst all zero, so no
// NaN or Inf reaches the transform; a shape that cannot be evaluated does the
// same (shapes accepted by NewEdgeAnalyzer are evaluable at its full-window
// length).
func (a *EdgeAnalyzer) buildWindow(dst []float32, m, points, at int) (sum float64, ok bool) {
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
		// A past-cap frame reaches here from Feed or Finish with the plan already
		// built by Reset; only a window missing from the shared cache allocates.
		if a.ensureDolph() != nil {
			return false
		}
		return dolphCentreInto(w, a.dolphRe, a.dolphIm, a.dolphPlan, a.shape.AttenuationDB) == nil
	}
	return false
}

// sumFloat64 adds v in order. The sum stays scalar on purpose: it sets the
// default hop through rounding, so it adds no tier dependence of its own. The
// Dolph window values it adds come from a simd DFT whose results are
// tolerance-stable, not bit-stable, across CPU tiers (simd f64/dft.go:27-28 at
// v1.14.0).
func sumFloat64(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s
}
