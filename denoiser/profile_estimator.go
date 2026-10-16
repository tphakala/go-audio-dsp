package denoiser

import (
	"errors"
	"math"
	"math/bits"
	"slices"

	"github.com/tphakala/go-audio-dsp/pcm"
	"github.com/tphakala/simd/f32"
)

// errWindowEvicted reports that the samples of the selected window were no
// longer held. The eviction rule makes it unreachable; it exists so a bug
// shows up as an error from Profile rather than a wrong profile or a panic.
var errWindowEvicted = errors.New("denoiser: noise profile window no longer held")

// candidate tracks the samples of the tracker's best window. The window's
// widened range always lies inside the region [rs, re) (absolute sample
// positions). While resident the samples are still in the estimator's ring;
// when the ring is about to overwrite them they are copied into snap.
type candidate struct {
	active, resident bool
	rs, re           int
	snap             []float32
}

// NoiseProfileEstimator measures the same noise profile as
// Denoiser.EstimateNoiseProfile from a clip supplied in chunks of any size,
// as float32 or int16, without ever holding the clip. Create one with
// Denoiser.NewNoiseProfileEstimator, feed it with AddFloat32 and AddInt16,
// and call Profile once the clip is complete (or earlier, for a provisional
// result).
//
// Profile returns exactly what EstimateNoiseProfile returns for the
// concatenation of everything added: the same error, the same spectrum and
// the same Info. Memory is bounded and independent of chunk sizes: a sample
// ring and one window snapshot fixed by the Denoiser's configuration
// (about 0.3 MB at 48 kHz with default framing) plus one float32 per hop block
// of the clip (225 KB for 300 s at 48 kHz). The input is copied, so the caller
// may reuse its chunk buffer as soon as an Add method returns.
//
// An estimator, like the Denoiser it came from, is not safe for concurrent
// use, and Profile must not run concurrently with that Denoiser's processing.
type NoiseProfileEstimator struct {
	d *Denoiser

	hop, n int
	w0     int // window length in hop blocks before it is capped by the clip length
	span   int // w0*hop
	margin int // widening margin: max(0, n-span)
	region int // span + 2*margin, the size of a candidate region

	ring []float32 // len 2*region, block aligned: sample p lives at ring[p%len(ring)]
	pos  int       // samples added since construction or Reset

	energy  []float32 // mean square per completed hop block
	tracker windowTracker
	cand    candidate // follows tracker's best window

	sorted, wrap []float32 // Profile scratch, reused across calls
}

// NewNoiseProfileEstimator returns an empty estimator for this Denoiser's
// FrameSize and sample rate, so its profile is accepted by SetNoiseProfile.
func (d *Denoiser) NewNoiseProfileEstimator() *NoiseProfileEstimator {
	e := d.newNoiseProfileEstimator(0)
	e.ring = make([]float32, 2*e.region)
	return e
}

// newNoiseProfileEstimator builds an estimator without its sample ring, with a
// capacity hint for the per-block energy array. EstimateNoiseProfile uses it
// as is: with the whole clip in hand it reads window samples from the caller's
// slice, so it needs the block statistics but neither the ring nor snapshots.
func (d *Denoiser) newNoiseProfileEstimator(blocks int) *NoiseProfileEstimator {
	w0 := max(1, int(math.Round(autoWindowSec*float64(d.sampleRate)/float64(d.hop))))
	span := w0 * d.hop
	margin := max(0, d.n-span)
	region := span + 2*margin
	return &NoiseProfileEstimator{
		d: d, hop: d.hop, n: d.n, w0: w0, span: span, margin: margin, region: region,
		energy:  make([]float32, 0, blocks),
		tracker: newWindowTracker(w0),
	}
}

// Len returns the number of samples added since construction or the last Reset.
func (e *NoiseProfileEstimator) Len() int { return e.pos }

// Reset discards everything added and makes the estimator behave like a new
// one, keeping its buffers for the next clip.
func (e *NoiseProfileEstimator) Reset() {
	e.pos = 0
	e.energy = e.energy[:0]
	e.tracker.reset()
	// Field by field: assigning a literal would drop snap and make the next
	// eviction allocate it again.
	e.cand.active, e.cand.resident = false, false
	e.cand.rs, e.cand.re = 0, 0
}

// AddFloat32 appends x to the clip. Chunks may have any length, including zero.
func (e *NoiseProfileEstimator) AddFloat32(x []float32) {
	for len(x) > 0 {
		k := e.reserve(len(x))
		off := e.pos % len(e.ring)
		copy(e.ring[off:off+k], x[:k])
		e.commit(k)
		x = x[k:]
	}
}

// AddInt16 appends x, converted to float32 exactly as pcm.Int16ToFloat32 does
// (scale 1/32768), so the result equals converting the whole clip first.
func (e *NoiseProfileEstimator) AddInt16(x []int16) {
	for len(x) > 0 {
		k := e.reserve(len(x))
		off := e.pos % len(e.ring)
		pcm.Int16ToFloat32(e.ring[off:off+k], x[:k])
		e.commit(k)
		x = x[k:]
	}
}

// reserve returns how many of the next want samples can be written without
// crossing a hop-block boundary, and first snapshots the candidate region if it
// is resident and the write would start to overwrite it.
func (e *NoiseProfileEstimator) reserve(want int) int {
	k := min(want, e.hop-e.pos%e.hop)
	oldest := e.pos + k - len(e.ring) // first sample position the write leaves in the ring
	if c := &e.cand; c.active && c.resident && c.rs < oldest {
		if c.snap == nil {
			c.snap = make([]float32, 0, e.region)
		}
		// c.re <= e.pos here (the region is complete), so the ring holds [rs, re).
		c.snap = e.appendRing(c.snap[:0], c.rs, c.re)
		c.resident = false
	}
	return k
}

// appendRing appends the ring contents of absolute positions [from, to) to dst.
func (e *NoiseProfileEstimator) appendRing(dst []float32, from, to int) []float32 {
	c := len(e.ring)
	for from < to {
		off := from % c
		k := min(to-from, c-off)
		dst = append(dst, e.ring[off:off+k]...)
		from += k
	}
	return dst
}

// commit accounts for k samples just written; when they complete a hop block
// it records the block energy and advances the window tracker.
func (e *NoiseProfileEstimator) commit(k int) {
	e.pos += k
	if e.pos%e.hop != 0 {
		return
	}
	off := (e.pos - e.hop) % len(e.ring)
	e.addBlock(f32.SumOfSquares(e.ring[off:off+e.hop]) / float32(e.hop))
}

// addBlock records the mean-square energy of the next hop block and advances
// the window tracker and candidate.
func (e *NoiseProfileEstimator) addBlock(blockEnergy float32) {
	e.energy = append(e.energy, blockEnergy)
	if e.tracker.push(e.energy) {
		e.setCandidate(e.tracker.best)
	}
}

// setCandidate points the candidate at the region of the window starting at
// block best.
func (e *NoiseProfileEstimator) setCandidate(best int) {
	c := &e.cand
	c.active, c.resident = true, true
	c.rs = max(0, best*e.hop-e.margin)
	c.re = best*e.hop + e.span + e.margin
}

// Profile returns the noise profile of everything added so far, with the
// contract of Denoiser.EstimateNoiseProfile: ErrProfileTooShort below one
// frame, ErrNoQuietRegion when no usable quiet window exists (see
// EstimateNoiseProfile). It does not change the estimator, so it may be called
// repeatedly and between Add calls.
func (e *NoiseProfileEstimator) Profile() (*NoiseProfile, error) {
	best, w, held, err := e.selectWindow()
	if err != nil {
		return nil, err
	}
	start, end, err := e.measuredRange(best, w)
	if err != nil {
		return nil, err
	}
	samples, err := e.windowSamples(start, end, held)
	if err != nil {
		return nil, err
	}
	return e.buildProfile(samples, start, end)
}

// selectWindow picks the quietest window free of digital silence among the
// blocks seen so far and applies the acceptance test. It returns the start
// block, the window length in blocks and whether the candidate holds the
// window's samples (false for a clip shorter than the window, which the ring
// holds whole). It is the selection half of EstimateNoiseProfile, shared by the
// whole-buffer and streaming paths.
func (e *NoiseProfileEstimator) selectWindow() (best, w int, held bool, err error) {
	if e.pos < e.n {
		return 0, 0, false, ErrProfileTooShort
	}
	nb := len(e.energy)
	w = min(e.w0, nb)

	var bestSum float64
	if nb >= e.w0 {
		best, bestSum, held = e.tracker.best, e.tracker.bestSum, true
	} else {
		best, bestSum = quietestWindow(e.energy, w)
	}
	if best < 0 || !e.accepts(w, bestSum) {
		return 0, 0, false, ErrNoQuietRegion
	}
	return best, w, held, nil
}

// buildProfile measures the profile of samples, which cover [start, end) of the clip.
func (e *NoiseProfileEstimator) buildProfile(samples []float32, start, end int) (*NoiseProfile, error) {
	p, err := e.d.NoiseProfileFromSamples(samples)
	if err != nil {
		return nil, err
	}
	p.info = ProfileInfo{
		Source:      ProfileAuto,
		StartSec:    float64(start) / float64(e.d.sampleRate),
		DurationSec: float64(end-start) / float64(e.d.sampleRate),
	}
	return p, nil
}

// accepts reports whether the window of w blocks with energy sum bestSum is
// quietMarginDB below the quietPercentile of the non-silent block energies
// (always true for clips with fewer than 2*w such blocks).
func (e *NoiseProfileEstimator) accepts(w int, bestSum float64) bool {
	e.sorted = slices.Grow(e.sorted[:0], len(e.energy))
	for _, v := range e.energy {
		if v >= silencePower {
			e.sorted = append(e.sorted, v)
		}
	}
	if len(e.sorted) < 2*w {
		return true
	}
	ref := selectNth(e.sorted, int(quietPercentile*float64(len(e.sorted)-1)))
	winMean := bestSum / float64(w)
	return winMean <= float64(ref)*math.Pow(10, -quietMarginDB/10)
}

// selectNth returns the value that sorted order puts at index k of a, reordering a
// in the process. It is an introselect: quickselect on a median-of-three pivot
// with a Hoare partition, falling back to a sort of the remaining range after
// 2*log2(n) rounds so adversarial or pathological (sorted, repeated) input stays
// O(n log n). On a long clip the full sort of the block energies was about 45% of
// EstimateNoiseProfile. k must be in [0, len(a)).
func selectNth(a []float32, k int) float32 {
	lo, hi := 0, len(a)-1
	for budget := 2 * bits.Len(uint(len(a))); lo < hi; budget-- {
		if budget == 0 {
			slices.Sort(a[lo : hi+1])
			break
		}
		mid := lo + (hi-lo)/2
		// Order a[lo] <= a[mid] <= a[hi]; the median sits at mid.
		if a[mid] < a[lo] {
			a[mid], a[lo] = a[lo], a[mid]
		}
		if a[hi] < a[lo] {
			a[hi], a[lo] = a[lo], a[hi]
		}
		if a[hi] < a[mid] {
			a[hi], a[mid] = a[mid], a[hi]
		}
		pivot := a[mid]
		i, j := lo, hi
		for i <= j {
			for a[i] < pivot {
				i++
			}
			for pivot < a[j] {
				j--
			}
			if i <= j {
				a[i], a[j] = a[j], a[i]
				i++
				j--
			}
		}
		// Now a[lo..j] <= pivot <= a[i..hi], and anything between j and i equals it.
		switch {
		case k <= j:
			hi = j
		case k >= i:
			lo = i
		default:
			return a[k]
		}
	}
	return a[k]
}

// measuredRange returns the sample range [start, end) measured for the window
// of w blocks starting at block best, or ErrNoQuietRegion when a block in it is
// digital silence. The range is cut at the last whole hop block, so it lies on
// the hop grid and every block in it has a recorded energy. The window can be
// silence-free while widening it to one frame reaches a silent block, and a
// silent block drags the measured profile below the real noise (to the power
// floor for an all-zero range), so the range is checked, not only the window.
func (e *NoiseProfileEstimator) measuredRange(best, w int) (start, end int, err error) {
	start, end = profileRange(best, w, e.hop, e.n, len(e.energy)*e.hop)
	for _, v := range e.energy[start/e.hop : end/e.hop] {
		if v < silencePower {
			return 0, 0, ErrNoQuietRegion
		}
	}
	return start, end, nil
}

// profileRange returns the sample range [start, end) measured for a window of
// w blocks starting at block best in a clip of total samples: the window
// itself, widened to one frame when it is shorter (frames longer than the
// window). Callers pass a total that is a whole number of hop blocks, which
// keeps start and end on the hop grid.
func profileRange(best, w, hop, frame, total int) (start, end int) {
	start = best * hop
	end = min(total, start+w*hop)
	if end-start < frame {
		end = min(total, start+frame)
		start = max(0, end-frame)
	}
	return start, end
}

// windowSamples returns the samples [start, end) from the ring or, when the
// ring has moved past them and held is set, from the candidate's snapshot. The result aliases
// internal memory and is only valid until the next Add or Profile call.
func (e *NoiseProfileEstimator) windowSamples(start, end int, held bool) ([]float32, error) {
	c := len(e.ring)
	if start >= e.pos-c {
		off := start % c
		if off+end-start <= c {
			return e.ring[off : off+end-start], nil
		}
		e.wrap = e.appendRing(e.wrap[:0], start, end)
		return e.wrap, nil
	}
	if cd := &e.cand; held && cd.active && !cd.resident && cd.rs <= start && end <= cd.re {
		return cd.snap[start-cd.rs : end-cd.rs], nil
	}
	return nil, errWindowEvicted
}
