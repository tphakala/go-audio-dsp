package gate

import (
	"math"

	"github.com/tphakala/simd/f32"
)

// blindEvery is how often (in frames) the blind tracker refreshes the published
// per-bin median floor. The per-bin below-median counts maintained on every push
// make each refresh an O(1)-amortized rebalance rather than a full histogram
// rescan, so a refresh is cheap; the floor is still refreshed only every few
// frames because the median is stable frame to frame, not because a refresh costs
// much.
const blindEvery = 4

// Blind-tracker histogram geometry: nb buckets of floorBucketDB dB each, starting
// at floorBaseDB. This spans roughly -130 dB to +62 dB of per-bin power, which
// covers normalized-float32 audio with margin. The quantization is exact to
// floorBucketDB, well inside the several-dB soft knee, so it does not move the
// mask meaningfully.
const (
	floorBuckets   = 96
	floorBucketDB  = 2.0
	floorBaseDB    = -130.0
	floorMaxWindow = 65535 // a single bin's count must fit a uint16

	// floorQuantScale and floorQuantOffset map a base-10 log power to a bucket
	// index: bucket = 10*log10(p)/floorBucketDB - floorBaseDB/floorBucketDB.
	floorQuantScale  = 10.0 / floorBucketDB
	floorQuantOffset = -floorBaseDB / floorBucketDB
)

// floorTracker estimates a per-bin noise floor as a rolling median of log-power
// over a sliding window, the right estimator for gapless continuous noise (wind,
// rain, cicada drone) where a minimum tracker never opens. Each bin keeps a
// sliding-window histogram of quantized log-power buckets; the published floor is
// the median bucket's centre power, corrected by 1/ln2 onto the mean-power scale
// so ThresholdDB means the same as on the learned path (a periodogram bin of
// Gaussian noise is exponential, whose median is mean*ln2). The median bucket is
// tracked incrementally: each bin carries its current median bucket and the count
// of window samples strictly below it, so push adjusts those counts in O(1) and
// publish rebalances the median in O(1) amortized rather than rescanning the
// histogram. All state is allocated up front; push and publish allocate nothing
// and are chunk-independent.
type floorTracker struct {
	bins, nb, window, every int
	ring                    []uint8   // window*bins: the quantized bucket per (ring slot, bin)
	hist                    []uint16  // bins*nb: per-bin bucket counts over the window
	level                   []float32 // nb: bucket-centre power * (1/ln2), precomputed
	logp                    []float32 // bins: log10(power) scratch for one push
	noise                   []float32 // bins: the published floor the Gate reads

	med   []int // bins: current per-bin median bucket
	below []int // bins: count of window samples strictly below med[k] (== sum of hist[k][0..med[k]-1])

	head   int // next ring slot to write
	filled int // frames currently in the window (<= window)
	since  int // frames since the last publish
}

// newFloorTracker allocates a tracker for bins bins over a window of window frames
// (capped so a bucket count fits a uint16), publishing every every frames.
func newFloorTracker(bins, window, every int) *floorTracker {
	window = min(max(window, 1), floorMaxWindow)
	t := &floorTracker{
		bins:   bins,
		nb:     floorBuckets,
		window: window,
		every:  max(every, 1),
		ring:   make([]uint8, window*bins),
		hist:   make([]uint16, bins*floorBuckets),
		level:  make([]float32, floorBuckets),
		logp:   make([]float32, bins),
		noise:  make([]float32, bins),
		med:    make([]int, bins),
		below:  make([]int, bins),
	}
	for b := range floorBuckets {
		centreDB := floorBaseDB + floorBucketDB*float64(b) + floorBucketDB/2 // bucket centre, power dB
		t.level[b] = float32(math.Pow(10, centreDB/10) / math.Ln2)
	}
	t.reset()
	return t
}

// reset clears the window and returns the published floor to epsPower, so a fresh
// or Reset stream passes audio through unattenuated until the tracker warms up.
func (t *floorTracker) reset() {
	clear(t.ring)
	clear(t.hist)
	clear(t.med)
	clear(t.below)
	t.head, t.filled, t.since = 0, 0, 0
	for k := range t.noise {
		t.noise[k] = epsPower
	}
}

// push feeds one frame's power spectrum into the sliding window, keeps each bin's
// below-median count current, and on the publish cadence past the warm-up refreshes
// the median floor.
func (t *floorTracker) push(power []float32) {
	f32.Log10Floored(t.logp, power, epsPower) // one SIMD pass; floors at epsPower (no -Inf)
	full := t.filled == t.window
	base := t.head * t.bins
	for k := range t.bins {
		lp := t.logp[k]
		var b int
		// A NaN or +Inf power (log10 gives NaN / +Inf) is coerced to bucket 0 rather
		// than skipped. Skipping while still advancing head would leave this bin's
		// ring slot holding a stale bucket; window frames later the eviction would
		// decrement a count that was never incremented and underflow the uint16,
		// corrupting the bin's median. Coercing keeps sum(hist)==filled per bin.
		if lp != lp || math.IsInf(float64(lp), 1) {
			b = 0
		} else {
			b = min(max(int(float64(lp)*floorQuantScale+floorQuantOffset), 0), t.nb-1)
		}
		// Keep below[k] == sum(hist[k][0..med[k]-1]) as the window slides. med[k] is
		// frozen here (only publish moves it), so a bucket entering or leaving strictly
		// below the frozen median shifts the count by one; publish then rebalances the
		// median against the refreshed half.
		m := t.med[k]
		if full {
			oldB := int(t.ring[base+k])
			t.hist[k*t.nb+oldB]--
			if oldB < m {
				t.below[k]--
			}
		}
		t.ring[base+k] = uint8(b)
		t.hist[k*t.nb+b]++
		if b < m {
			t.below[k]++
		}
	}
	t.head++
	if t.head == t.window {
		t.head = 0
	}
	if !full {
		t.filled++
	}
	t.since++
	// Warm-up: hold the epsPower floor (pass-through) until the window is a quarter
	// full, then publish on the cadence.
	if t.since >= t.every && t.filled*4 >= t.window {
		t.publish()
		t.since = 0
	}
}

// publish rebalances each bin's incrementally-tracked median to the current window
// and writes the per-bin floor. The median bucket is the first bucket whose
// inclusive prefix count reaches half = ceil(filled/2). It resumes from the
// previous median and its maintained below-median count and steps by however far
// the median has shifted, so for a stable floor a refresh costs O(1) amortized per
// bin rather than a rescan from bucket 0. The published floor is identical to that
// rescan (TestBlindMedianMatchesRescan).
func (t *floorTracker) publish() {
	half := (t.filled + 1) / 2 // ceil(0.5 * filled)
	for k := range t.bins {
		// row is this bin's histogram (len == t.nb); the loop guards keep m within
		// [0, len(row)-1] so row[m] stays in range.
		row := t.hist[k*t.nb : k*t.nb+t.nb]
		m, below := t.med[k], t.below[k]
		// Move the median down while too many samples sit strictly below it, then up
		// while its bucket's inclusive prefix still falls short of half. The loops are
		// mutually exclusive and below stays == sum(row[0..m-1]) across each step.
		for m > 0 && below >= half {
			m--
			below -= int(row[m])
		}
		for m < len(row)-1 && below+int(row[m]) < half {
			below += int(row[m])
			m++
		}
		t.med[k], t.below[k] = m, below
		t.noise[k] = max(t.level[m], epsPower)
	}
}

// trackWindowFrames converts the blind-floor window from seconds to frames,
// matching the flagship tracker's conversion.
func trackWindowFrames(sec float32, sampleRate, hop int) int {
	return max(1, int(math.Round(float64(sec)*float64(sampleRate)/float64(hop))))
}
