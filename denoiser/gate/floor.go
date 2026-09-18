package gate

import (
	"math"

	"github.com/tphakala/simd/f32"
)

// blindEvery is how often (in frames) the blind tracker recomputes the published
// per-bin median floor. The median is stable frame to frame, so refreshing every
// few frames keeps the histogram walk off most frames' hot path.
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
// Gaussian noise is exponential, whose median is mean*ln2). All state is
// allocated up front; push and publish allocate nothing and are chunk-independent.
type floorTracker struct {
	bins, nb, window, every int
	ring                    []uint8   // window*bins: the quantized bucket per (ring slot, bin)
	hist                    []uint16  // bins*nb: per-bin bucket counts over the window
	level                   []float32 // nb: bucket-centre power * (1/ln2), precomputed
	logp                    []float32 // bins: log10(power) scratch for one push
	noise                   []float32 // bins: the published floor the Gate reads

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
	t.head, t.filled, t.since = 0, 0, 0
	for k := range t.noise {
		t.noise[k] = epsPower
	}
}

// push feeds one frame's power spectrum into the sliding window and, on the
// publish cadence past the warm-up, recomputes the median floor.
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
		if full {
			t.hist[k*t.nb+int(t.ring[base+k])]--
		}
		t.ring[base+k] = uint8(b)
		t.hist[k*t.nb+b]++
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

// publish recomputes the per-bin median floor from the histograms.
func (t *floorTracker) publish() {
	half := (t.filled + 1) / 2 // ceil(0.5 * filled)
	for k := range t.bins {
		row := t.hist[k*t.nb : k*t.nb+t.nb]
		cum, b := 0, 0
		// Bound on len(row) (== t.nb by construction), not the struct field, so
		// the compiler proves row[b] in-bounds and drops the per-bucket check.
		for ; b < len(row)-1; b++ {
			cum += int(row[b])
			if cum >= half {
				break
			}
		}
		t.noise[k] = max(t.level[b], epsPower)
	}
}

// trackWindowFrames converts the blind-floor window from seconds to frames,
// matching the flagship tracker's conversion.
func trackWindowFrames(sec float32, sampleRate, hop int) int {
	return max(1, int(math.Round(float64(sec)*float64(sampleRate)/float64(hop))))
}
