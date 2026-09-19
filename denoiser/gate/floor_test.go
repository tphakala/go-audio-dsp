package gate

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestBlindFloorOnContinuousNoise pins the blind rolling-median floor to the
// learned mean-power floor of the same gapless noise. The 1/ln2 median-to-mean
// correction makes the two agree in level; dropping it biases the blind floor by
// about 1.6 dB, and a broken histogram pop scatters it. The check is on the mean
// bias and mean absolute deviation across bins, which is robust to per-bin
// quantization while still catching a systematic error.
func TestBlindFloorOnContinuousNoise(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	x := whiteNoise(3*sr, dbToLin(-30), 21)
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Process(x); err != nil { // drive the blind tracker; read before any Flush/Reset
		t.Fatal(err)
	}
	blind := slices.Clone(g.tracker.noise)

	ref, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.LearnNoise(x); err != nil {
		t.Fatal(err)
	}
	learned := ref.NoiseFloor()

	var sumSigned, sumAbs float64
	var cnt int
	for k := 2; k < len(blind)-1; k++ { // skip DC and Nyquist edges
		if blind[k] <= epsPower || learned[k] <= epsPower {
			continue
		}
		d := 10*math.Log10(float64(blind[k])) - 10*math.Log10(float64(learned[k]))
		sumSigned += d
		sumAbs += math.Abs(d)
		cnt++
	}
	if cnt == 0 {
		t.Fatal("no comparable bins")
	}
	meanSigned, meanAbs := sumSigned/float64(cnt), sumAbs/float64(cnt)
	t.Logf("blind vs learned floor over %d bins: mean bias %.2f dB, mean abs dev %.2f dB", cnt, meanSigned, meanAbs)
	if math.Abs(meanSigned) > 1.0 {
		t.Errorf("blind floor mean bias %.2f dB (>1 dB); the 1/ln2 median-to-mean correction is likely wrong", meanSigned)
	}
	if meanAbs > 1.5 {
		t.Errorf("blind floor mean abs deviation %.2f dB (>1.5 dB)", meanAbs)
	}
}

// TestBlindFloorIgnoresSparseSignal pins that the blind floor is a percentile, not
// a mean: a loud tone present only 20% of the time sits above the median, so the
// floor at the tone bin still tracks the noise. Replacing the median with a mean
// would pull the tone-bin floor up by many dB.
func TestBlindFloorIgnoresSparseSignal(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	dur := 3 * sr
	noise := whiteNoise(dur, dbToLin(-30), 22)
	tn := tone(dur, sr, 3000, dbToLin(-6))
	x := make([]float32, dur)
	period := sr / 2    // 0.5 s
	onFor := period / 5 // 20% duty cycle
	for i := range x {
		x[i] = noise[i]
		if i%period < onFor {
			x[i] += tn[i]
		}
	}
	toneBin := freqToBin(3000, sr, frame)

	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Process(x); err != nil {
		t.Fatal(err)
	}
	blind := slices.Clone(g.tracker.noise)

	ref, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.LearnNoise(noise); err != nil {
		t.Fatal(err)
	}
	noiseFloor := ref.NoiseFloor()

	bdb := 10 * math.Log10(float64(blind[toneBin]))
	ndb := 10 * math.Log10(float64(noiseFloor[toneBin]))
	t.Logf("tone bin %d: blind floor %.1f dB, noise-only floor %.1f dB", toneBin, bdb, ndb)
	if math.Abs(bdb-ndb) > 2.0 {
		t.Errorf("blind floor at the tone bin is %.1f dB, %.1f dB off the noise-only floor %.1f dB; a 20%%-duty tone should not move a median floor this far (mean instead of percentile?)", bdb, bdb-ndb, ndb)
	}
}

// TestNaNChunkKeepsHistogramConsistent pins the blind tracker's non-finite
// handling: a NaN power coerces to bucket 0 rather than being skipped, so every
// bin's histogram counts always sum to the number of frames in the window. If a
// non-finite push were skipped, the later eviction of its stale ring slot would
// decrement a count that was never incremented and underflow the uint16, breaking
// the invariant. A short window makes eviction run well within the stream.
func TestNaNChunkKeepsHistogramConsistent(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	p := ParamsFor(denoiser.Heavy) // L = 2
	p.FloorWindowSec = 0.1         // ~19 frames, far shorter than the stream
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
	if err != nil {
		t.Fatal(err)
	}
	x := whiteNoise(40000, 0.1, 30) // ~600 frames, so eviction runs many times
	for i := 2000; i < 2200; i++ {
		x[i] = float32(math.NaN())
	}
	out, err := g.Process(x) // do not Flush: Reset would clear the histogram
	if err != nil {
		t.Fatal(err)
	}
	tr := g.tracker
	for k := range tr.bins {
		sum := 0
		for b := range tr.nb {
			sum += int(tr.hist[k*tr.nb+b])
		}
		if sum != tr.filled {
			t.Fatalf("bin %d: histogram sum %d != filled %d (a non-finite push corrupted the sliding window)", k, sum, tr.filled)
		}
	}
	tail := out[len(out)-2000:]
	for i, v := range tail {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("tail sample %d non-finite (%v); the blind path did not recover after the NaN chunk", i, v)
		}
	}
}

// TestBlindMedianMatchesRescan pins the incrementally-tracked per-bin median to a
// from-scratch histogram rescan. The whole point of the incremental tracker is to
// publish the identical floor as recomputing each median from bucket 0, so on every
// publish frame the tracked median bucket must equal the rescan result and below[k]
// must equal sum(hist[k][0..med-1]). It drives varied power (including non-finite,
// coerced to bucket 0) through warm-up, steady state, and window eviction. Break the
// rebalance in publish (for example flip the up-loop's `< half` to `<= half`, or
// drop the down-loop) and this test turns red.
func TestBlindMedianMatchesRescan(t *testing.T) {
	const bins, window = 40, 50 // short window so eviction runs well within the drive
	tr := newFloorTracker(bins, window, blindEvery)
	rng := rand.New(rand.NewSource(7))
	power := make([]float32, bins)
	publishes := 0
	for frame := range 500 {
		for k := range power {
			db := -120 + rng.Float64()*180 // spread across the bucket range
			power[k] = float32(math.Pow(10, db/10))
		}
		if frame%37 == 0 { // exercise the bucket-0 coercion path
			power[frame%bins] = float32(math.Inf(1))
		}
		tr.push(power)
		if tr.since != 0 { // since resets to 0 only on a publish frame
			continue
		}
		publishes++
		half := (tr.filled + 1) / 2
		for k := range tr.bins {
			// Replicate the original from-scratch scan: first bucket whose inclusive
			// prefix reaches half, capped at nb-1.
			cum := 0
			want := 0
			for ; want < tr.nb-1; want++ {
				cum += int(tr.hist[k*tr.nb+want])
				if cum >= half {
					break
				}
			}
			if tr.med[k] != want {
				t.Fatalf("frame %d bin %d: tracked median bucket %d != rescan %d", frame, k, tr.med[k], want)
			}
			sumBelow := 0
			for b := range tr.med[k] {
				sumBelow += int(tr.hist[k*tr.nb+b])
			}
			if tr.below[k] != sumBelow {
				t.Fatalf("frame %d bin %d: below %d != sum(hist[0..med-1]) %d", frame, k, tr.below[k], sumBelow)
			}
			if got := max(tr.level[want], epsPower); tr.noise[k] != got {
				t.Fatalf("frame %d bin %d: published floor %g != level[%d] %g", frame, k, tr.noise[k], want, got)
			}
		}
	}
	if publishes < 50 {
		t.Fatalf("only %d publishes observed; the drive did not exercise steady-state refreshes", publishes)
	}
}
