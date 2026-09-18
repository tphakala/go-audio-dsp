package stft

import (
	"math/rand/v2"
	"testing"
)

// TestMeanPowerIntoMatchesManualAverage pins MeanPowerInto to the mean of the
// per-frame PowerInto rows. It uses a signal spanning several internal batches
// (the batch is 64 frames) so the batch-boundary slicing is exercised. Within
// one build the transform is deterministic, so the batched average is bit
// identical to averaging one whole-clip PowerInto pass; deleting the divide by
// frames (sums instead of means) or the accumulation turns it red.
func TestMeanPowerIntoMatchesManualAverage(t *testing.T) {
	p, err := New(Config{FrameSize: 256, HopSize: 64, Window: Hann})
	if err != nil {
		t.Fatal(err)
	}
	bins := p.NumBins()
	rng := rand.New(rand.NewPCG(1, 2))
	x := make([]float32, 256+64*200) // ~201 frames, > 3 batches of 64
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	frames := p.NumFrames(len(x), NoPad)
	got := make([]float32, bins)
	if n := p.MeanPowerInto(got, x); n != frames {
		t.Fatalf("MeanPowerInto returned %d frames, want %d", n, frames)
	}

	flat := make([]float32, frames*bins)
	if n := p.PowerInto(flat, x, NoPad); n != frames {
		t.Fatalf("PowerInto wrote %d frames, want %d", n, frames)
	}
	ref := make([]float32, bins)
	for k := range bins {
		var acc float64
		for f := range frames {
			acc += float64(flat[f*bins+k])
		}
		ref[k] = float32(acc / float64(frames))
	}
	for k := range bins {
		if got[k] != ref[k] {
			t.Errorf("bin %d: MeanPowerInto %g, manual average %g", k, got[k], ref[k])
		}
	}
}

// TestMeanPowerIntoShortSignalReturnsZero pins the sub-frame guard: fewer than
// one full frame averages nothing, returns 0, and leaves dst untouched (the
// caller's buffer, not a NaN from dividing by zero frames). Deleting the
// frames == 0 early return divides by zero and clobbers dst.
func TestMeanPowerIntoShortSignalReturnsZero(t *testing.T) {
	p, err := New(Config{FrameSize: 256, HopSize: 64})
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]float32, p.NumBins())
	for i := range dst {
		dst[i] = 42
	}
	if n := p.MeanPowerInto(dst, make([]float32, 100)); n != 0 {
		t.Fatalf("MeanPowerInto returned %d frames, want 0 for sub-frame input", n)
	}
	for i, v := range dst {
		if v != 42 {
			t.Fatalf("dst[%d] = %g, MeanPowerInto must leave dst unchanged when it averages 0 frames", i, v)
		}
	}
}
