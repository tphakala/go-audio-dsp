package gate

import (
	"errors"
	"math"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestLearnedFloorImmutableDuringStreaming pins that a learned floor never adapts:
// streaming loud audio does not change what NoiseFloor() reports, and the floor
// survives Flush's Reset. (On the learned path the blind tracker is never pushed,
// so this pins floor immutability, not the tracker itself.) Any code path that
// mutates the learned floor during streaming turns it red.
func TestLearnedFloorImmutableDuringStreaming(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	noise := whiteNoise(frame*8, dbToLin(-40), 4)
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.LearnNoise(noise); err != nil {
		t.Fatal(err)
	}
	before := g.NoiseFloor()
	if before == nil {
		t.Fatal("NoiseFloor nil after LearnNoise")
	}
	x := whiteNoise(sr, dbToLin(-40), 5)
	tn := tone(len(x), sr, 3000, dbToLin(-6))
	for i := range x {
		x[i] += tn[i]
	}
	_ = streamAll(t, g, x)
	after := g.NoiseFloor()
	if after == nil {
		t.Fatal("NoiseFloor nil after streaming (learned floor did not survive Reset)")
	}
	for k := range before {
		if before[k] != after[k] {
			t.Fatalf("learned floor changed at bin %d after streaming: %g -> %g", k, before[k], after[k])
		}
	}
}

// TestSilentLearnedExcerptIsFinite pins the epsPower floor on the learned floor:
// learning from digital silence and then processing silent frames must not divide
// zero power by a zero floor (NaN). Removing the floor in LearnNoise/copyFloor
// yields non-finite output on the silent gap.
func TestSilentLearnedExcerptIsFinite(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.LearnNoise(make([]float32, frame*4)); err != nil {
		t.Fatal(err)
	}
	x := whiteNoise(10000, 0.1, 50)
	for i := 3000; i < 5000; i++ { // a silent gap: 0 power / floor
		x[i] = 0
	}
	out := streamAll(t, g, x)
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output sample %d non-finite (%v); the epsPower floor on a silent learned excerpt is missing", i, v)
		}
	}
}

// TestNoiseFloorNilWhenBlind pins the learned/blind state reported by NoiseFloor,
// including SetNoiseFloor(nil) returning to blind tracking.
func TestNoiseFloorNilWhenBlind(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if g.NoiseFloor() != nil {
		t.Error("NoiseFloor should be nil on a blind gate")
	}
	if err := g.LearnNoise(whiteNoise(frame*4, 0.1, 1)); err != nil {
		t.Fatal(err)
	}
	if g.NoiseFloor() == nil {
		t.Error("NoiseFloor should be non-nil after LearnNoise")
	}
	if err := g.SetNoiseFloor(nil); err != nil {
		t.Fatal(err)
	}
	if g.NoiseFloor() != nil {
		t.Error("SetNoiseFloor(nil) should return the gate to blind tracking (nil floor)")
	}
}

// TestNoiseAPIErrors pins the length and size guards on the noise API.
func TestNoiseAPIErrors(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetNoiseFloor(make([]float32, g.FrameSize()/2)); !errors.Is(err, ErrFloorMismatch) {
		t.Errorf("SetNoiseFloor wrong length: err %v, want ErrFloorMismatch", err)
	}
	if err := g.LearnNoise(make([]float32, frame-1)); !errors.Is(err, ErrNoiseTooShort) {
		t.Errorf("LearnNoise short excerpt: err %v, want ErrNoiseTooShort", err)
	}
}
