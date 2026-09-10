package pcm

import (
	"slices"
	"testing"
)

// TestScaleInt16ScalesRoundsSaturates pins the three behaviors ScaleInt16
// promises: a plain multiply, round-to-even on half-integer results, and
// saturation past full scale instead of wraparound.
func TestScaleInt16ScalesRoundsSaturates(t *testing.T) {
	cases := []struct {
		name   string
		in     []int16
		factor float32
		want   []int16
	}{
		{"scales", []int16{100, -100, 10000, -10000}, 2.0, []int16{200, -200, 20000, -20000}},
		{"saturates", []int16{20000, -20000, 32767, -32768}, 4.0, []int16{32767, -32768, 32767, -32768}},
		{"round half to even", []int16{1, 3, 5, 7}, 2.5, []int16{2, 8, 12, 18}}, // 2.5, 7.5, 12.5, 17.5
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := slices.Clone(c.in)
			ScaleInt16(got, c.factor)
			if !slices.Equal(got, c.want) {
				t.Fatalf("ScaleInt16(%v, %v) = %v, want %v", c.in, c.factor, got, c.want)
			}
		})
	}
}

// TestScaleInt16ChunkBoundary scales an input longer than two scratch chunks so
// a bug that processes only the first chunk leaves the tail untouched and fails.
func TestScaleInt16ChunkBoundary(t *testing.T) {
	const n = 2*scaleChunk + 3
	in := make([]int16, n)
	for i := range in {
		in[i] = int16(i%50 - 25) // small, so factor 2 never saturates
	}
	want := make([]int16, n)
	for i := range in {
		want[i] = in[i] * 2
	}
	got := slices.Clone(in)
	ScaleInt16(got, 2.0)
	if !slices.Equal(got, want) {
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("first mismatch at %d (of %d): got %d, want %d", i, n, got[i], want[i])
			}
		}
	}
}

// TestScaleInt16UnityAndEmpty checks the no-op cases: factor 1 leaves every
// sample byte-for-byte (including full scale, which a needless round-trip would
// still preserve but which documents the contract), and an empty slice is safe.
func TestScaleInt16UnityAndEmpty(t *testing.T) {
	in := []int16{1, -1, 12345, -12345, 32767, -32768}
	got := slices.Clone(in)
	ScaleInt16(got, 1.0)
	if !slices.Equal(got, in) {
		t.Fatalf("unity ScaleInt16 changed samples: %v -> %v", in, got)
	}
	ScaleInt16(nil, 2.0)      // must not panic
	ScaleInt16([]int16{}, 2.0) // must not panic
}
