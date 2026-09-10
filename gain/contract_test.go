package gain

import (
	"errors"
	"math"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/pcm"
)

func TestContract(t *testing.T) {
	g, err := New(3)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 1, 100, 4096} {
		if got := g.MaxOutputLen(n); got != n {
			t.Errorf("MaxOutputLen(%d) = %d, want %d", n, got, n)
		}
	}
	if g.Latency() != 0 {
		t.Errorf("Latency() = %d, want 0", g.Latency())
	}
	g.Reset() // must be callable and a no-op
	// A no-op Reset leaves processing unchanged.
	in := []float32{0.3, -0.7, 0.9}
	a := make([]float32, len(in))
	if _, err := g.ProcessInto(in, a); err != nil {
		t.Fatal(err)
	}
	g.Reset()
	b := make([]float32, len(in))
	if _, err := g.ProcessInto(in, b); err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("Reset changed processing at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestErrInvalidConfigIsShared(t *testing.T) {
	_, err := New(math.NaN())
	if !errors.Is(err, dsp.ErrInvalidConfig) {
		t.Fatalf("err = %v, want dsp.ErrInvalidConfig", err)
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v, want gain.ErrInvalidConfig", err)
	}
}

func TestErrBufferTooSmallIsShared(t *testing.T) {
	g, _ := New(3)
	_, err := g.ProcessInto(make([]float32, 4), make([]float32, 1))
	if !errors.Is(err, dsp.ErrBufferTooSmall) {
		t.Fatalf("err = %v, want dsp.ErrBufferTooSmall", err)
	}
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want gain.ErrBufferTooSmall", err)
	}
}

func TestErrOddByteLengthIsShared(t *testing.T) {
	g, _ := New(3)
	err := g.ApplyBytes(make([]byte, 3))
	if !errors.Is(err, pcm.ErrOddByteLength) {
		t.Fatalf("err = %v, want pcm.ErrOddByteLength", err)
	}
	if !errors.Is(err, ErrOddByteLength) {
		t.Fatalf("err = %v, want gain.ErrOddByteLength", err)
	}
}
