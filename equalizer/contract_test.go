package equalizer

import (
	"errors"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

func TestContract(t *testing.T) {
	e, err := New(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 1, 100, 4096} {
		if got := e.MaxOutputLen(n); got != n {
			t.Errorf("MaxOutputLen(%d) = %d, want %d", n, got, n)
		}
	}
	if e.Latency() != 0 {
		t.Errorf("Latency() = %d, want 0", e.Latency())
	}
}

func TestErrInvalidConfigIsShared(t *testing.T) {
	_, err := New(Config{SampleRate: 0})
	if !errors.Is(err, dsp.ErrInvalidConfig) {
		t.Fatalf("err = %v, want dsp.ErrInvalidConfig", err)
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v, want equalizer.ErrInvalidConfig", err)
	}
}

func TestErrBufferTooSmallIsShared(t *testing.T) {
	e, err := New(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	_, err = e.ProcessInto(make([]float32, 100), make([]float32, 1))
	if !errors.Is(err, dsp.ErrBufferTooSmall) {
		t.Fatalf("err = %v, want dsp.ErrBufferTooSmall", err)
	}
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want equalizer.ErrBufferTooSmall", err)
	}
}
