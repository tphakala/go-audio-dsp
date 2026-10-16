//go:build !race

package dspshared

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// nopHook is a non-allocating method-value hook, bound like the production
// blocks bind theirs, so the test measures the Stream and not the hook.
type nopHook struct{}

func (nopHook) frame(_ int64, spec []complex64, _ []float32, _ bool) []complex64 { return spec }
func (nopHook) resetState()                                                      {}

// TestStreamZeroAllocSteadyState pins that ProcessInto, and ProcessInto plus
// FlushInto, allocate nothing per call once warmed. Any per-frame allocation in
// the driver, such as a Feed callback that escapes, turns it red. (Excluded
// under -race, which adds its own bookkeeping allocations.)
func TestStreamZeroAllocSteadyState(t *testing.T) {
	an, err := stft.NewAnalyzer(STFTConfig(testFrame, testHop))
	if err != nil {
		t.Fatal(err)
	}
	var h nopHook
	s := new(Stream)
	if err := s.Init(an, 0, h.frame, h.resetState); err != nil {
		t.Fatal(err)
	}
	s.Reset()

	in := testSignal(4*testHop, 4)
	out := make([]float32, s.MaxOutputLen(len(in)))
	for range 50 {
		if _, err := s.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
	}
	if a := testing.AllocsPerRun(200, func() {
		if _, err := s.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("ProcessInto allocated %.1f objects/op, want 0", a)
	}

	fb := make([]float32, s.Latency()+testHop)
	if a := testing.AllocsPerRun(200, func() {
		if _, err := s.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
		if _, err := s.FlushInto(fb); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("ProcessInto+FlushInto allocated %.1f objects/op, want 0", a)
	}
}
