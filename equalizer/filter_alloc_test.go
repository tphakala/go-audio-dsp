//go:build !race

// The race detector adds allocations to instrumented code, so these
// zero-allocation assertions are only meaningful without -race.

package equalizer

import (
	"slices"
	"testing"
)

// TestFilterProcessIntoZeroAlloc checks a Filter filters in place with no
// steady-state allocation.
func TestFilterProcessIntoZeroAlloc(t *testing.T) {
	f := mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6, Passes: 2}, 48000)
	in := eqSignal(4096)
	out := make([]float32, len(in))
	if _, err := f.ProcessInto(in, out); err != nil { // warm up
		t.Fatal(err)
	}
	if a := testing.AllocsPerRun(50, func() {
		if _, err := f.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("Filter.ProcessInto allocated %v times, want 0", a)
	}
}

// TestFilterChainProcessIntoZeroAlloc checks the chain's float32 path allocates
// nothing in steady state.
func TestFilterChainProcessIntoZeroAlloc(t *testing.T) {
	c := chainOf(t, testConfig())
	in := eqSignal(4096)
	out := make([]float32, len(in))
	if _, err := c.ProcessInto(in, out); err != nil { // warm up
		t.Fatal(err)
	}
	if a := testing.AllocsPerRun(50, func() {
		if _, err := c.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("FilterChain.ProcessInto allocated %v times, want 0", a)
	}
}

// TestFilterChainApplyFloat64ZeroAlloc checks the chain's float64 path allocates
// nothing in steady state.
func TestFilterChainApplyFloat64ZeroAlloc(t *testing.T) {
	c := chainOf(t, testConfig())
	buf := make([]float64, 4096)
	for i := range buf {
		buf[i] = 0.3 * float64((i%17)-8)
	}
	// Positive control: the warm-up must actually filter the buffer, so a future
	// no-op ApplyFloat64 cannot pass this zero-alloc test falsely.
	before := slices.Clone(buf)
	c.ApplyFloat64(buf) // warm up
	if slices.Equal(buf, before) {
		t.Fatal("ApplyFloat64 left the buffer unchanged; it did no real work")
	}
	if a := testing.AllocsPerRun(50, func() {
		c.ApplyFloat64(buf)
	}); a != 0 {
		t.Errorf("FilterChain.ApplyFloat64 allocated %v times, want 0", a)
	}
}
