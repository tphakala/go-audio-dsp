//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. This zero-allocation assertion is
// therefore only meaningful without -race. Response is a visualization helper
// that allocates its result by design and is not covered here.

package equalizer

import "testing"

func TestProcessIntoZeroAlloc(t *testing.T) {
	e, err := New(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	in := eqSignal(4096)
	out := make([]float32, len(in))
	if _, err := e.ProcessInto(in, out); err != nil { // warm up
		t.Fatal(err)
	}
	if a := testing.AllocsPerRun(50, func() {
		if _, err := e.ProcessInto(in, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("ProcessInto allocated %v times, want 0", a)
	}
}
