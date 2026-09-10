//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package gain

import "testing"

func TestZeroAlloc(t *testing.T) {
	g, err := New(6)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]float32, 4096)
	out := make([]float32, 4096)
	i16 := make([]int16, 4096)
	for i := range in {
		in[i] = float32(i%200-100) / 100
	}
	if _, err := g.ProcessInto(in, out); err != nil { // warm up
		t.Fatal(err)
	}
	// ApplyBytes is zero-alloc only on little-endian hosts (pcm.InPlaceInt16 is
	// zero-copy there); its allocation behavior is covered by the pcm package's
	// alloc test, which guards on host endianness.
	checks := []struct {
		name string
		fn   func()
	}{
		{"ProcessInto", func() { _, _ = g.ProcessInto(in, out) }},
		{"ApplyInt16", func() { g.ApplyInt16(i16) }},
	}
	for _, c := range checks {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}
