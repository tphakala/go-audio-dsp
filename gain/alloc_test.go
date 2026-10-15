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
	b := make([]byte, 8192)
	scratch := make([]byte, 8192)
	checks := []struct {
		name string
		fn   func()
	}{
		{"ProcessInto", func() {
			if _, err := g.ProcessInto(in, out); err != nil {
				t.Fatal(err)
			}
		}},
		{"ApplyInt16", func() { g.ApplyInt16(i16) }},
		{"ApplyBytes", func() {
			if err := g.ApplyBytes(b); err != nil {
				t.Fatal(err)
			}
		}},
		{"ApplyBytesTo", func() {
			if err := g.ApplyBytesTo(scratch, b); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, c := range checks {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}

func TestAppliedBytesAllocs(t *testing.T) {
	src := make([]byte, 4096)
	unity, _ := New(0)
	boost, _ := New(6)
	if a := testing.AllocsPerRun(50, func() { _, _ = unity.AppliedBytes(src) }); a != 0 {
		t.Errorf("AppliedBytes at 0 dB allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(50, func() { _, _ = boost.AppliedBytes(src) }); a != 1 {
		t.Errorf("AppliedBytes at +6 dB allocated %v times, want exactly 1", a)
	}
}
