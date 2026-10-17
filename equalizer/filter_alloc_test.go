//go:build !race

// The race detector adds allocations to instrumented code, so these
// zero-allocation assertions are only meaningful without -race.

package equalizer

import (
	"slices"
	"testing"
)

// TestFilterProcessIntoZeroAlloc checks a Filter filters in place with no
// steady-state allocation, whichever constructor built it.
func TestFilterProcessIntoZeroAlloc(t *testing.T) {
	cases := append(filterCases(), filterCase{"NewFilter 2 passes", func(t *testing.T) *Filter {
		t.Helper()
		return mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6, Passes: 2}, 48000)
	}})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.build(t)
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
		})
	}
}

// TestFilterApplyFloat64ZeroAlloc checks a single Filter's float64 path
// allocates nothing in steady state, whichever constructor built it.
func TestFilterApplyFloat64ZeroAlloc(t *testing.T) {
	for _, tc := range filterCases() {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.build(t)
			buf := make([]float64, 4096)
			for i := range buf {
				buf[i] = 0.3 * float64((i%17)-8)
			}
			// Positive control: the warm-up must actually filter the buffer, so a
			// no-op ApplyFloat64 cannot pass this zero-alloc test falsely.
			before := slices.Clone(buf)
			f.ApplyFloat64(buf) // warm up
			if slices.Equal(buf, before) {
				t.Fatal("ApplyFloat64 left the buffer unchanged; it did no real work")
			}
			if a := testing.AllocsPerRun(50, func() {
				f.ApplyFloat64(buf)
			}); a != 0 {
				t.Errorf("Filter.ApplyFloat64 allocated %v times, want 0", a)
			}
		})
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

// TestFilterChainResponseIntoZeroAlloc checks ResponseInto allocates nothing,
// for a populated and an empty chain.
func TestFilterChainResponseIntoZeroAlloc(t *testing.T) {
	freqs, err := LogSweep(20, 20000, 512)
	if err != nil {
		t.Fatal(err)
	}
	// The null filter (every coefficient zero) drives the floored path under AllocsPerRun.
	null, err := NewBiquadFilter([]Biquad{{}}, testConfig().SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	floored := chainOf(t, testConfig())
	if err := floored.AddFilter(null); err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]*FilterChain{"chain": chainOf(t, testConfig()), "floored": floored, "empty": NewFilterChain()} {
		t.Run(name, func(t *testing.T) {
			dst := make([]ResponsePoint, len(freqs))
			// Positive control: the warm-up must write real points.
			if _, err := c.ResponseInto(dst, freqs); err != nil {
				t.Fatal(err)
			}
			if dst[0].Hz != freqs[0] {
				t.Fatal("ResponseInto left dst unwritten; it did no real work")
			}
			if a := testing.AllocsPerRun(50, func() {
				if _, err := c.ResponseInto(dst, freqs); err != nil {
					t.Fatal(err)
				}
			}); a != 0 {
				t.Errorf("ResponseInto allocated %v times, want 0", a)
			}
		})
	}
}
