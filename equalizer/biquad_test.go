package equalizer

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
)

// biquadsOf reads a Filter's section coefficients back as Biquads.
func biquadsOf(f *Filter) []Biquad {
	out := make([]Biquad, len(f.sections))
	for i, s := range f.sections {
		out[i] = Biquad{B0: s.c.b0, B1: s.c.b1, B2: s.c.b2, A1: s.c.a1, A2: s.c.a2}
	}
	return out
}

// TestNewBiquadFilterMatchesNewFilter pins that a Filter built from the
// coefficients of a NewFilter band runs exactly the same arithmetic on both
// paths, so NewBiquadFilter is the same section runtime and not a second copy.
func TestNewBiquadFilterMatchesNewFilter(t *testing.T) {
	x := eqSignal(4096)
	bands := []Band{
		{Type: HighPass, Frequency: 80, Q: 0.7071},
		{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6},
		{Type: LowShelf, Frequency: 200, Q: 0.7071, GainDB: -3},
		{Type: BandPass, Frequency: 2000, WidthHz: 500, Passes: 3},
	}
	for _, b := range bands {
		t.Run(b.Type.String(), func(t *testing.T) {
			t.Parallel()
			ref := mustFilter(t, b, 48000)
			got, err := NewBiquadFilter(biquadsOf(ref), 48000)
			if err != nil {
				t.Fatal(err)
			}
			if got.NumSections() != ref.NumSections() {
				t.Errorf("NumSections = %d, want %d", got.NumSections(), ref.NumSections())
			}

			want32 := make([]float32, len(x))
			got32 := make([]float32, len(x))
			if _, err := ref.ProcessInto(x, want32); err != nil {
				t.Fatal(err)
			}
			if _, err := got.ProcessInto(x, got32); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(want32, got32) {
				t.Error("ProcessInto output differs from the NewFilter output")
			}

			ref.Reset()
			got.Reset()
			want64 := make([]float64, len(x))
			for i, v := range x {
				want64[i] = float64(v)
			}
			got64 := slices.Clone(want64)
			ref.ApplyFloat64(want64)
			got.ApplyFloat64(got64)
			if !slices.Equal(want64, got64) {
				t.Error("ApplyFloat64 output differs from the NewFilter output")
			}
		})
	}
}

// TestNewBiquadFilterCopiesSections pins that the filter owns its coefficients:
// changing the caller's slice after construction does not change the output.
func TestNewBiquadFilterCopiesSections(t *testing.T) {
	x := eqSignal(1024)
	secs := biquadsOf(mustFilter(t, Band{Type: BandPass, Frequency: 1000, WidthHz: 200}, 48000))

	ref, err := NewBiquadFilter(secs, 48000)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(x))
	if _, err := ref.ProcessInto(x, want); err != nil {
		t.Fatal(err)
	}

	f, err := NewBiquadFilter(secs, 48000)
	if err != nil {
		t.Fatal(err)
	}
	secs[0] = Biquad{B0: 1} // a pass-through; would change the output if shared
	got := make([]float32, len(x))
	if _, err := f.ProcessInto(x, got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(want, got) {
		t.Fatal("mutating the caller's slice after construction changed the filter output")
	}
}

// TestNewBiquadFilterInvalid pins the construction checks: each rejection wraps
// ErrInvalidConfig, returns no filter, and names the section at fault.
func TestNewBiquadFilterInvalid(t *testing.T) {
	ok := Biquad{B0: 0.1, B1: 0, B2: -0.1, A1: -1.5, A2: 0.7}
	with := func(mod func(*Biquad)) []Biquad {
		b := ok
		mod(&b)
		return []Biquad{ok, b} // the bad one is section 1
	}
	tests := []struct {
		name     string
		sections []Biquad
		rate     int
		wantIdx  bool // the message names a section index
	}{
		{"zero rate", []Biquad{ok}, 0, false},
		{"negative rate", []Biquad{ok}, -48000, false},
		{"empty", nil, 48000, false},
		{"too many sections", slices.Repeat([]Biquad{ok}, MaxPasses+1), 48000, false},
		{"NaN B0", with(func(b *Biquad) { b.B0 = math.NaN() }), 48000, true},
		{"Inf A1", with(func(b *Biquad) { b.A1 = math.Inf(1) }), 48000, true},
		{"B1 beyond float32", with(func(b *Biquad) { b.B1 = 1e39 }), 48000, true},
		{"A2 = 1", with(func(b *Biquad) { b.A2 = 1 }), 48000, true},
		{"A2 = -1", with(func(b *Biquad) { b.A2 = -1 }), 48000, true},
		{"A2 = 1.2", with(func(b *Biquad) { b.A2 = 1.2 }), 48000, true},
		{"A1 = 1.95, A2 = 0.9", with(func(b *Biquad) { b.A1, b.A2 = 1.95, 0.9 }), 48000, true},
		{"A1 = 2, A2 = 0.99", with(func(b *Biquad) { b.A1, b.A2 = 2, 0.99 }), 48000, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, err := NewBiquadFilter(tc.sections, tc.rate)
			if !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("err = %v, want ErrInvalidConfig", err)
			}
			if f != nil {
				t.Error("filter is non-nil on error")
			}
			if tc.wantIdx && !strings.Contains(err.Error(), "section 1") {
				t.Errorf("error %q does not name section 1", err)
			}
		})
	}

	// A section just inside the stability triangle is accepted, so the tests
	// above pin the threshold and not merely "some error".
	if _, err := NewBiquadFilter([]Biquad{{B0: 0.01, B1: 0, B2: -0.01, A1: -1.9999, A2: 0.99995}}, 48000); err != nil {
		t.Errorf("near-boundary stable section rejected: %v", err)
	}
	// The count limit is inclusive.
	if _, err := NewBiquadFilter(slices.Repeat([]Biquad{ok}, MaxPasses), 48000); err != nil {
		t.Errorf("%d sections rejected: %v", MaxPasses, err)
	}
}

// TestFilterChainAcceptsBiquadFilter pins that a coefficient-built Filter
// carries its sample rate into the chain's one-rate check.
func TestFilterChainAcceptsBiquadFilter(t *testing.T) {
	secs := biquadsOf(mustFilter(t, Band{Type: BandPass, Frequency: 1000, WidthHz: 200}, 48000))
	a, err := NewBiquadFilter(secs, 48000)
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewBiquadFilter(secs, 44100)
	if err != nil {
		t.Fatal(err)
	}
	c := NewFilterChain()
	if err := c.AddFilter(a); err != nil {
		t.Fatalf("AddFilter at the matching rate: %v", err)
	}
	if err := c.AddFilter(b); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("AddFilter at another rate: err = %v, want ErrInvalidConfig", err)
	}
}
