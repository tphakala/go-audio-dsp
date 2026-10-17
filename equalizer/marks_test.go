package equalizer

import (
	"math"
	"slices"
	"testing"
)

func TestFilterMarks(t *testing.T) {
	const fs = 48000
	for ft := LowPass; ft <= Peaking; ft++ {
		for _, passes := range []int{0, 16} {
			b := Band{Type: ft, Frequency: 1234.5, Q: 1, WidthHz: 100, GainDB: 3, Passes: passes}
			f := mustFilter(t, b, fs)
			if got, want := f.Marks(), []float64{1234.5}; !slices.Equal(got, want) {
				t.Errorf("%v, %d passes: Marks = %v, want %v", ft, passes, got, want)
			}
		}
	}

	bw, err := NewButterworthBandPass(990, 1010, 4, fs)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := bw.Marks(), []float64{math.Sqrt(990 * 1010)}; !slices.Equal(got, want) {
		t.Errorf("Butterworth Marks = %v, want %v", got, want)
	}

	bq, err := NewBiquadFilter([]Biquad{{B0: 1}}, fs)
	if err != nil {
		t.Fatal(err)
	}
	if got := bq.Marks(); got != nil {
		t.Errorf("biquad Marks = %v, want nil", got)
	}

	// The returned slice is a copy: mutating it leaves the filter unchanged.
	f := mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 100, GainDB: 3}, fs)
	f.Marks()[0] = 5
	if got := f.Marks(); !slices.Equal(got, []float64{1000}) {
		t.Errorf("Marks after mutating a returned slice = %v, want [1000]", got)
	}
}

func TestFilterChainMarksTrackAcceptedFilters(t *testing.T) {
	const fs = 48000
	c := NewFilterChain()
	bq, err := NewBiquadFilter([]Biquad{{B0: 1}}, fs)
	if err != nil {
		t.Fatal(err)
	}
	steps := []struct {
		name string
		do   func() error
		fail bool // AddFilter must reject the filter
		want []float64
	}{
		{"empty chain", func() error { return nil }, false, nil},
		{"add 1 kHz peaking", func() error {
			return c.AddFilter(mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 100, GainDB: 3}, fs))
		}, false, []float64{1000}},
		{"rejects nil", func() error { return c.AddFilter(nil) }, true, []float64{1000}},
		{"rejects rate mismatch", func() error {
			return c.AddFilter(mustFilter(t, Band{Type: Peaking, Frequency: 5000, WidthHz: 100, GainDB: 3}, 44100))
		}, true, []float64{1000}},
		{"add 1 kHz band-reject", func() error {
			return c.AddFilter(mustFilter(t, Band{Type: BandReject, Frequency: 1000, WidthHz: 100}, fs))
		}, false, []float64{1000}},
		{"add biquad filter", func() error { return c.AddFilter(bq) }, false, []float64{1000}},
		{"add 300 Hz high-pass", func() error {
			return c.AddFilter(mustFilter(t, Band{Type: HighPass, Frequency: 300, Q: 0.7071}, fs))
		}, false, []float64{300, 1000}},
	}
	for _, s := range steps {
		err := s.do()
		if s.fail != (err != nil) {
			t.Fatalf("%s: err = %v", s.name, err)
		}
		if got := c.Marks(); !slices.Equal(got, s.want) {
			t.Errorf("%s: Marks = %v, want %v", s.name, got, s.want)
		}
	}
}

func TestFilterChainMarksIgnoreOrder(t *testing.T) {
	bands := []Band{
		{Type: Peaking, Frequency: 2000, WidthHz: 100, GainDB: 3},
		{Type: HighPass, Frequency: 300, Q: 0.7071},
		{Type: BandReject, Frequency: 2000, WidthHz: 50},
	}
	rev := slices.Clone(bands)
	slices.Reverse(rev)
	a := chainOf(t, Config{SampleRate: 48000, Bands: bands}).Marks()
	b := chainOf(t, Config{SampleRate: 48000, Bands: rev}).Marks()
	if !slices.Equal(a, b) || !slices.Equal(a, []float64{300, 2000}) {
		t.Errorf("Marks = %v and %v, want [300 2000] for both orders", a, b)
	}
}

func TestEqualizerMarksMatchChain(t *testing.T) {
	cfg := testConfig()
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := chainOf(t, cfg).Marks()
	if got := e.Marks(); !slices.Equal(got, want) || len(got) == 0 {
		t.Errorf("Equalizer Marks = %v, want the chain's %v", got, want)
	}
	empty, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	if got := empty.Marks(); got != nil {
		t.Errorf("empty Equalizer Marks = %v, want nil", got)
	}
	if got := NewFilterChain().Marks(); got != nil {
		t.Errorf("empty chain Marks = %v, want nil", got)
	}
}

// TestEqualizerMarksDedupeAndCopy pins that two bands of different types at one
// frequency give one mark, and that the returned slice is a copy.
func TestEqualizerMarksDedupeAndCopy(t *testing.T) {
	e, err := New(Config{SampleRate: 48000, Bands: []Band{
		{Type: Peaking, Frequency: 1000, WidthHz: 100, GainDB: 3},
		{Type: BandReject, Frequency: 1000, WidthHz: 50},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := e.Marks(); !slices.Equal(got, []float64{1000}) {
		t.Fatalf("Marks = %v, want [1000]", got)
	}
	e.Marks()[0] = 5
	if got := e.Marks(); !slices.Equal(got, []float64{1000}) {
		t.Errorf("Marks after mutating a returned slice = %v, want [1000]", got)
	}
}
