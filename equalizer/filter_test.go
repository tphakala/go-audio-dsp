package equalizer

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// mustFilter builds a Filter or fails the test.
func mustFilter(t *testing.T, b Band, sampleRate int) *Filter {
	t.Helper()
	f, err := NewFilter(b, sampleRate)
	if err != nil {
		t.Fatalf("NewFilter(%v): %v", b.Type, err)
	}
	return f
}

// TestFilterMatchesEqualizerSingleBand pins that a Filter's float32 path is
// bit-identical to an Equalizer built from the same single band, so the new
// composable primitive cannot drift from the trusted flat cascade.
func TestFilterMatchesEqualizerSingleBand(t *testing.T) {
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
			e, err := New(Config{SampleRate: 48000, Bands: []Band{b}})
			if err != nil {
				t.Fatal(err)
			}
			eqOut := make([]float32, len(x))
			if _, err := e.ProcessInto(x, eqOut); err != nil {
				t.Fatal(err)
			}

			f := mustFilter(t, b, 48000)
			fOut := make([]float32, len(x))
			if _, err := f.ProcessInto(x, fOut); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(eqOut, fOut) {
				t.Fatal("Filter output differs from the single-band Equalizer output")
			}
			if f.NumSections() != e.NumSections() {
				t.Errorf("NumSections = %d, want %d", f.NumSections(), e.NumSections())
			}
		})
	}
}

// TestFilterApplyFloat64Composition pins pass-composition of the float64 path: a
// 2-pass filter's ApplyFloat64 equals two independent 1-pass filters of the same
// band applied in sequence, so Passes=N builds and runs N identical sections.
// This equality holds however each section rounds (both sides round at the same
// points), so it does not by itself prove float64 continuity between passes; that
// no-truncation property is proven by TestFilterApplyFloat64NoIntermediateTruncation.
func TestFilterApplyFloat64Composition(t *testing.T) {
	oneBand := Band{Type: HighPass, Frequency: 120, Q: 0.9}
	twoBand := oneBand
	twoBand.Passes = 2

	x := make([]float64, 2048)
	for i := range x {
		x[i] = math.Sin(0.05 * float64(i))
	}

	two := slices.Clone(x)
	mustFilter(t, twoBand, 48000).ApplyFloat64(two)

	onePass := slices.Clone(x)
	mustFilter(t, oneBand, 48000).ApplyFloat64(onePass)
	mustFilter(t, oneBand, 48000).ApplyFloat64(onePass)

	if !slices.Equal(two, onePass) {
		t.Fatal("2-pass ApplyFloat64 must equal two 1-pass ApplyFloat64 applied in sequence (pass composition)")
	}
}

// TestFilterApplyFloat64NoIntermediateTruncation pins that ApplyFloat64 keeps
// full float64 precision between passes: truncating the intermediate to float32
// by hand changes the result, which it could not do if ApplyFloat64 already
// truncated to float32 at each stage the way the float32 path does.
func TestFilterApplyFloat64NoIntermediateTruncation(t *testing.T) {
	one := Band{Type: Peaking, Frequency: 500, WidthHz: 80, GainDB: 12}
	two := one
	two.Passes = 2

	x := make([]float64, 2048)
	for i := range x {
		x[i] = 0.5 * math.Sin(0.06*float64(i))
	}

	// Full float64 through both passes (a 2-pass filter runs run64 twice with no
	// truncation between).
	full := slices.Clone(x)
	mustFilter(t, two, 48000).ApplyFloat64(full)

	// The same two passes, but with the intermediate truncated to float32 by hand.
	trunc := slices.Clone(x)
	mustFilter(t, one, 48000).ApplyFloat64(trunc)
	for i := range trunc {
		trunc[i] = float64(float32(trunc[i]))
	}
	mustFilter(t, one, 48000).ApplyFloat64(trunc)

	if slices.Equal(full, trunc) {
		t.Fatal("ApplyFloat64 must keep float64 between passes: truncating the intermediate by hand should change the result")
	}
}

// TestFilterBufferTooSmallConsumesNothing pins the Processor contract: an
// undersized out returns ErrBufferTooSmall and leaves the filter state untouched
// so the next call resumes as if the short call never happened.
func TestFilterBufferTooSmallConsumesNothing(t *testing.T) {
	b := Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6}
	x := eqSignal(2000)

	refOut := make([]float32, len(x))
	if _, err := mustFilter(t, b, 48000).ProcessInto(x, refOut); err != nil {
		t.Fatal(err)
	}

	f := mustFilter(t, b, 48000)
	if _, err := f.ProcessInto(x, make([]float32, len(x)-1)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	out := make([]float32, len(x))
	if _, err := f.ProcessInto(x, out); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(out, refOut) {
		t.Fatal("filter state changed after an undersized call")
	}
}

// TestFilterAliasedInPlace pins that out may be exactly in.
func TestFilterAliasedInPlace(t *testing.T) {
	b := Band{Type: LowShelf, Frequency: 200, Q: 0.7071, GainDB: -3}
	x := eqSignal(1000)

	refOut := make([]float32, len(x))
	if _, err := mustFilter(t, b, 48000).ProcessInto(x, refOut); err != nil {
		t.Fatal(err)
	}

	buf := slices.Clone(x)
	if _, err := mustFilter(t, b, 48000).ProcessInto(buf, buf); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(buf, refOut) {
		t.Fatal("in-place output differs from the separate-buffer output")
	}
}

// TestFilterContract pins the zero-latency Processor accessors and the
// empty-input path: MaxOutputLen is the identity, Latency is 0, and an empty
// input consumes and writes nothing.
func TestFilterContract(t *testing.T) {
	f := mustFilter(t, Band{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6}, 48000)
	for _, n := range []int{0, 1, 100, 4096} {
		if got := f.MaxOutputLen(n); got != n {
			t.Errorf("MaxOutputLen(%d) = %d, want %d", n, got, n)
		}
	}
	if f.Latency() != 0 {
		t.Errorf("Latency() = %d, want 0", f.Latency())
	}
	if n, err := f.ProcessInto(nil, nil); n != 0 || err != nil {
		t.Errorf("ProcessInto(nil, nil) = %d, %v; want 0, nil", n, err)
	}
}

// TestNewFilterInvalid pins that NewFilter rejects a bad sample rate and a bad
// band, wrapping ErrInvalidConfig.
func TestNewFilterInvalid(t *testing.T) {
	if _, err := NewFilter(Band{Type: LowPass, Frequency: 1000, Q: 1}, 0); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("zero sample rate: err = %v, want ErrInvalidConfig", err)
	}
	if _, err := NewFilter(Band{Type: 0, Frequency: 1000, Q: 1}, 48000); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("zero FilterType: err = %v, want ErrInvalidConfig", err)
	}
}
