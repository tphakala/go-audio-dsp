package equalizer

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// chainOf builds a FilterChain with one Filter per band in cfg, or fails.
func chainOf(t *testing.T, cfg Config) *FilterChain {
	t.Helper()
	c := NewFilterChain()
	for _, b := range cfg.Bands {
		if err := c.AddFilter(mustFilter(t, b, cfg.SampleRate)); err != nil {
			t.Fatal(err)
		}
	}
	return c
}

// TestFilterChainMatchesEqualizer pins that a FilterChain of one Filter per band
// is bit-identical to an Equalizer of the same bands: the composable path cannot
// drift from the trusted flat cascade.
func TestFilterChainMatchesEqualizer(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(8192)

	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eqOut := make([]float32, len(x))
	if _, err := e.ProcessInto(x, eqOut); err != nil {
		t.Fatal(err)
	}

	c := chainOf(t, cfg)
	chainOut := make([]float32, len(x))
	if _, err := c.ProcessInto(x, chainOut); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(eqOut, chainOut) {
		t.Fatal("FilterChain output differs from the Equalizer output for the same bands")
	}
	if c.Len() != len(cfg.Bands) {
		t.Errorf("Len = %d, want %d", c.Len(), len(cfg.Bands))
	}
	if c.NumSections() != e.NumSections() {
		t.Errorf("NumSections = %d, want %d", c.NumSections(), e.NumSections())
	}

	// A multi-pass band (Passes > 1) must match too: a Filter of N passes is N
	// biquad sections, exactly the Equalizer's N sections for that band. The local
	// config keeps the shared testConfig untouched.
	mp := Config{SampleRate: 48000, Bands: []Band{
		{Type: HighPass, Frequency: 120, Q: 0.9, Passes: 2},
		{Type: Peaking, Frequency: 1500, WidthHz: 300, GainDB: 4},
	}}
	eMP, err := New(mp)
	if err != nil {
		t.Fatal(err)
	}
	mpEqOut := make([]float32, len(x))
	if _, err := eMP.ProcessInto(x, mpEqOut); err != nil {
		t.Fatal(err)
	}
	mpChainOut := make([]float32, len(x))
	if _, err := chainOf(t, mp).ProcessInto(x, mpChainOut); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mpEqOut, mpChainOut) {
		t.Fatal("multi-pass FilterChain output differs from the Equalizer output for the same bands")
	}
}

// TestFilterChainEmptyIsIdentity pins that an empty chain copies input unchanged.
func TestFilterChainEmptyIsIdentity(t *testing.T) {
	x := eqSignal(256)
	out := make([]float32, len(x))
	n, err := NewFilterChain().ProcessInto(x, out)
	if err != nil || n != len(x) {
		t.Fatalf("ProcessInto empty chain: n=%d err=%v", n, err)
	}
	if !slices.Equal(out, x) {
		t.Fatal("an empty chain must be an identity block")
	}
}

// TestFilterChainAddNilRejected pins that AddFilter rejects a nil filter with
// ErrInvalidConfig rather than deferring to a nil dereference during processing.
func TestFilterChainAddNilRejected(t *testing.T) {
	c := NewFilterChain()
	if err := c.AddFilter(nil); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("AddFilter(nil) err = %v, want ErrInvalidConfig", err)
	}
	if c.Len() != 0 {
		t.Errorf("Len = %d after a rejected add, want 0", c.Len())
	}
}

// TestFilterChainChunkInvariance pins that chunked processing equals one-shot
// processing, so the chain carries every filter's state correctly across calls.
func TestFilterChainChunkInvariance(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(10000)

	refOut := make([]float32, len(x))
	if _, err := chainOf(t, cfg).ProcessInto(x, refOut); err != nil {
		t.Fatal(err)
	}

	for _, chunk := range []int{1, 7, 333, 1000} {
		c := chainOf(t, cfg)
		out := make([]float32, len(x))
		for i := 0; i < len(x); i += chunk {
			end := min(i+chunk, len(x))
			if _, err := c.ProcessInto(x[i:end], out[i:end]); err != nil {
				t.Fatal(err)
			}
		}
		if !slices.Equal(out, refOut) {
			t.Fatalf("chunk %d: chunked output differs from one-shot", chunk)
		}
	}
}

// TestFilterChainApplyFloat64AcrossFilters pins that the chain's float64 path
// applies each filter's ApplyFloat64 in order over one shared buffer (the chain
// composes its filters). It does not by itself prove float64 continuity between
// filters, since both sides route through Filter.ApplyFloat64; the no-truncation
// property is proven at the section level by
// TestFilterApplyFloat64NoIntermediateTruncation.
func TestFilterChainApplyFloat64AcrossFilters(t *testing.T) {
	cfg := testConfig()
	x := make([]float64, 2048)
	for i := range x {
		x[i] = 0.4 * math.Sin(0.03*float64(i))
	}

	viaChain := slices.Clone(x)
	chainOf(t, cfg).ApplyFloat64(viaChain)

	viaFilters := slices.Clone(x)
	for _, b := range cfg.Bands {
		mustFilter(t, b, cfg.SampleRate).ApplyFloat64(viaFilters)
	}

	if !slices.Equal(viaChain, viaFilters) {
		t.Fatal("chain ApplyFloat64 must equal applying each filter's ApplyFloat64 in order")
	}
}

// TestFilterChainResetClearsState pins that Reset returns the chain to a fresh
// stream so the same input reproduces the same output.
func TestFilterChainResetClearsState(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(2000)
	c := chainOf(t, cfg)

	first := make([]float32, len(x))
	if _, err := c.ProcessInto(x, first); err != nil {
		t.Fatal(err)
	}
	c.Reset()
	second := make([]float32, len(x))
	if _, err := c.ProcessInto(x, second); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(first, second) {
		t.Fatal("after Reset the same input must produce the same output")
	}
}

// TestFilterChainBufferTooSmallConsumesNothing pins the Processor contract for a
// chain: an undersized out returns ErrBufferTooSmall and leaves every filter's
// state untouched, so a following full call equals a reference chain's output. It
// also pins the empty-input branch (nil and zero-length return 0, nil) and the
// MaxOutputLen/Latency accessors.
func TestFilterChainBufferTooSmallConsumesNothing(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(2000)

	refOut := make([]float32, len(x))
	if _, err := chainOf(t, cfg).ProcessInto(x, refOut); err != nil {
		t.Fatal(err)
	}

	c := chainOf(t, cfg)
	if _, err := c.ProcessInto(x, make([]float32, len(x)-1)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	out := make([]float32, len(x))
	if _, err := c.ProcessInto(x, out); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(out, refOut) {
		t.Fatal("chain state changed after an undersized call")
	}

	// Empty input consumes nothing and returns no error, for both nil and a
	// zero-length slice.
	if n, err := c.ProcessInto(nil, out); n != 0 || err != nil {
		t.Fatalf("ProcessInto(nil, out) = (%d, %v), want (0, nil)", n, err)
	}
	if n, err := c.ProcessInto([]float32{}, out); n != 0 || err != nil {
		t.Fatalf("ProcessInto(empty, out) = (%d, %v), want (0, nil)", n, err)
	}

	// Accessors report one output sample per input and zero latency.
	if got := c.MaxOutputLen(len(x)); got != len(x) {
		t.Errorf("MaxOutputLen(%d) = %d, want %d", len(x), got, len(x))
	}
	if got := c.Latency(); got != 0 {
		t.Errorf("Latency() = %d, want 0", got)
	}
}
