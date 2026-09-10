package equalizer

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// eqSignal returns n samples of a deterministic multi-sine test signal.
func eqSignal(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		t := float64(i)
		x[i] = float32(0.5*math.Sin(0.02*t) + 0.3*math.Sin(0.13*t+0.7) + 0.2*math.Sin(0.5*t))
	}
	return x
}

// testConfig is a representative 3-band chain used across the processing tests.
func testConfig() Config {
	return Config{SampleRate: 48000, Bands: []Band{
		{Type: HighPass, Frequency: 80, Q: 0.7071},
		{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6},
		{Type: LowShelf, Frequency: 200, Q: 0.7071, GainDB: -3},
	}}
}

// whole runs x through a fresh equalizer in one call.
func whole(t *testing.T, cfg Config, x []float32) []float32 {
	t.Helper()
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(x))
	if _, err := e.ProcessInto(x, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestChunkInvariance(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(10000)
	ref := whole(t, cfg, x)

	for _, chunk := range []int{1, 7, 333, 1000} {
		e, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 0, len(x))
		buf := make([]float32, chunk)
		for i := 0; i < len(x); i += chunk {
			end := min(i+chunk, len(x))
			in := x[i:end]
			n, err := e.ProcessInto(in, buf[:len(in)])
			if err != nil {
				t.Fatalf("chunk %d at %d: %v", chunk, i, err)
			}
			got = append(got, buf[:n]...)
		}
		if !slices.Equal(got, ref) {
			for i := range ref {
				if got[i] != ref[i] {
					t.Fatalf("chunk %d: sample %d = %v, whole-buffer = %v", chunk, i, got[i], ref[i])
				}
			}
		}
	}
}

func TestBufferTooSmallConsumesNothing(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(2000)
	ref := whole(t, cfg, x)

	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// An undersized call must return ErrBufferTooSmall and touch no state.
	if _, err := e.ProcessInto(x, make([]float32, len(x)-1)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	out := make([]float32, len(x))
	if _, err := e.ProcessInto(x, out); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(out, ref) {
		t.Fatal("output after a rejected call differs from a fresh run (state was touched)")
	}
}

func TestResetClearsState(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(3000)
	ref := whole(t, cfg, x)

	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(x))
	if _, err := e.ProcessInto(x, out); err != nil { // dirty the state
		t.Fatal(err)
	}
	e.Reset()
	if _, err := e.ProcessInto(x, out); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(out, ref) {
		t.Fatal("output after Reset differs from a fresh run")
	}
}

func TestEmptyConfigIsIdentity(t *testing.T) {
	e, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	if e.NumSections() != 0 {
		t.Fatalf("NumSections() = %d, want 0 for empty config", e.NumSections())
	}
	x := eqSignal(500)
	out := make([]float32, len(x))
	n, err := e.ProcessInto(x, out)
	if err != nil || n != len(x) {
		t.Fatalf("ProcessInto n=%d err=%v", n, err)
	}
	if !slices.Equal(out, x) {
		t.Fatal("empty equalizer changed the signal")
	}
}

func TestAliasedInPlace(t *testing.T) {
	cfg := testConfig()
	x := eqSignal(4000)
	ref := whole(t, cfg, x)

	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	aliased := slices.Clone(x)
	if _, err := e.ProcessInto(aliased, aliased); err != nil { // out == in
		t.Fatal(err)
	}
	if !slices.Equal(aliased, ref) {
		t.Fatal("in-place ProcessInto differs from the separate-buffer result")
	}
}

func TestProcessIntoEmpty(t *testing.T) {
	e, err := New(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	if n, err := e.ProcessInto(nil, nil); n != 0 || err != nil {
		t.Errorf("ProcessInto(nil,nil) = %d,%v, want 0,nil", n, err)
	}
	if n, err := e.ProcessInto([]float32{}, make([]float32, 4)); n != 0 || err != nil {
		t.Errorf("ProcessInto(empty) = %d,%v, want 0,nil", n, err)
	}
}
