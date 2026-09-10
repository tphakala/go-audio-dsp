package gain

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

func TestFactorFromDB(t *testing.T) {
	cases := []struct {
		dB   float64
		want float64
		tol  float64
	}{
		{0, 1, 0},                    // exact
		{20, 10, 1e-9},               // +20 dB = 10x
		{-20, 0.1, 1e-12},            // -20 dB = 0.1x
		{6.020599913279624, 2, 1e-9}, // +6.02 dB = 2x
		{-6.020599913279624, 0.5, 1e-9},
	}
	for _, c := range cases {
		got := FactorFromDB(c.dB)
		if math.Abs(got-c.want) > c.tol {
			t.Errorf("FactorFromDB(%g) = %g, want %g (tol %g)", c.dB, got, c.want, c.tol)
		}
	}
	if FactorFromDB(0) != 1 {
		t.Errorf("FactorFromDB(0) must be exactly 1, got %g", FactorFromDB(0))
	}
}

func TestNewRejectsNonFinite(t *testing.T) {
	for _, dB := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := New(dB); !errors.Is(err, dsp.ErrInvalidConfig) {
			t.Errorf("New(%g) err = %v, want dsp.ErrInvalidConfig", dB, err)
		}
	}
	g, err := New(6)
	if err != nil {
		t.Fatalf("New(6): %v", err)
	}
	if g.GainDB() != 6 {
		t.Errorf("GainDB() = %g, want 6", g.GainDB())
	}
}

func TestProcessIntoScales(t *testing.T) {
	g, err := New(-20) // factor 0.1
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{1, -0.5, 0.25, 0}
	out := make([]float32, len(in))
	n, err := g.ProcessInto(in, out)
	if err != nil || n != len(in) {
		t.Fatalf("ProcessInto n=%d err=%v", n, err)
	}
	for i := range in {
		want := in[i] * g.factor // Scale is no-fuse, so this is bit-exact
		if out[i] != want {
			t.Errorf("out[%d] = %v, want %v", i, out[i], want)
		}
	}
}

func TestProcessIntoUnityCopies(t *testing.T) {
	g, err := New(0) // factor 1
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{0.5, -0.25, 0.125}
	out := []float32{9, 9, 9} // sentinel, distinct buffer
	if _, err := g.ProcessInto(in, out); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(out, in) {
		t.Errorf("unity ProcessInto out = %v, want %v", out, in)
	}
}

func TestProcessIntoAliased(t *testing.T) {
	g, err := New(-6) // factor ~0.5, non-unity
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{1, -0.5, 0.25, -0.9, 0.7}
	ref := make([]float32, len(in))
	if _, err := g.ProcessInto(in, ref); err != nil {
		t.Fatal(err)
	}
	aliased := slices.Clone(in)
	if _, err := g.ProcessInto(aliased, aliased); err != nil { // out == in
		t.Fatal(err)
	}
	if !slices.Equal(aliased, ref) {
		t.Errorf("aliased ProcessInto = %v, want %v", aliased, ref)
	}
}

func TestProcessIntoBufferTooSmall(t *testing.T) {
	g, _ := New(3)
	in := []float32{1, 2, 3, 4}
	out := []float32{7, 7, 7} // one short
	n, err := g.ProcessInto(in, out)
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	if n != 0 {
		t.Errorf("n = %d, want 0 on undersize", n)
	}
	if !slices.Equal(out, []float32{7, 7, 7}) {
		t.Errorf("out was written on undersize: %v", out)
	}
	if !slices.Equal(in, []float32{1, 2, 3, 4}) {
		t.Errorf("in was modified on undersize: %v", in)
	}
}

func TestProcessIntoEmpty(t *testing.T) {
	g, _ := New(6)
	if n, err := g.ProcessInto(nil, nil); n != 0 || err != nil {
		t.Errorf("ProcessInto(nil,nil) = %d,%v, want 0,nil", n, err)
	}
	if n, err := g.ProcessInto([]float32{}, []float32{}); n != 0 || err != nil {
		t.Errorf("ProcessInto(empty) = %d,%v, want 0,nil", n, err)
	}
	if n, err := g.ProcessInto(nil, make([]float32, 4)); n != 0 || err != nil {
		t.Errorf("ProcessInto(nil,out) = %d,%v, want 0,nil", n, err)
	}
}

func TestApplyInt16Saturates(t *testing.T) {
	g, _ := New(20) // factor 10
	s := []int16{20000, -20000, 100, -100}
	g.ApplyInt16(s)
	want := []int16{32767, -32768, 1000, -1000}
	if !slices.Equal(s, want) {
		t.Errorf("ApplyInt16 = %v, want %v (saturate, scaled by factor)", s, want)
	}
}

func TestApplyBytesOddLength(t *testing.T) {
	g, _ := New(6)
	b := []byte{1, 0, 2} // 3 bytes, odd
	orig := slices.Clone(b)
	if err := g.ApplyBytes(b); !errors.Is(err, ErrOddByteLength) {
		t.Fatalf("err = %v, want ErrOddByteLength", err)
	}
	if !slices.Equal(b, orig) {
		t.Errorf("odd-length bytes were modified: %v", b)
	}
}

func TestApplyBytesMatchesApplyInt16(t *testing.T) {
	g, _ := New(9) // arbitrary non-unity gain
	samples := []int16{0, 1, -1, 12345, -12345, 4000, -4000, 32767, -32768}

	viaInt16 := slices.Clone(samples)
	g.ApplyInt16(viaInt16)

	b := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	if err := g.ApplyBytes(b); err != nil {
		t.Fatal(err)
	}
	viaBytes := make([]int16, len(samples))
	for i := range viaBytes {
		viaBytes[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	if !slices.Equal(viaBytes, viaInt16) {
		t.Errorf("ApplyBytes = %v, ApplyInt16 = %v (must agree)", viaBytes, viaInt16)
	}
}
