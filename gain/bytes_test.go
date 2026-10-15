package gain

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/pcm"
)

var testGains = []float64{0, 1e-7, -60, -12.3, -0.5, 0.1, 3, 6.02, 12.7, 30, 60, 120}

func leBytes(s []int16) []byte {
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func allInt16() []int16 {
	s := make([]int16, 65536)
	for i := range s {
		s[i] = int16(i - 32768)
	}
	return s
}

func garbage(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = 0xA5
	}
	return b
}

func TestApplyBytesToMatchesApplyBytes(t *testing.T) {
	src := leBytes(allInt16())
	for _, db := range testGains {
		g, err := New(db)
		if err != nil {
			t.Fatal(err)
		}
		want := slices.Clone(src)
		if err := g.ApplyBytes(want); err != nil {
			t.Fatal(err)
		}
		dst := garbage(len(src)) // a skipped copy at unity would leave this behind
		if err := g.ApplyBytesTo(dst, src); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(dst, want) {
			t.Errorf("%g dB: ApplyBytesTo differs from ApplyBytes", db)
		}
	}
}

func TestApplyBytesToLeavesSrc(t *testing.T) {
	g, _ := New(6)
	src := leBytes([]int16{1, -2, 3000})
	orig := slices.Clone(src)
	dst := make([]byte, len(src))
	if err := g.ApplyBytesTo(dst, src); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(src, orig) {
		t.Error("src was modified")
	}
}

func TestApplyBytesToErrors(t *testing.T) {
	for _, db := range []float64{0, 6} {
		g, _ := New(db)
		dst := garbage(8)

		if err := g.ApplyBytesTo(dst, make([]byte, 3)); !errors.Is(err, ErrOddByteLength) {
			t.Errorf("%g dB odd: err = %v", db, err)
		}
		if err := g.ApplyBytesTo(dst, make([]byte, 10)); !errors.Is(err, ErrBufferTooSmall) {
			t.Errorf("%g dB short dst: err = %v", db, err)
		}
		if !slices.Equal(dst, garbage(8)) {
			t.Errorf("%g dB: dst written on error", db)
		}
	}
}

func TestAppliedBytesOddLengthAtUnity(t *testing.T) {
	g, _ := New(0)
	if _, err := g.AppliedBytes(make([]byte, 3)); !errors.Is(err, ErrOddByteLength) {
		t.Errorf("err = %v, want ErrOddByteLength", err)
	}
}

func TestAppliedBytesUnityReturnsSrc(t *testing.T) {
	g, _ := New(0)
	src := leBytes([]int16{1, 2, 3})
	out, err := g.AppliedBytes(src)
	if err != nil {
		t.Fatal(err)
	}
	if &out[0] != &src[0] || len(out) != len(src) {
		t.Error("0 dB must return src itself")
	}
}

func TestAppliedBytesCopies(t *testing.T) {
	g, _ := New(6)
	src := leBytes([]int16{1000, -1000, 5})
	orig := slices.Clone(src)
	out, err := g.AppliedBytes(src)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(src, orig) {
		t.Error("src was modified")
	}
	if slices.Equal(out, src) || &out[0] == &src[0] {
		t.Error("result must be a distinct, scaled copy")
	}
	want := slices.Clone(src)
	_ = g.ApplyBytes(want)
	if !slices.Equal(out, want) {
		t.Error("AppliedBytes differs from ApplyBytes")
	}
}

func TestApplyBytesSaturatesAtLargeBoost(t *testing.T) {
	g, _ := New(120)
	b := leBytes([]int16{1, -1, 300, -300, 32767, -32768, 0})
	if err := g.ApplyBytes(b); err != nil {
		t.Fatal(err)
	}
	want := leBytes([]int16{32767, -32768, 32767, -32768, 32767, -32768, 0})
	if !slices.Equal(b, want) {
		t.Errorf("got %v, want %v", b, want)
	}
}

// TestApplyBytesWithinOneLSBOfFloat64Reference documents the rounding contract:
// float32 ties-to-even may differ from a float64 round-half-away reference by
// one LSB, never more, over every int16 input.
func TestApplyBytesWithinOneLSBOfFloat64Reference(t *testing.T) {
	samples := allInt16()
	for _, db := range testGains {
		g, _ := New(db)
		b := leBytes(samples)
		if err := g.ApplyBytes(b); err != nil {
			t.Fatal(err)
		}
		factor := math.Pow(10, db/20)
		for i, s := range samples {
			ref := math.Max(math.MinInt16, math.Min(math.MaxInt16, math.Round(float64(s)*factor)))
			got := int16(binary.LittleEndian.Uint16(b[2*i:]))
			if d := math.Abs(float64(got) - ref); d > 1 {
				t.Fatalf("%g dB sample %d: got %d, reference %v", db, s, got, ref)
			}
		}
	}
}

func TestApplyBytesMatchesScaleBytes(t *testing.T) {
	g, _ := New(-3)
	src := leBytes(allInt16())
	want := make([]byte, len(src))
	if _, err := pcm.ScaleBytes(want, src, float32(math.Pow(10, -3.0/20))); err != nil {
		t.Fatal(err)
	}
	got := slices.Clone(src)
	if err := g.ApplyBytes(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Error("ApplyBytes differs from pcm.ScaleBytes")
	}
}
