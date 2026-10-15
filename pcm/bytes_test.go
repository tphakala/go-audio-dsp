package pcm

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"
)

// bothPaths runs fn on the zero-copy path (where the host allows it) and on the
// byte-wise path big-endian and trap-on-unaligned hosts take.
func bothPaths(t *testing.T, fn func(t *testing.T)) {
	t.Helper()
	saved := canReinterpretInt16
	defer func() { canReinterpretInt16 = saved }()
	for _, reinterpret := range []bool{saved, false} {
		canReinterpretInt16 = reinterpret
		name := "bytewise"
		if reinterpret {
			name = "reinterpret"
		}
		t.Run(name, fn)
		if !saved {
			return
		}
	}
}

func leBytes(s []int16) []byte {
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func testSamples(n int) []int16 {
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(i*7919 + i*i)
	}
	return s
}

// misaligned returns a copy of b that starts at an odd offset of a larger array.
func misaligned(b []byte) []byte {
	big := make([]byte, len(b)+3)
	out := big[1 : 1+len(b)]
	copy(out, b)
	return out
}

func TestBytesToInt16(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		want := testSamples(777)
		src := misaligned(leBytes(want))
		orig := slices.Clone(src)
		dst := make([]int16, len(want))
		n, err := BytesToInt16(dst, src)
		if err != nil || n != len(want) {
			t.Fatalf("n=%d err=%v", n, err)
		}
		if !slices.Equal(dst, want) {
			t.Error("decoded samples differ")
		}
		if !slices.Equal(src, orig) {
			t.Error("src was modified")
		}

		short := make([]int16, 10)
		if n, err := BytesToInt16(short, src); err != nil || n != 10 || !slices.Equal(short, want[:10]) {
			t.Errorf("short dst: n=%d err=%v", n, err)
		}
		if n, err := BytesToInt16(dst, src[:0]); err != nil || n != 0 {
			t.Errorf("empty src: n=%d err=%v", n, err)
		}
		if _, err := BytesToInt16(dst, src[:3]); !errors.Is(err, ErrOddByteLength) {
			t.Errorf("odd src err = %v", err)
		}
	})
}

var scaleGains = []float64{0, 1e-7, -60, -12.3, -0.5, 0.1, 3, 6.02, 12.7, 30, 60, 120}

func factorOf(db float64) float32 { return float32(math.Pow(10, db/20)) }

func TestScaleBytesMatchesScaleInt16(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		for _, db := range scaleGains {
			f := factorOf(db)
			for _, n := range []int{1, 2, scaleChunk - 1, scaleChunk, scaleChunk + 1, 3*scaleChunk + 5} {
				samples := testSamples(n)
				want := slices.Clone(samples)
				ScaleInt16(want, f)
				srcFull := leBytes(samples)
				src := misaligned(srcFull)
				garbage := make([]byte, 2*n+2)
				for i := range garbage {
					garbage[i] = 0xA5
				}
				dst := garbage[1 : 1+2*n]
				got, err := ScaleBytes(dst, src, f)
				if err != nil || got != n {
					t.Fatalf("db=%g n=%d: got=%d err=%v", db, n, got, err)
				}
				if !slices.Equal(dst, leBytes(want)) {
					t.Fatalf("db=%g n=%d: out-of-place result differs from ScaleInt16", db, n)
				}
				if !slices.Equal(src, srcFull) {
					t.Fatalf("db=%g n=%d: src modified", db, n)
				}
				if garbage[0] != 0xA5 || garbage[len(garbage)-1] != 0xA5 {
					t.Fatalf("db=%g n=%d: wrote outside dst", db, n)
				}
			}
		}
	})
}

func TestScaleBytesAliased(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		for _, db := range scaleGains {
			f := factorOf(db)
			samples := testSamples(2*scaleChunk + 3)
			want := slices.Clone(samples)
			ScaleInt16(want, f)
			b := misaligned(leBytes(samples))
			if n, err := ScaleBytes(b, b, f); err != nil || n != len(samples) {
				t.Fatalf("db=%g: n=%d err=%v", db, n, err)
			}
			if !slices.Equal(b, leBytes(want)) {
				t.Fatalf("db=%g: aliased result differs", db)
			}
		}
	})
}

func TestScaleBytesOddLength(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		src := []byte{1, 0, 2}
		dst := []byte{9, 9, 9, 9}
		for _, f := range []float32{1, 2} {
			if n, err := ScaleBytes(dst, src, f); !errors.Is(err, ErrOddByteLength) || n != 0 {
				t.Errorf("factor %g: n=%d err=%v", f, n, err)
			}
			if !slices.Equal(dst, []byte{9, 9, 9, 9}) {
				t.Errorf("factor %g: dst written on error", f)
			}
		}
	})
}

func TestScaleBytesShortDst(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		samples := testSamples(8)
		src := leBytes(samples)
		for _, f := range []float32{1, 2} {
			dst := make([]byte, 7) // three whole samples, one stray byte
			for i := range dst {
				dst[i] = 0x55
			}
			n, err := ScaleBytes(dst, src, f)
			if err != nil || n != 3 {
				t.Fatalf("factor %g: n=%d err=%v", f, n, err)
			}
			want := slices.Clone(samples[:3])
			ScaleInt16(want, f)
			if !slices.Equal(dst[:6], leBytes(want)) || dst[6] != 0x55 {
				t.Errorf("factor %g: partial result wrong: %v", f, dst)
			}
		}
	})
}

func TestScaleBytesSaturates(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		src := leBytes([]int16{1000, -1000, 32767, -32768})
		dst := make([]byte, len(src))
		if _, err := ScaleBytes(dst, src, factorOf(120)); err != nil {
			t.Fatal(err)
		}
		want := leBytes([]int16{32767, -32768, 32767, -32768})
		if !slices.Equal(dst, want) {
			t.Errorf("got %v, want %v", dst, want)
		}
	})
}

func TestScaleBytesEmpty(t *testing.T) {
	for _, f := range []float32{1, 2} {
		if n, err := ScaleBytes(make([]byte, 4), nil, f); err != nil || n != 0 {
			t.Errorf("empty src, factor %g: n=%d err=%v", f, n, err)
		}
		if n, err := ScaleBytes(nil, make([]byte, 4), f); err != nil || n != 0 {
			t.Errorf("empty dst, factor %g: n=%d err=%v", f, n, err)
		}
	}
}
