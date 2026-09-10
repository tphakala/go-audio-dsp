package pcm

import (
	"errors"
	"math"
	"testing"

	simdf32 "github.com/tphakala/simd/f32"
)

// TestInt16Float32RoundTripExhaustive checks that every int16 survives the
// int16 -> float32 -> int16 round-trip unchanged. This holds only because the
// scale pair is 1/32768 and 32768 (both exact); a truncating or 1/32767 scale
// would drift.
func TestInt16Float32RoundTripExhaustive(t *testing.T) {
	src := make([]int16, 65536)
	for i := range src {
		src[i] = int16(i - 32768)
	}
	f := make([]float32, len(src))
	Int16ToFloat32(f, src)
	got := make([]int16, len(src))
	Float32ToInt16(got, f)
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("round-trip int16 %d -> %v -> %d", src[i], f[i], got[i])
		}
	}
}

// TestInt16ToFloat32Values pins the exact float32 for the endpoints and a few
// small magnitudes, so a scale change is caught.
func TestInt16ToFloat32Values(t *testing.T) {
	src := []int16{-32768, -1, 0, 1, 32767}
	want := []float32{-1.0, -1.0 / 32768, 0, 1.0 / 32768, 32767.0 / 32768}
	dst := make([]float32, len(src))
	if n := Int16ToFloat32(dst, src); n != len(src) {
		t.Fatalf("n = %d, want %d", n, len(src))
	}
	for i := range src {
		if dst[i] != want[i] {
			t.Fatalf("Int16ToFloat32(%d) = %v, want %v", src[i], dst[i], want[i])
		}
	}
}

// TestFloat32ToInt16SaturateAndRound pins saturation past full scale, the
// round-to-even tie rule, and the NaN/Inf mapping.
func TestFloat32ToInt16SaturateAndRound(t *testing.T) {
	// Values in int16-magnitude units divided by 32768 so scale returns them.
	const s = 1.0 / 32768.0
	cases := []struct {
		in   float32
		want int16
	}{
		{2.0, 32767},   // past +full scale saturates
		{-2.0, -32768}, // past -full scale saturates
		{1.0, 32767},   // +1.0*32768 = 32768 saturates to 32767
		{-1.0, -32768}, // -1.0*32768 = -32768 in range
		{0.5 * s, 0},   // 0.5 ties to even -> 0
		{1.5 * s, 2},   // 1.5 ties to even -> 2
		{2.5 * s, 2},   // 2.5 ties to even -> 2
		{float32(math.Inf(1)), 32767},
		{float32(math.Inf(-1)), -32768},
		{float32(math.NaN()), 0},
	}
	for _, c := range cases {
		dst := make([]int16, 1)
		Float32ToInt16(dst, []float32{c.in})
		if dst[0] != c.want {
			t.Fatalf("Float32ToInt16(%v) = %d, want %d", c.in, dst[0], c.want)
		}
	}
}

// TestBytesToFloat32ByteOrder pins little-endian decoding for the endpoints.
func TestBytesToFloat32ByteOrder(t *testing.T) {
	// LE bytes: 0x0000=0, 0x0001(LE)= {0x01,0x00}=1, 0x8000={0x00,0x80}=-32768,
	// 0x7fff={0xff,0x7f}=32767.
	src := []byte{0x00, 0x00, 0x01, 0x00, 0x00, 0x80, 0xff, 0x7f}
	want := []float32{0, 1.0 / 32768, -1.0, 32767.0 / 32768}
	dst := make([]float32, 4)
	n, err := BytesToFloat32(dst, src)
	if err != nil || n != 4 {
		t.Fatalf("BytesToFloat32 n=%d err=%v", n, err)
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("sample %d = %v, want %v", i, dst[i], want[i])
		}
	}
}

// TestBytesFloat32RoundTrip checks bytes -> float32 -> bytes is identity for a
// full sweep of int16 values encoded little-endian.
func TestBytesFloat32RoundTrip(t *testing.T) {
	src := make([]byte, 65536*2)
	for v := range 65536 {
		u := uint16(int16(v - 32768))
		src[2*v] = byte(u)
		src[2*v+1] = byte(u >> 8)
	}
	f := make([]float32, 65536)
	if _, err := BytesToFloat32(f, src); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, len(src))
	if _, err := Float32ToBytes(out, f); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if out[i] != src[i] {
			t.Fatalf("byte %d = %#x, want %#x", i, out[i], src[i])
		}
	}
}

func TestOddByteLength(t *testing.T) {
	if _, err := BytesToFloat32(make([]float32, 4), make([]byte, 3)); !errors.Is(err, ErrOddByteLength) {
		t.Fatalf("BytesToFloat32 odd src: err = %v, want ErrOddByteLength", err)
	}
	if _, err := Float32ToBytes(make([]byte, 3), make([]float32, 4)); !errors.Is(err, ErrOddByteLength) {
		t.Fatalf("Float32ToBytes odd dst: err = %v, want ErrOddByteLength", err)
	}
}

// TestPartialConversion checks a short dst yields a partial count, not an error.
func TestPartialConversion(t *testing.T) {
	src := []int16{1, 2, 3, 4}
	dst := make([]float32, 2)
	if n := Int16ToFloat32(dst, src); n != 2 {
		t.Fatalf("Int16ToFloat32 short dst: n = %d, want 2", n)
	}
	bsrc := []byte{1, 0, 2, 0, 3, 0}
	bdst := make([]float32, 1)
	if n, err := BytesToFloat32(bdst, bsrc); err != nil || n != 1 {
		t.Fatalf("BytesToFloat32 short dst: n=%d err=%v, want 1,nil", n, err)
	}
}

// TestInPlaceInt16 checks the closure sees the samples and its edits are written
// back into the byte slice.
func TestInPlaceInt16(t *testing.T) {
	// int16 {100, -200, 300} little-endian.
	orig := []int16{100, -200, 300}
	b := make([]byte, len(orig)*2)
	int16ToBytesLE(b, orig)

	var seen []int16
	InPlaceInt16(b, func(s []int16) {
		seen = append(seen, s...)
		for i := range s {
			s[i] *= 2
		}
	})
	if len(seen) != len(orig) {
		t.Fatalf("saw %d samples, want %d", len(seen), len(orig))
	}
	for i := range orig {
		if seen[i] != orig[i] {
			t.Fatalf("closure saw sample %d = %d, want %d", i, seen[i], orig[i])
		}
	}
	// Read the bytes back: must be the doubled values.
	got := make([]int16, len(orig))
	bytesToInt16LE(got, b)
	for i := range orig {
		if got[i] != orig[i]*2 {
			t.Fatalf("written-back sample %d = %d, want %d", i, got[i], orig[i]*2)
		}
	}
}

// TestInPlaceInt16OddTrailingByte checks a trailing odd byte is left untouched.
func TestInPlaceInt16OddTrailingByte(t *testing.T) {
	b := []byte{0x01, 0x00, 0x42} // one sample (1) plus a stray byte
	InPlaceInt16(b, func(s []int16) {
		if len(s) != 1 || s[0] != 1 {
			t.Fatalf("samples = %v, want [1]", s)
		}
		s[0] = 7
	})
	if b[2] != 0x42 {
		t.Fatalf("trailing byte = %#x, want 0x42", b[2])
	}
	if b[0] != 0x07 || b[1] != 0x00 {
		t.Fatalf("sample bytes = %#x %#x, want 0x07 0x00", b[0], b[1])
	}
}

// TestScalarFloatToInt16MatchesSIMD verifies the big-endian scalar mirror agrees
// with the SIMD primitive across a sweep, so the fallback path is numerically
// identical to the fast path.
func TestScalarFloatToInt16MatchesSIMD(t *testing.T) {
	in := make([]float32, 0, 4004)
	for i := -2000; i <= 2000; i++ {
		in = append(in, float32(i)*0.001) // -2.0 .. 2.0, crosses saturation
	}
	// Non-finite inputs: the scalar mirror must map them the way SIMD does
	// (NaN -> 0, +Inf -> 32767, -Inf -> -32768).
	in = append(in, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)))
	want := make([]int16, len(in))
	simdf32.Float32ToInt16Scale(want, in, floatToInt16Scale)
	for i := range in {
		if got := scalarFloatToInt16(in[i]); got != want[i] {
			t.Fatalf("scalarFloatToInt16(%v) = %d, SIMD = %d", in[i], got, want[i])
		}
	}
}

// TestBytesToFloat32LEMatchesFastPath verifies the scalar byte-swap decode
// matches the little-endian reinterpret fast path for arbitrary bytes.
func TestBytesToFloat32LEMatchesFastPath(t *testing.T) {
	src := make([]byte, 512)
	for i := range src {
		src[i] = byte(i*37 + 11)
	}
	fast := make([]float32, len(src)/2)
	Int16ToFloat32(fast, bytesAsInt16(src))
	scalar := make([]float32, len(src)/2)
	bytesToFloat32LE(scalar, src)
	for i := range fast {
		if fast[i] != scalar[i] {
			t.Fatalf("sample %d: fast %v != scalar %v", i, fast[i], scalar[i])
		}
	}
	// And float32ToBytesLE is the inverse of bytesToFloat32LE for
	// exactly-represented values (whole int16 round-trip through the byte forms).
	back := make([]byte, len(src))
	float32ToBytesLE(back, scalar)
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("byte %d: encode(decode) %#x != %#x", i, back[i], src[i])
		}
	}
}

// TestBigEndianFallbackPaths forces the scalar big-endian branches on a
// little-endian host so the composed fallback (which no CI architecture runs)
// is exercised end to end, not just its helpers in isolation. The helpers do
// explicit little-endian byte math, so results must match the fast path.
func TestBigEndianFallbackPaths(t *testing.T) {
	saved := nativeLittleEndian
	nativeLittleEndian = false
	defer func() { nativeLittleEndian = saved }()

	// LE bytes: {0x01,0x00}=1, {0x00,0x80}=-32768, {0xff,0x7f}=32767.
	src := []byte{0x00, 0x00, 0x01, 0x00, 0x00, 0x80, 0xff, 0x7f}
	dst := make([]float32, 4)
	if n, err := BytesToFloat32(dst, src); err != nil || n != 4 {
		t.Fatalf("BE BytesToFloat32: n=%d err=%v", n, err)
	}
	want := []float32{0, 1.0 / 32768, -1.0, 32767.0 / 32768}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("BE BytesToFloat32[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
	out := make([]byte, len(src))
	if n, err := Float32ToBytes(out, dst); err != nil || n != 4 {
		t.Fatalf("BE Float32ToBytes: n=%d err=%v", n, err)
	}
	for i := range src {
		if out[i] != src[i] {
			t.Fatalf("BE round-trip byte %d = %#x, want %#x", i, out[i], src[i])
		}
	}
	// InPlaceInt16 BE branch: decode into scratch, mutate, re-encode write-back.
	b := make([]byte, 6)
	int16ToBytesLE(b, []int16{10, -20, 30})
	InPlaceInt16(b, func(s []int16) {
		if len(s) != 3 || s[0] != 10 || s[1] != -20 || s[2] != 30 {
			t.Fatalf("BE InPlaceInt16 samples = %v, want [10 -20 30]", s)
		}
		for i := range s {
			s[i]++
		}
	})
	got := make([]int16, 3)
	bytesToInt16LE(got, b)
	if got[0] != 11 || got[1] != -19 || got[2] != 31 {
		t.Fatalf("BE InPlaceInt16 write-back = %v, want [11 -19 31]", got)
	}
}
