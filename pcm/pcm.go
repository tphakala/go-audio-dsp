package pcm

import (
	"encoding/binary"
	"errors"
	"math"
	"unsafe"

	simdf32 "github.com/tphakala/simd/f32"
)

// Scaling conventions for the int16 <-> float32 boundary. int16 is exactly
// representable in float32 and 32768 is a power of two, so forward-then-back is
// a lossless round-trip for every int16 value.
const (
	int16ToFloatScale = float32(1.0 / 32768.0)
	floatToInt16Scale = float32(32768.0)
)

// ErrOddByteLength reports a byte slice whose length is not a multiple of two,
// so it cannot hold whole little-endian int16 samples.
var ErrOddByteLength = errors.New("pcm: byte slice length is not a multiple of 2")

// nativeLittleEndian is true when the host stores multi-byte integers
// little-endian. On such hosts (amd64, arm64, 386 and the rest of Go's SIMD
// targets) a []byte of little-endian int16 PCM and a []int16 share one memory
// layout, so byte conversion reinterprets with no copy; big-endian hosts take a
// scalar byte-swap path.
var nativeLittleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// Int16ToFloat32 converts int16 PCM to float32 in [-1, 1), scaling by 1/32768,
// and writes into dst. It converts min(len(dst), len(src)) samples and returns
// that count; a shorter dst yields a partial conversion, not an error. The
// conversion is exact and SIMD-accelerated. dst and src must not overlap.
func Int16ToFloat32(dst []float32, src []int16) int {
	n := min(len(dst), len(src))
	simdf32.Int16ToFloat32Scale(dst[:n], src[:n], int16ToFloatScale)
	return n
}

// Float32ToInt16 converts float32 PCM (nominal [-1, 1]) to int16, scaling by
// 32768 with round-to-nearest-ties-to-even and saturation to the int16 range so
// a full-scale or out-of-range sample never wraps. It converts min(len(dst),
// len(src)) samples and returns that count; a shorter dst yields a partial
// conversion, not an error. SIMD-accelerated. dst and src must not overlap.
func Float32ToInt16(dst []int16, src []float32) int {
	n := min(len(dst), len(src))
	simdf32.Float32ToInt16Scale(dst[:n], src[:n], floatToInt16Scale)
	return n
}

// BytesToFloat32 decodes interleaved little-endian int16 PCM from src into dst
// as float32 in [-1, 1). len(src) must be even (two bytes per sample); it
// decodes min(len(dst), len(src)/2) samples and returns that count, so a
// shorter dst yields a partial decode rather than an error. On little-endian
// hosts the bytes are reinterpreted as int16 with no copy and converted with
// SIMD; big-endian hosts decode byte by byte.
func BytesToFloat32(dst []float32, src []byte) (int, error) {
	if len(src)%2 != 0 {
		return 0, ErrOddByteLength
	}
	if nativeLittleEndian {
		return Int16ToFloat32(dst, bytesAsInt16(src)), nil
	}
	return bytesToFloat32LE(dst, src), nil
}

// Float32ToBytes encodes float32 PCM (nominal [-1, 1]) from src into dst as
// interleaved little-endian int16, scaling by 32768 with round-to-even and
// saturation. len(dst) must be even; it encodes min(len(dst)/2, len(src))
// samples and returns that count, so a shorter dst yields a partial encode
// rather than an error. On little-endian hosts dst is reinterpreted as int16
// and converted with SIMD; big-endian hosts encode byte by byte.
func Float32ToBytes(dst []byte, src []float32) (int, error) {
	if len(dst)%2 != 0 {
		return 0, ErrOddByteLength
	}
	if nativeLittleEndian {
		return Float32ToInt16(bytesAsInt16(dst), src), nil
	}
	return float32ToBytesLE(dst, src), nil
}

// InPlaceInt16 exposes b, interleaved little-endian int16 PCM, to apply as an
// []int16 and reflects any modification apply makes back into b. It allocates
// nothing on little-endian hosts, where the slice aliases b's backing array;
// big-endian hosts decode into a scratch slice and re-encode afterward. A
// trailing odd byte (when len(b) is odd) is left untouched. This lets code that
// works on []int16 operate directly on a []byte transport buffer with no copy.
func InPlaceInt16(b []byte, apply func(samples []int16)) {
	n := len(b) / 2
	if n == 0 {
		return
	}
	if nativeLittleEndian {
		apply(bytesAsInt16(b))
		return
	}
	buf := make([]int16, n)
	bytesToInt16LE(buf, b)
	apply(buf)
	int16ToBytesLE(b, buf)
}

// bytesAsInt16 reinterprets a little-endian int16 byte slice as []int16 without
// copying: the result aliases b's backing array, so writes through it modify b.
// Valid only on little-endian hosts. A trailing odd byte is dropped.
//
// SAFETY: &b[0] is taken only when n > 0, so an empty slice never indexes out of
// range. The result length is exactly len(b)/2, so no read reaches past b. The
// returned slice's data pointer keeps b's backing array alive for its lifetime,
// and int16 holds no pointers so it adds no GC scan cost. The int16 view can be
// unaligned when b starts at an odd address (a []byte carries no alignment
// guarantee); unaligned int16 access is defined and safe on the little-endian
// SIMD targets this path runs on (amd64, arm64, 386) and does not fault under
// the race detector's checkptr.
func bytesAsInt16(b []byte) []int16 {
	n := len(b) / 2
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int16)(unsafe.Pointer(&b[0])), n)
}

// bytesToFloat32LE converts little-endian int16 bytes to scaled float32, correct
// on any host. It is the byte-swap path taken on big-endian hosts and the
// reference the fast path is tested against. It decodes min(len(dst), len(src)/2)
// samples and returns that count; len(src) must be even.
func bytesToFloat32LE(dst []float32, src []byte) int {
	n := min(len(dst), len(src)/2)
	for i := range n {
		v := int16(binary.LittleEndian.Uint16(src[2*i:]))
		dst[i] = float32(v) * int16ToFloatScale
	}
	return n
}

// float32ToBytesLE converts scaled float32 to little-endian int16 bytes, correct
// on any host. It encodes min(len(dst)/2, len(src)) samples and returns that
// count; len(dst) must be even.
func float32ToBytesLE(dst []byte, src []float32) int {
	n := min(len(dst)/2, len(src))
	for i := range n {
		binary.LittleEndian.PutUint16(dst[2*i:], uint16(scalarFloatToInt16(src[i])))
	}
	return n
}

// bytesToInt16LE decodes little-endian int16 bytes into dst, correct on any
// host. len(src) must be even and len(dst) >= len(src)/2.
func bytesToInt16LE(dst []int16, src []byte) {
	for i := range len(src) / 2 {
		dst[i] = int16(binary.LittleEndian.Uint16(src[2*i:]))
	}
}

// int16ToBytesLE encodes int16 samples into dst as little-endian bytes, correct
// on any host. len(dst) must be >= 2*len(src).
func int16ToBytesLE(dst []byte, src []int16) {
	for i, v := range src {
		binary.LittleEndian.PutUint16(dst[2*i:], uint16(v))
	}
}

// scalarFloatToInt16 mirrors simd's Float32ToInt16Scale for one sample on
// big-endian hosts: scale by 32768, round to nearest (ties to even), and
// saturate to the int16 range, with +Inf -> 32767, -Inf -> -32768, NaN -> 0.
func scalarFloatToInt16(f float32) int16 {
	if f != f { // NaN
		return 0
	}
	v := math.RoundToEven(float64(f) * float64(floatToInt16Scale))
	if v >= math.MaxInt16 {
		return math.MaxInt16
	}
	if v <= math.MinInt16 {
		return math.MinInt16
	}
	return int16(v)
}
