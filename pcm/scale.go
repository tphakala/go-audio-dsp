package pcm

import simdf32 "github.com/tphakala/simd/f32"

// scaleChunk bounds the stack scratch buffer used by ScaleInt16 so its memory
// stays O(1) regardless of input length. It is a constant so the scratch is a
// fixed-size stack array with no heap allocation.
//
// 1024 float32 is a 4 KiB scratch. The size trades loop and kernel amortization
// (larger is better) against two costs that grow with it and are paid on every
// call that does work: Go zeroes the stack scratch where ScaleInt16 declares it,
// and a scratch that overflows L1 evicts the samples being scaled. A 32 KiB
// scratch makes a short streaming frame pay a 32 KiB clear per call and spills
// L1; 4 KiB does neither. 1024 was picked from a one-off sweep of this constant
// benchmarked across both target architectures (amd64 and arm64): it roughly
// halves small-frame latency against a 32 KiB scratch and is at worst within a
// couple of percent on large buffers. BenchmarkScaleInt16 tracks the shipped
// value's per-size cost; re-run the sweep (edit this constant) to retune.
const scaleChunk = 1024

// ScaleInt16 multiplies every sample in s by factor in place, rounding each
// scaled value to the nearest integer (ties to even) and saturating to the
// int16 range so a boost can never wrap around. A factor of 1 or an empty s
// returns without touching s.
//
// The multiply and the saturating round run through the SIMD int16<->float32
// kernels (Int16ToFloat32Scale then Float32ToInt16Scale). Work proceeds in
// fixed-size chunks with a stack scratch, so ScaleInt16 allocates nothing
// regardless of len(s). The tie rule and saturation match Float32ToInt16, so
// int16 gain is consistent with the rest of the package.
func ScaleInt16(s []int16, factor float32) {
	if factor == 1 {
		return
	}
	scaleInt16To(s, s, factor)
}

// scaleInt16To is the one kernel loop behind ScaleInt16 and ScaleBytes: it
// writes src*factor into dst, rounding to nearest (ties to even) and saturating.
// dst may be exactly src; a shifted overlap is not supported. Both slices must
// have the same length (the SIMD kernels require matching lengths), which the
// callers guarantee. It does not special-case factor 1, because an out-of-place
// caller still has to copy there.
func scaleInt16To(dst, src []int16, factor float32) {
	var scratch [scaleChunk]float32
	for start := 0; start < len(src); start += scaleChunk {
		end := min(start+scaleChunk, len(src))
		seg := src[start:end]
		buf := scratch[:len(seg)]
		// buf = seg * factor in int16-magnitude units, then round-to-even and
		// saturate back into dst. Float32ToInt16Scale clamps to [-32768, 32767].
		simdf32.Int16ToFloat32Scale(buf, seg, factor)
		simdf32.Float32ToInt16Scale(dst[start:end], buf, 1.0)
	}
}

// ScaleBytes scales interleaved little-endian int16 PCM bytes from src into dst
// by factor, rounding to nearest (ties to even) and saturating exactly as
// ScaleInt16. It scales min(len(dst), len(src))/2 samples and returns that
// count, so a shorter dst yields a partial scale rather than an error. An odd
// len(src) returns ErrOddByteLength and writes nothing. dst may be exactly src
// for an in-place scale; a shifted overlap is not supported. A factor of 1
// still copies src into dst when they differ, so dst always holds the result.
//
// On hosts where the zero-copy reinterpret is safe (little-endian with
// unaligned access) both buffers are viewed as int16 and scaled with SIMD;
// elsewhere samples are decoded and encoded byte by byte through a fixed
// scratch. ScaleBytes allocates nothing in either case and never writes src
// unless dst aliases it.
func ScaleBytes(dst, src []byte, factor float32) (int, error) {
	if len(src)%2 != 0 {
		return 0, ErrOddByteLength
	}
	n := min(len(dst), len(src)) / 2
	if n == 0 {
		return 0, nil
	}
	dst, src = dst[:2*n], src[:2*n]
	if factor == 1 {
		copy(dst, src) // a no-op when dst is src
		return n, nil
	}
	if canReinterpretInt16 {
		scaleInt16To(bytesAsInt16(dst), bytesAsInt16(src), factor)
		return n, nil
	}
	var scratch [scaleChunk]int16
	for start := 0; start < n; start += scaleChunk {
		end := min(start+scaleChunk, n)
		buf := scratch[:end-start]
		bytesToInt16LE(buf, src[2*start:2*end])
		scaleInt16To(buf, buf, factor)
		int16ToBytesLE(dst[2*start:2*end], buf)
	}
	return n, nil
}
