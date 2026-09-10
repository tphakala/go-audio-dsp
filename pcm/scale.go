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
	if factor == 1 || len(s) == 0 {
		return
	}
	var scratch [scaleChunk]float32
	for start := 0; start < len(s); start += scaleChunk {
		end := min(start+scaleChunk, len(s))
		seg := s[start:end]
		buf := scratch[:len(seg)]
		// buf = seg * factor in int16-magnitude units, then round-to-even and
		// saturate back into seg. Float32ToInt16Scale clamps to [-32768, 32767].
		simdf32.Int16ToFloat32Scale(buf, seg, factor)
		simdf32.Float32ToInt16Scale(seg, buf, 1.0)
	}
}
