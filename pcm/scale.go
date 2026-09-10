package pcm

import simdf32 "github.com/tphakala/simd/f32"

// scaleChunk bounds the stack scratch buffer used by ScaleInt16 so its memory
// stays O(1) regardless of input length.
const scaleChunk = 8192

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
