// Package pcm converts between the 16-bit integer PCM transport form and the
// float32 processing form used by the DSP blocks in go-audio-dsp.
//
// Audio arrives and leaves as interleaved little-endian int16, often carried as
// []byte on a transport; the blocks process float32 in [-1, 1]. The conversions
// write into caller-provided buffers, so they allocate nothing; on little-endian
// hosts (amd64, arm64 and the rest of Go's SIMD targets) a []byte of int16 PCM
// and a []int16 share one memory layout, so byte conversion also reinterprets
// the backing array with no copy and hands it to the SIMD int16<->float32
// kernels from github.com/tphakala/simd. Big-endian hosts take a scalar
// byte-swap path with identical numeric results; there InPlaceInt16 uses a
// scratch buffer.
//
// The int16<->float32 scaling is 1/32768 forward and 32768 back. int16 is
// exactly representable in float32 and 32768 is a power of two, so the pair is a
// lossless round-trip for every int16 value; the reverse direction rounds to
// nearest (ties to even) and saturates so a full-scale sample never wraps. This
// matches the measurement convention in the loudnorm package.
//
// ScaleInt16 applies a gain to int16 samples in place with that same
// round-to-even, saturating convention, for callers that scale on the int16
// transport side without converting to float32.
package pcm
