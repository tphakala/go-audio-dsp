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
// scratch buffer. BytesToInt16 is the read-only decoder: it fills a caller's
// []int16 without ever writing the source bytes, so a reader can walk a large
// byte buffer through a small scratch on any host.
//
// The int16<->float32 scaling is 1/32768 forward and 32768 back. int16 is
// exactly representable in float32 and 32768 is a power of two, so the pair is a
// lossless round-trip for every int16 value; the reverse direction rounds to
// nearest (ties to even) and saturates so a full-scale sample never wraps. This
// matches the measurement convention in the loudnorm package.
//
// ScaleInt16 applies a gain to int16 samples in place with that same
// round-to-even, saturating convention, for callers that scale on the int16
// transport side without converting to float32. ScaleBytes is the byte form: it
// scales from one []byte into another (or into itself), so a caller can gain a
// chunk of a read-only source into a reusable scratch with no extra copy. Both
// allocate nothing.
//
// DownmixBytesToFloat32 and DownmixFloat32 fold interleaved multi-channel audio
// to mono at the same boundary. They average the channels with equal weight, so
// one channel of a stereo pair at amplitude x comes out at x/2 (as with sox
// "channels 1"), and the int16 form scales like BytesToFloat32: with one
// channel the result is bit-identical to it. A channel count below 1 wraps
// ErrInvalidConfig, and input that is not a whole number of frames returns
// ErrPartialFrame (or ErrOddByteLength for an odd byte count). The byte form
// keeps the zero-copy read on hosts that allow it and the byte-wise read
// elsewhere; both allocate nothing.
package pcm
