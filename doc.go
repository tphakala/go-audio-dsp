// Package dsp defines the streaming contract shared by the audio processing
// blocks in go-audio-dsp.
//
// A block is a Processor: a stateful transform that consumes float32 PCM in
// arbitrary-size chunks and writes finalized samples into a caller-owned output
// buffer, allocating nothing in steady state. Blocks whose output lags their
// input carry a tail and also implement Flusher to drain it at end of stream.
// One shared sentinel, ErrBufferTooSmall, signals an undersized output buffer.
//
// The contract lets a consumer chain blocks over reused buffers: size each
// stage's output with MaxOutputLen, feed one block's output into the next, and
// call FlushInto on each block that implements Flusher at end of stream. A block
// processes one stream at a time and is not safe for concurrent use; run one
// instance per route.
//
// Concrete blocks live in sub-packages. The denoiser satisfies this contract
// today; further blocks (loudness normalization, biquad EQ, gain) adopt it as
// they land. Conversion between the int16 PCM transport form and the float32
// processing form lives in the pcm sub-package.
//
// The hot loops build on github.com/tphakala/simd, so they take its AVX/NEON
// paths where available and fall back to portable Go elsewhere. There is no CGo.
package dsp
