// Package dsp defines the streaming contract shared by the audio processing
// blocks in go-audio-dsp.
//
// A block is a Processor: a stateful transform that consumes single-channel
// (mono) float32 PCM in arbitrary-size chunks and writes finalized samples into
// a caller-owned output buffer. It is sample-rate agnostic (a block that needs
// a rate fixes it at construction; the contract itself carries none) and
// allocation-free in steady state: a block reuses its internal scratch across
// calls, and any scratch it sizes to the input grows to fit the largest chunk
// seen without shrinking back. Process interleaved multi-channel
// audio as one block per channel. Blocks whose output lags their input carry a
// tail and also implement Flusher to drain it at end of stream. Two shared
// sentinels, ErrBufferTooSmall and ErrInvalidConfig, signal an undersized
// output buffer and a configuration a block's constructor cannot honour.
//
// The contract lets a consumer chain blocks over reused buffers: size each
// stage's output with MaxOutputLen, feed one block's output into the next, and
// call FlushInto on each block that implements Flusher at end of stream. A block
// processes one stream at a time and is not safe for concurrent use; run one
// instance per route.
//
// Concrete blocks live in sub-packages: the denoiser, the equalizer and the gain
// block satisfy this contract. The loudnorm package measures and normalizes
// whole clips rather than streaming, so it does not implement Processor.
// Conversion between the int16 PCM transport form and the float32 processing
// form lives in the pcm sub-package.
//
// The hot loops build on github.com/tphakala/simd, so they take its AVX/NEON
// paths where available and fall back to portable Go elsewhere. There is no CGo.
package dsp
