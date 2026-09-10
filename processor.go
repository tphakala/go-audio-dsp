package dsp

// Processor is the streaming contract every audio block satisfies: a stateful
// transform that consumes single-channel (mono) float32 PCM in arbitrary-size
// chunks and writes the samples that have become final into a caller-owned
// output buffer, allocating nothing in steady state. Output is aligned
// sample-for-sample with input and lags it by Latency samples.
//
// A Processor serves one stream at a time and is not safe for concurrent use;
// run one instance per route and reuse it for a new stream by calling Reset.
type Processor interface {
	// ProcessInto consumes in and writes the output samples that become final
	// into out, returning how many were written (possibly zero).
	//
	// out must have room for at least MaxOutputLen(len(in)) samples. If it is
	// shorter than the call would emit, ProcessInto returns ErrBufferTooSmall
	// and consumes nothing, leaving the block unchanged so the caller can retry
	// with a larger out.
	ProcessInto(in, out []float32) (n int, err error)

	// MaxOutputLen returns an output length that always holds ProcessInto's
	// output for an input of inputLen samples (an upper bound, not the exact
	// count). A caller sizing out with this value never sees ErrBufferTooSmall.
	MaxOutputLen(inputLen int) int

	// Latency is how many samples the output lags the input in steady state. A
	// memoryless block returns 0.
	Latency() int

	// Reset clears the stream state so the next ProcessInto starts a new stream.
	// Configuration is retained.
	Reset()
}

// Flusher is implemented by a Processor that carries a tail: samples still held
// in internal buffers after the last input, which a block with non-zero latency
// has and a memoryless block does not. A pipeline drains each Flusher at end of
// stream; blocks that do not implement it have nothing to drain.
type Flusher interface {
	// FlushInto ends the stream: it writes the remaining output so the total
	// output length equals the total input length, then resets the stream state
	// so the block can start a new stream. It returns ErrBufferTooSmall and does
	// nothing if out is too small.
	FlushInto(out []float32) (n int, err error)
}
