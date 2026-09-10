package dsp

import "errors"

// ErrBufferTooSmall reports that a ProcessInto or FlushInto output slice was
// shorter than the samples the call would emit. The call consumes nothing and
// leaves the block unchanged, so the caller can retry with a larger buffer.
// Blocks report an undersized output buffer with this one shared sentinel, so a
// consumer that chains blocks can test for it uniformly with errors.Is.
var ErrBufferTooSmall = errors.New("dsp: output buffer too small")
