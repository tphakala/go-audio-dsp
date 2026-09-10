package dsp

import (
	"errors"
	"testing"
)

// passThrough is a memoryless reference Processor: it copies input to output.
// It exercises the interface shape and the ErrBufferTooSmall retry contract.
type passThrough struct{}

func (passThrough) ProcessInto(in, out []float32) (int, error) {
	if len(out) < len(in) {
		return 0, ErrBufferTooSmall
	}
	return copy(out, in), nil
}
func (passThrough) MaxOutputLen(inputLen int) int { return inputLen }
func (passThrough) Latency() int                  { return 0 }
func (passThrough) Reset()                        {}

// Compile-time proof the reference block satisfies the contract.
var _ Processor = passThrough{}

// TestBufferContract checks that an undersized out yields ErrBufferTooSmall and
// consumes nothing, while a buffer sized with MaxOutputLen succeeds.
func TestBufferContract(t *testing.T) {
	var p Processor = passThrough{}
	in := []float32{1, 2, 3}

	if _, err := p.ProcessInto(in, make([]float32, 2)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("undersized out: err = %v, want ErrBufferTooSmall", err)
	}

	out := make([]float32, p.MaxOutputLen(len(in)))
	n, err := p.ProcessInto(in, out)
	if err != nil || n != len(in) {
		t.Fatalf("sized out: n=%d err=%v, want %d, nil", n, err, len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("out[%d] = %v, want %v", i, out[i], in[i])
		}
	}
}
