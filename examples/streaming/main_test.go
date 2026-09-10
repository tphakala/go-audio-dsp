package main

import "testing"

// TestRunChainPreservesLength checks the chain returns exactly as many PCM bytes
// as it was given: the denoiser is drained with FlushInto and the equalizer and
// gain are length-preserving. A missing flush would drop the denoiser's tail.
func TestRunChainPreservesLength(t *testing.T) {
	in := synthClip()
	for _, chunk := range []int{1, 480, 4800, 7000} {
		out, err := RunChain(in, chunk)
		if err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		if len(out) != len(in) {
			t.Errorf("chunk %d: out %d bytes, in %d bytes", chunk, len(out), len(in))
		}
	}
}

// TestRunChainRejectsNonPositiveChunk checks the chunk-size guard: a zero or
// negative chunk returns an error rather than hanging or panicking.
func TestRunChainRejectsNonPositiveChunk(t *testing.T) {
	in := synthClip()
	for _, chunk := range []int{0, -1} {
		if _, err := RunChain(in, chunk); err == nil {
			t.Errorf("RunChain(chunk=%d) returned nil error, want a rejection", chunk)
		}
	}
}

// TestRunChainRejectsOddLength checks that input that is not whole int16 samples
// is rejected rather than silently dropping the trailing byte.
func TestRunChainRejectsOddLength(t *testing.T) {
	if _, err := RunChain([]byte{1, 2, 3}, 4800); err == nil {
		t.Error("RunChain(odd-length input) returned nil error, want a rejection")
	}
}
