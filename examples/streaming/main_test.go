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
