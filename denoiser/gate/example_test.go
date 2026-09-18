package gate_test

import (
	"fmt"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/denoiser/gate"
)

// ExampleDenoise gates a whole clip with the blind noise-floor estimator. The
// output is sample-aligned with the input and has the same length.
func ExampleDenoise() {
	const sampleRate = 48000
	clip := make([]float32, sampleRate/2) // 0.5 s (silence stands in for real audio)

	out, err := gate.Denoise(clip, gate.Config{SampleRate: sampleRate, Strength: denoiser.Light})
	if err != nil {
		fmt.Println("denoise:", err)
		return
	}
	fmt.Printf("in %d samples, out %d samples\n", len(clip), len(out))
	// Output: in 24000 samples, out 24000 samples
}

// ExampleGate_streaming shows the streaming API: learn a floor from a noise-only
// excerpt, then process chunks and flush the tail at end of stream.
func ExampleGate_streaming() {
	const sampleRate = 48000
	g, err := gate.New(gate.Config{SampleRate: sampleRate, Strength: denoiser.Medium})
	if err != nil {
		fmt.Println("new:", err)
		return
	}
	if err := g.LearnNoise(make([]float32, g.FrameSize()*4)); err != nil {
		fmt.Println("learn:", err)
		return
	}

	total := 0
	out := make([]float32, g.MaxOutputLen(4096))
	for range 10 { // ten 4096-sample chunks
		n, err := g.ProcessInto(make([]float32, 4096), out)
		if err != nil {
			fmt.Println("process:", err)
			return
		}
		total += n
	}
	tail := make([]float32, g.Latency()+g.HopSize())
	n, err := g.FlushInto(tail)
	if err != nil {
		fmt.Println("flush:", err)
		return
	}
	total += n
	fmt.Printf("processed %d samples, output %d samples\n", 10*4096, total)
	// Output: processed 40960 samples, output 40960 samples
}
