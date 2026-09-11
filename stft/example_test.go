package stft_test

import (
	"fmt"

	"github.com/tphakala/go-audio-dsp/stft"
)

// Whole-clip: compute a power spectrogram as a frame-contiguous matrix.
func ExamplePlan_PowerInto() {
	p, err := stft.New(stft.Config{FrameSize: 1024, HopSize: 256})
	if err != nil {
		panic(err)
	}
	signal := make([]float32, 48000) // one second at 48 kHz
	frames := p.NumFrames(len(signal), stft.NoPad)
	power := make([]float32, frames*p.NumBins())
	got := p.PowerInto(power, signal, stft.NoPad)
	fmt.Printf("%d frames x %d bins\n", got, p.NumBins())
	// Output: 184 frames x 513 bins
}

// Streaming: feed arbitrary chunks and act on each completed frame's spectrum.
func ExampleAnalyzer() {
	a, err := stft.NewAnalyzer(stft.Config{FrameSize: 512, HopSize: 128})
	if err != nil {
		panic(err)
	}
	synth := make([]float32, a.FrameSize())
	frames := 0
	chunk := make([]float32, 300) // any chunk size works
	for range 20 {
		a.Feed(chunk, func(spec []complex64, power []float32) {
			// ... inspect power, or modify spec and resynthesize ...
			a.Inverse(synth, spec)
			frames++
		})
	}
	fmt.Printf("processed %d frames\n", frames)
	// Output: processed 43 frames
}
