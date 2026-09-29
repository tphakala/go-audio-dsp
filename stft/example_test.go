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

// Streaming resynthesis: analyze, leave the spectrum unchanged (a real consumer
// would apply a per-bin gain here), and overlap-add it back. The first
// FrameSize/HopSize-1 blocks are leading zeros and are discarded.
func ExampleSynthesizer() {
	const n, hop = 256, 64
	an, err := stft.NewAnalyzer(stft.Config{FrameSize: n, HopSize: hop})
	if err != nil {
		panic(err)
	}
	syn, err := stft.NewSynthesizer(an)
	if err != nil {
		panic(err)
	}
	in := make([]float32, 0, 1024+n)
	for i := range 1024 {
		in = append(in, float32(i%50)/50)
	}
	an.Feed(make([]float32, n-hop), func([]complex64, []float32) {}) // preroll: no frame completes
	out := make([]float32, 1024+n)
	written, frame := 0, 0
	an.Feed(append(in[:1024:1024], make([]float32, n)...), func(spec []complex64, _ []float32) {
		syn.Add(spec)
		if frame >= n/hop-1 {
			written += syn.Finish(out[written:])
		} else {
			syn.Discard()
		}
		frame++
	})
	var maxErr float32
	for i := range in {
		maxErr = max(maxErr, max(out[i]-in[i], in[i]-out[i]))
	}
	fmt.Println("reconstructed:", written >= len(in), maxErr < 1e-4)
	// Output: reconstructed: true true
}
