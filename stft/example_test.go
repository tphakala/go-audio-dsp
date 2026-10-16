package stft_test

import (
	"fmt"
	"math"

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

// Dolph-Chebyshev window: a parameterized window is generated once and passed as
// CustomWindow. At this length and attenuation the window peaks at the center sample.
func ExampleDolphChebyshev() {
	w, err := stft.DolphChebyshev(2048, 106.5)
	if err != nil {
		panic(err)
	}
	a, err := stft.NewAnalyzer(stft.Config{FrameSize: 2048, CustomWindow: w})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%d %.4f %d\n", len(w), w[len(w)/2], a.NumBins())
	// Output: 2048 1.0000 1025
}

// EdgeAnalyzer: the sox 14.4.2 frame schedule with edge-adapted windows. A
// Dolph-Chebyshev shape at 106.5 dB, hop 350 and 48000 samples gives 137 frames,
// and a full-scale cosine centred on bin N/8 reads 0 dBFS within 0.01 dB in a frame
// without padding.
func ExampleEdgeAnalyzer() {
	const n, hop, k0 = 1024, 350, 128
	a, err := stft.NewEdgeAnalyzer(stft.EdgeConfig{
		FrameSize: n,
		Shape:     stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: 106.5},
	})
	if err != nil {
		panic(err)
	}
	if err := a.Reset(hop); err != nil {
		panic(err)
	}
	signal := make([]float32, 48000)
	for i := range signal {
		signal[i] = float32(math.Cos(2 * math.Pi * k0 * float64(i) / n))
	}
	frames, mid := 0, 0.0
	onFrame := func(power []float32) {
		if frames == 60 {
			mid = 10 * math.Log10(float64(power[k0]))
		}
		frames++
	}
	a.Feed(signal, onFrame)
	a.Finish(onFrame)
	fmt.Printf("%d frames, expected %d, mid-stream level within 0.01 dB of 0 dBFS: %t\n",
		frames, a.NumFrames(int64(len(signal))), math.Abs(mid) < 0.01)
	// Output: 137 frames, expected 137, mid-stream level within 0.01 dB of 0 dBFS: true
}
