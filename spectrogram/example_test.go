package spectrogram_test

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/spectrogram"
	"github.com/tphakala/go-audio-dsp/stft"
)

// Fit a 15 s clip at 24 kHz into at most 1026 columns of 513 bins, with a
// Dolph-Chebyshev window sized for a 100 dB display range. The values are
// unclamped decibels, so the renderer applies its own range and colormap.
func ExampleFit() {
	const rate = 24000
	f, err := spectrogram.NewFit(spectrogram.FitConfig{
		SampleRate: rate,
		FrameSize:  1024,
		Columns:    1026,
		Window: stft.WindowShape{
			Family:        stft.ShapeDolphChebyshev,
			AttenuationDB: spectrogram.DolphAttenuationDB(100, 0, 0),
		},
	})
	if err != nil {
		panic(err)
	}
	clip := make([]float32, 15*rate)
	for i := range clip {
		clip[i] = float32(0.5 * math.Sin(2*math.Pi*3000*float64(i)/rate))
	}
	m, err := f.Compute(clip, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println(f.Hop(), f.BlockSteps(), m.Frames, f.Bins())
	// Output: 350 1 1026 513
}
