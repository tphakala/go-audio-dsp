package mel

import (
	"fmt"

	"github.com/tphakala/go-audio-dsp/stft"
)

// ExampleExtractor computes a whole-clip log-mel matrix and reports its shape.
func ExampleExtractor() {
	ex, err := New(Config{
		SampleRate: 16000, FrameSize: 512, HopSize: 128, NumMels: 40,
		MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	})
	if err != nil {
		panic(err)
	}
	sig := toneSig(4096, 16000, []float64{1000}, 0.8)
	m := ex.Compute(sig, stft.NoPad)
	fmt.Printf("%d mels x %d frames\n", m.Mels, m.Frames)
	// Output:
	// 40 mels x 29 frames
}

// ExampleColumnSource streams the same audio and counts the emitted columns.
func ExampleColumnSource() {
	cs, err := NewColumnSource(Config{
		SampleRate: 16000, FrameSize: 512, HopSize: 128, NumMels: 40,
		MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	})
	if err != nil {
		panic(err)
	}
	sig := toneSig(4096, 16000, []float64{1000}, 0.8)
	columns := 0
	cs.Feed(sig, func(col []float32, centerSample int64) {
		columns++
	})
	fmt.Printf("%d columns of %d mels\n", columns, cs.NumMels())
	// Output:
	// 29 columns of 40 mels
}
