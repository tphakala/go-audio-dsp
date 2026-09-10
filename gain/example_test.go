package gain_test

import (
	"fmt"

	"github.com/tphakala/go-audio-dsp/gain"
)

// ProcessInto scales a float32 chunk by a fixed decibel gain, in place or into a
// separate buffer.
func ExampleGain_ProcessInto() {
	g, err := gain.New(-20) // -20 dB = 0.1x
	if err != nil {
		panic(err)
	}
	in := []float32{0.5, -0.25}
	out := make([]float32, len(in))
	if _, err := g.ProcessInto(in, out); err != nil {
		panic(err)
	}
	fmt.Printf("%.3f %.3f\n", out[0], out[1])
	// Output: 0.050 -0.025
}

// ApplyInt16 scales interleaved int16 PCM in place, saturating instead of
// wrapping when a boost exceeds full scale.
func ExampleGain_ApplyInt16() {
	g, err := gain.New(20) // +20 dB = 10x
	if err != nil {
		panic(err)
	}
	s := []int16{100, -100, 20000}
	g.ApplyInt16(s)
	fmt.Println(s) // 20000*10 saturates to 32767
	// Output: [1000 -1000 32767]
}
