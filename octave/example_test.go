package octave_test

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/octave"
)

// A one-second meter at 48 kHz fed a full scale 1 kHz sine: the 1 kHz band reads
// -3.01 dB, and every band is reported in the order of Bands.
func Example() {
	const rate = 48000
	m, err := octave.New(octave.Config{SampleRate: rate, PeriodSamples: rate})
	if err != nil {
		panic(err)
	}

	sine := make([]float32, 2*rate)
	for i := range sine {
		sine[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / rate))
	}

	period := 0
	m.Process(sine, func(meanSquare []float64) {
		period++
		if period != 2 { // the first period includes the start-up transient
			return
		}
		for i, b := range m.Bands() {
			if b.Nominal == 1000 {
				fmt.Printf("%.0f Hz band: %.2f dB\n", b.Nominal, octave.DB(meanSquare[i]))
			}
		}
	})
	fmt.Println("bands:", m.NumBands())
	// Output:
	// 1000 Hz band: -3.01 dB
	// bands: 30
}

// The Butterworth profile has its -3 dB points exactly on the band edges and
// reads the same -3.01 dB for a centre sine; it is opt-in through Config.Filter.
func Example_butterworth() {
	const rate = 48000
	m, err := octave.New(octave.Config{SampleRate: rate, PeriodSamples: rate, Filter: octave.FilterButterworth})
	if err != nil {
		panic(err)
	}

	sine := make([]float32, 2*rate)
	for i := range sine {
		sine[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / rate))
	}

	period := 0
	m.Process(sine, func(meanSquare []float64) {
		period++
		if period != 2 { // the first period includes the start-up transient
			return
		}
		for i, b := range m.Bands() {
			if b.Nominal == 1000 {
				fmt.Printf("%.0f Hz band: %.2f dB\n", b.Nominal, octave.DB(meanSquare[i]))
			}
		}
	})
	fmt.Println("bands:", m.NumBands())
	// Output:
	// 1000 Hz band: -3.01 dB
	// bands: 30
}
