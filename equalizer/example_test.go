package equalizer_test

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/equalizer"
)

// Response reports the equalizer's magnitude at a set of frequencies, for
// drawing a response curve. A peaking filter has its full gain at the center.
func ExampleEqualizer_Response() {
	e, err := equalizer.New(equalizer.Config{
		SampleRate: 48000,
		Bands: []equalizer.Band{
			{Type: equalizer.Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6},
		},
	})
	if err != nil {
		panic(err)
	}
	pts := e.Response([]float64{1000})
	fmt.Printf("%.1f dB at %.0f Hz\n", pts[0].GainDB, pts[0].Hz)
	// Output: 6.0 dB at 1000 Hz
}

// LogSweepWithMarks adds each filter's own frequency to the log grid, so a
// narrow notch is sampled at its centre instead of missed between points. The
// null reads the per-section floor, SectionFloorDB.
func ExampleLogSweepWithMarks() {
	f, err := equalizer.NewFilter(equalizer.Band{Type: equalizer.BandReject, Frequency: 1000, WidthHz: 1000.0 / 30}, 48000)
	if err != nil {
		panic(err)
	}
	chain := equalizer.NewFilterChain()
	if err := chain.AddFilter(f); err != nil {
		panic(err)
	}
	freqs, err := equalizer.LogSweepWithMarks(20, 20000, 4, chain.Marks())
	if err != nil {
		panic(err)
	}
	pts := chain.Response(freqs)
	deepest := pts[0]
	for _, p := range pts {
		if p.GainDB < deepest.GainDB {
			deepest = p
		}
	}
	fmt.Printf("%d points, deepest %.1f dB at %.0f Hz\n", len(freqs), deepest.GainDB, deepest.Hz)
	// Output: 5 points, deepest -200.0 dB at 1000 Hz
}

// A Butterworth band-pass reads about -3 dB on the band edges and
// 0 dB at the geometric centre; Response reads the realised attenuation at all
// three for a one-third-octave band at 1 kHz.
func ExampleNewButterworthBandPass() {
	lower, upper := 1000*math.Pow(10, -0.05), 1000*math.Pow(10, 0.05)
	f, err := equalizer.NewButterworthBandPass(lower, upper, 4, 48000)
	if err != nil {
		panic(err)
	}
	for _, p := range f.Response([]float64{lower, math.Sqrt(lower * upper), upper}) {
		fmt.Printf("%7.1f Hz: %.2f dB down\n", p.Hz, math.Abs(p.GainDB))
	}
	// Output:
	//   891.3 Hz: 3.01 dB down
	//  1000.0 Hz: 0.00 dB down
	//  1122.0 Hz: 3.01 dB down
}

// A FilterChain reports the combined response of its filters: a 10 Hz
// high-pass is 3 dB down at its corner and a +6 dB peak at 1 kHz reads 6 dB.
func ExampleFilterChain_Response() {
	chain := equalizer.NewFilterChain()
	for _, b := range []equalizer.Band{
		{Type: equalizer.HighPass, Frequency: 10, Q: 0.7071},
		{Type: equalizer.Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6},
	} {
		f, err := equalizer.NewFilter(b, 48000)
		if err != nil {
			panic(err)
		}
		if err := chain.AddFilter(f); err != nil {
			panic(err)
		}
	}
	freqs, err := equalizer.LogSweep(10, 1000, 3)
	if err != nil {
		panic(err)
	}
	for _, p := range chain.Response(freqs) {
		fmt.Printf("%6.0f Hz: %+.1f dB\n", p.Hz, p.GainDB)
	}
	// Output:
	//     10 Hz: -3.0 dB
	//    100 Hz: +0.0 dB
	//   1000 Hz: +6.0 dB
}
