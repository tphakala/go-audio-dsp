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

// A Butterworth band-pass has its -3 dB points exactly on the band edges and
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
