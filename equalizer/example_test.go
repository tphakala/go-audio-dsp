package equalizer_test

import (
	"fmt"

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
