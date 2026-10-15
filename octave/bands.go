package octave

import "math"

// Band is one 1/3-octave band of the ISO 266 / IEC 61260-1 base-10 series.
type Band struct {
	// Nominal is the preferred centre frequency in Hz used to label the band: 25,
	// 31.5, 40, 50, 63, 80, 100, ..., 1000, 1250, ..., 20000.
	Nominal float64
	// Exact is the base-10 centre frequency 1000 * 10^(n/10) in Hz that the
	// filter is tuned to.
	Exact float64
	// Lower and Upper are the band edges in Hz, Exact * 10^(-0.05) and
	// Exact * 10^(+0.05), a ratio of 10^0.1 (one third of an octave).
	Lower, Upper float64
}

// nominalMantissa holds the ten preferred 1/3-octave values of one decade, as
// the nominal frequency of band index 0 to 9 (1000 Hz, 1250 Hz, ..., 8000 Hz).
var nominalMantissa = [10]float64{1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000}

// Band indices n run over the base-10 series: n = 0 is 1000 Hz, n = -16 is 25 Hz
// and n = 13 is 20 kHz. The scan covers 1 Hz to about 1 MHz.
const (
	minBandIndex = -30
	maxBandIndex = 60
)

// bandAt returns the band with base-10 index n.
func bandAt(n int) Band {
	decade := n / 10
	if n%10 < 0 {
		decade-- // floor division
	}
	m := n - 10*decade // 0..9
	var nominal float64
	if decade < 0 {
		nominal = nominalMantissa[m] / math.Pow10(-decade) // exact: 10^k is exact for small k
	} else {
		nominal = nominalMantissa[m] * math.Pow10(decade)
	}
	exact := 1000 * math.Pow(10, float64(n)/10)
	return Band{
		Nominal: nominal,
		Exact:   exact,
		Lower:   exact * math.Pow(10, -0.05),
		Upper:   exact * math.Pow(10, 0.05),
	}
}

// selectBands returns the bands whose nominal frequency lies in [minFreq,
// maxFreq] and whose upper edge is below margin * nyquist, ascending.
func selectBands(minFreq, maxFreq, margin, nyquist float64) []Band {
	var bands []Band
	for n := minBandIndex; n <= maxBandIndex; n++ {
		b := bandAt(n)
		if b.Nominal < minFreq*(1-1e-9) || b.Nominal > maxFreq*(1+1e-9) {
			continue
		}
		if !(b.Upper < margin*nyquist) {
			continue
		}
		bands = append(bands, b)
	}
	return bands
}
