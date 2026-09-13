package spectrogram

import "math"

// QuantizeColumn linearly maps the values in col to uint8 indices in dst over
// the range [lo, hi]: lo maps to 0, hi to 255, and values outside the range
// clamp to the ends. It writes min(len(dst), len(col)) entries and returns that
// count. This is the transport encoding a browser waterfall wants (integer bin
// indices over a fixed range rather than float dB); the colormap that turns an
// index into a pixel, and the axes and legend, stay in the consumer. For a DB
// column the natural range is the clamp window [GainDB-DynamicRangeDB, GainDB].
// hi must be > lo and neither bound may be NaN; otherwise nothing is written and
// it returns 0. A NaN element maps to 0.
//
// It is allocation-free and complements the scale stage so quantization has one
// definition in the library rather than being re-derived per consumer.
func QuantizeColumn(dst []uint8, col []float32, lo, hi float64) int {
	n := min(len(dst), len(col))
	if n == 0 || math.IsNaN(lo) || math.IsNaN(hi) || hi <= lo {
		return 0
	}
	scale := 255 / (hi - lo)
	for i := range n {
		v := (float64(col[i]) - lo) * scale
		switch {
		case math.IsNaN(v):
			dst[i] = 0
		case v <= 0:
			dst[i] = 0
		case v >= 255:
			dst[i] = 255
		default:
			dst[i] = uint8(math.Round(v))
		}
	}
	return n
}
