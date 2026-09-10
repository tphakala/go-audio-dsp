package dsp

import "math"

// FactorFromDB converts a decibel gain to the linear amplitude multiplier it
// represents: 10^(gainDB/20). A gain of 0 dB returns exactly 1. The gain and
// loudnorm blocks share it so their applied broadband gain uses one formula.
func FactorFromDB(gainDB float64) float64 {
	if gainDB == 0 {
		return 1
	}
	return math.Pow(10, gainDB/20)
}
