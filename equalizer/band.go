package equalizer

import "fmt"

// FilterType selects the biquad shape for a Band. The zero value is invalid on
// purpose, so an unset Band is rejected rather than silently becoming a filter.
type FilterType int

const (
	// LowPass passes frequencies below the corner and attenuates above it. Uses Q.
	LowPass FilterType = iota + 1
	// HighPass passes frequencies above the corner and attenuates below it. Uses Q.
	HighPass
	// AllPass leaves the magnitude flat and shifts phase around the corner. Uses Q.
	AllPass
	// BandPass passes a band around the center and attenuates away from it, with
	// a constant 0 dB peak gain (the cookbook's "constant 0 dB peak" variant).
	// Uses WidthHz.
	BandPass
	// BandReject attenuates a band around the center and passes elsewhere (the
	// cookbook's notch). Uses WidthHz.
	BandReject
	// LowShelf boosts or cuts frequencies below the corner by GainDB. Uses Q and GainDB.
	LowShelf
	// HighShelf boosts or cuts frequencies above the corner by GainDB. Uses Q and GainDB.
	HighShelf
	// Peaking boosts or cuts a band around the center by GainDB. Uses WidthHz and GainDB.
	Peaking
)

// String returns the filter type name.
func (t FilterType) String() string {
	switch t {
	case LowPass:
		return "low-pass"
	case HighPass:
		return "high-pass"
	case AllPass:
		return "all-pass"
	case BandPass:
		return "band-pass"
	case BandReject:
		return "band-reject"
	case LowShelf:
		return "low-shelf"
	case HighShelf:
		return "high-shelf"
	case Peaking:
		return "peaking"
	}
	return fmt.Sprintf("FilterType(%d)", int(t))
}

func (t FilterType) valid() bool { return t >= LowPass && t <= Peaking }

// usesWidth reports whether the type takes its bandwidth from WidthHz (in Hz)
// rather than from Q.
func (t FilterType) usesWidth() bool {
	return t == BandPass || t == BandReject || t == Peaking
}

// usesGain reports whether the type applies GainDB.
func (t FilterType) usesGain() bool {
	return t == Peaking || t == LowShelf || t == HighShelf
}

// maxPasses caps how many identical sections a single Band may cascade, so a
// stray large Passes cannot request an unbounded allocation.
const maxPasses = 16

// Band is one biquad stage of an equalizer.
//
// Frequency (Hz) is required for every type. The bandwidth comes from Q for
// LowPass, HighPass, AllPass, LowShelf and HighShelf, and from WidthHz (in Hz)
// for BandPass, BandReject and Peaking; the field the type does not use is
// ignored. GainDB applies to Peaking, LowShelf and HighShelf. Passes cascades
// that many identical sections for a steeper response (0 means 1).
type Band struct {
	Type      FilterType
	Frequency float64 // center or corner frequency in Hz, in (0, SampleRate/2)
	Q         float64 // quality factor (> 0); used by LowPass, HighPass, AllPass, LowShelf, HighShelf
	WidthHz   float64 // bandwidth in Hz (> 0); used by BandPass, BandReject, Peaking
	GainDB    float64 // gain in dB; used by Peaking, LowShelf, HighShelf
	Passes    int     // cascaded identical sections, 0 means 1, at most maxPasses
}

// Config configures an Equalizer. SampleRate is required; Bands may be empty, in
// which case the Equalizer is an identity (it copies input to output unchanged).
type Config struct {
	SampleRate int
	Bands      []Band
}
