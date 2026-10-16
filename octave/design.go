package octave

import "fmt"

// FilterDesign selects how each band's band-pass filter is designed. The zero
// value is FilterBiquad, the design of earlier releases.
type FilterDesign int

const (
	// FilterBiquad is the default: Config.Order identical RBJ band-pass biquad
	// sections per band, each with its -3 dB points at the band edges.
	FilterBiquad FilterDesign = iota
	// FilterButterworth is an eight-pole Butterworth band-pass per band with its
	// -3 dB points exactly on the band edges and 0 dB at the band centre; see the
	// package documentation for what it meets and what it costs.
	FilterButterworth
)

// String returns the design name.
func (d FilterDesign) String() string {
	switch d {
	case FilterBiquad:
		return "biquad"
	case FilterButterworth:
		return "butterworth"
	}
	return fmt.Sprintf("FilterDesign(%d)", int(d))
}

const (
	// butterworthOrder is the prototype order of the FilterButterworth band-pass:
	// 8 poles, four biquad sections per band. The relative attenuation of every
	// band whose upper edge is below butterworthMaxMargin of Nyquist is within the
	// class 1 template of IEC 61260:1995 at order 4. A sweep of the upper edge at
	// 48 kHz with the check in template_test.go found order 3 leaving class 1
	// above 0.464 of Nyquist (the 10 to 20 kHz bands at 48 kHz) and order 4 above
	// 0.968.
	butterworthOrder = 4

	// butterworthMaxMargin is the largest NyquistMargin FilterButterworth
	// accepts. Above it a top band's upper edge can sit where the order 4 design
	// leaves the class 1 template (0.24 dB outside at 0.975 of Nyquist, 0.86 dB
	// at 0.99), and the documented guarantee would no longer hold for every band
	// the meter includes.
	butterworthMaxMargin = 0.95
)

// validateDesign checks the Config fields that depend on the filter design. The
// margin is the effective NyquistMargin, the default applied.
func validateDesign(cfg Config, margin float64) error {
	switch cfg.Filter {
	case FilterBiquad:
		return nil
	case FilterButterworth:
		if cfg.Order != 0 {
			return fmt.Errorf("%w: Order must be 0 with FilterButterworth, got %d", ErrInvalidConfig, cfg.Order)
		}
		if margin > butterworthMaxMargin {
			return fmt.Errorf("%w: NyquistMargin must be <= %g with FilterButterworth, got %g", ErrInvalidConfig, butterworthMaxMargin, cfg.NyquistMargin)
		}
		return nil
	}
	return fmt.Errorf("%w: unknown Filter %d", ErrInvalidConfig, int(cfg.Filter))
}
