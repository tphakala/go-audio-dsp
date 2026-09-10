package loudnorm

import "github.com/tphakala/go-audio-dsp/pcm"

// MeasureBytes measures interleaved little-endian int16 PCM carried as bytes,
// the transport form of the signal. len(b) must be even (two bytes per sample);
// otherwise ErrOddByteLength is returned. On little-endian hosts the bytes are
// read as int16 with no copy.
func MeasureBytes(b []byte, sampleRate, channels int) (Measurement, error) {
	if len(b)%2 != 0 {
		return Measurement{}, ErrOddByteLength
	}
	if len(b) == 0 {
		// InPlaceInt16 skips the closure on empty input, so validate and
		// measure directly to stay identical to MeasureInt16 (which rejects a
		// bad sample rate or channel count and reports -Inf for no samples).
		return MeasureInt16(nil, sampleRate, channels)
	}
	var (
		meas Measurement
		err  error
	)
	pcm.InPlaceInt16(b, func(s []int16) {
		meas, err = MeasureInt16(s, sampleRate, channels)
	})
	return meas, err
}

// NormalizeBytes normalizes interleaved little-endian int16 PCM carried as bytes
// in place, writing the gained samples back into b, and returns the outcome.
// len(b) must be even; otherwise ErrOddByteLength is returned. On little-endian
// hosts the normalization is zero-copy: it needs no scratch buffer the size of
// the clip. Gain is applied with rounding and saturation so a boost never wraps;
// silent input is left untouched (GainDB = 0).
func NormalizeBytes(b []byte, opts Options) (Result, error) {
	if len(b)%2 != 0 {
		return Result{}, ErrOddByteLength
	}
	if len(b) == 0 {
		// InPlaceInt16 skips the closure on empty input, so validate and
		// normalize directly to stay identical to NormalizeInt16 (which rejects
		// invalid options and is a no-op for no samples).
		return NormalizeInt16(nil, opts)
	}
	var (
		res Result
		err error
	)
	pcm.InPlaceInt16(b, func(s []int16) {
		res, err = NormalizeInt16(s, opts)
	})
	return res, err
}
