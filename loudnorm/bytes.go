package loudnorm

import (
	"context"

	"github.com/tphakala/go-audio-dsp/pcm"
)

// MeasureBytes measures interleaved little-endian int16 PCM carried as bytes,
// the transport form of the signal. len(b) must be even (two bytes per sample);
// otherwise ErrOddByteLength is returned. b is only read, never written, on
// every host: the samples are decoded chunk by chunk into a small scratch, so
// no copy of the clip is made and a concurrent reader of b is safe.
func MeasureBytes(b []byte, sampleRate, channels int) (Measurement, error) {
	if len(b)%2 != 0 {
		return Measurement{}, ErrOddByteLength
	}
	if err := validateDims(sampleRate, channels, len(b)/2); err != nil {
		return Measurement{}, err
	}
	m := acquireMeter(sampleRate, channels)
	defer releaseMeter(m)
	// A background context never cancels, so feedBytes cannot fail here.
	_ = feedBytes(context.Background(), m, b, channels, 1)
	return result(m), nil
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
