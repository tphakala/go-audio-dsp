package loudnorm

import (
	"errors"

	"github.com/tphakala/go-audio-dsp/pcm"
)

// Sentinel errors reported by the package. Returned validation errors wrap one
// of these with a descriptive detail and can be tested with errors.Is.
var (
	// ErrSampleRateTooLow reports a sample rate below the 8000 Hz minimum, below
	// which the BS.1770 K-weighting is undefined.
	ErrSampleRateTooLow = errors.New("loudnorm: sample rate too low")
	// ErrInvalidChannels reports a channel count that is not positive.
	ErrInvalidChannels = errors.New("loudnorm: invalid channel count")
	// ErrLengthNotMultiple reports an interleaved sample count that is not a
	// multiple of the channel count.
	ErrLengthNotMultiple = errors.New("loudnorm: sample count not a multiple of channel count")
	// ErrTargetOutOfRange reports a target loudness that is not finite or lies
	// outside the open interval (absolute gate, 0) LUFS.
	ErrTargetOutOfRange = errors.New("loudnorm: target loudness out of range")
	// ErrCeilingInvalid reports a true-peak ceiling that is not finite or is
	// above 0 dBTP.
	ErrCeilingInvalid = errors.New("loudnorm: invalid true-peak ceiling")
	// ErrOddByteLength reports a byte buffer whose length is not a multiple of
	// two, so it cannot hold whole little-endian int16 samples. It is the shared
	// pcm.ErrOddByteLength so a caller chaining blocks can test every block's
	// odd-length error uniformly with errors.Is.
	ErrOddByteLength = pcm.ErrOddByteLength
)
