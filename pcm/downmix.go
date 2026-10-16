package pcm

import (
	"encoding/binary"
	"errors"
	"fmt"

	dsp "github.com/tphakala/go-audio-dsp"
)

// ErrPartialFrame reports interleaved input whose sample count is not a
// multiple of the channel count.
var ErrPartialFrame = errors.New("pcm: sample count is not a whole number of frames")

// ErrInvalidConfig is dsp.ErrInvalidConfig, returned (wrapped) for a channel
// count below 1.
var ErrInvalidConfig = dsp.ErrInvalidConfig

// DownmixBytesToFloat32 averages interleaved little-endian int16 frames of
// channels samples each into mono float32 in [-1, 1) with equal weight per
// channel, as sox "channels 1" does: one channel of a stereo pair at amplitude
// x gives x/2. Each output is float32(sum of the frame's samples) * (1/(32768*channels)),
// with the scale rounded once to float32. The frame sum is exact and the scale
// is a power of two for 1, 2 and 4 channels, so those results are exact;
// channels=1 equals BytesToFloat32 bit for bit.
//
// channels below 1 returns an error wrapping ErrInvalidConfig. An odd len(src)
// returns ErrOddByteLength, and a sample count that is not a multiple of
// channels returns ErrPartialFrame; both are checked against the whole of src
// before any frame is converted, and report a count of 0. Otherwise it converts
// min(len(dst), frames) frames and returns that count, so a shorter dst yields a
// partial downmix rather than an error. Little-endian hosts that permit
// unaligned access read src through the zero-copy int16 view; other hosts decode
// byte by byte with identical results. It allocates nothing. dst and src must
// not overlap.
func DownmixBytesToFloat32(dst []float32, src []byte, channels int) (int, error) {
	if err := checkChannels(channels); err != nil {
		return 0, err
	}
	if len(src)%2 != 0 {
		return 0, ErrOddByteLength
	}
	if (len(src)/2)%channels != 0 {
		return 0, ErrPartialFrame
	}
	if canReinterpretInt16 {
		return downmixInt16(dst, bytesAsInt16(src), channels), nil
	}
	return downmixBytesLE(dst, src, channels), nil
}

// DownmixFloat32 averages interleaved float32 frames of channels samples each
// into mono: each output is the sum of the frame's samples times 1/channels.
// The sum is carried in float64 and rounded to float32 once.
//
// channels below 1 returns an error wrapping ErrInvalidConfig and a
// len(src) that is not a multiple of channels returns ErrPartialFrame; both
// report a count of 0. Otherwise it converts min(len(dst), frames) frames and
// returns that count. It allocates nothing. dst and src must not overlap.
func DownmixFloat32(dst, src []float32, channels int) (int, error) {
	if err := checkChannels(channels); err != nil {
		return 0, err
	}
	if len(src)%channels != 0 {
		return 0, ErrPartialFrame
	}
	n := min(len(dst), len(src)/channels)
	inv := 1 / float64(channels)
	for i := range n {
		var sum float64
		for _, v := range src[i*channels : (i+1)*channels] {
			sum += float64(v)
		}
		dst[i] = float32(sum * inv)
	}
	return n, nil
}

// checkChannels rejects a channel count below 1 with an error wrapping
// ErrInvalidConfig.
func checkChannels(channels int) error {
	if channels < 1 {
		return fmt.Errorf("%w: channels must be >= 1, got %d", ErrInvalidConfig, channels)
	}
	return nil
}

// downmixScale is the per-frame multiplier of the int16 downmix: it folds the
// 1/32768 sample scale and the 1/channels average into one float32 constant, so
// channels=1 reproduces int16ToFloatScale exactly.
func downmixScale(channels int) float32 {
	return float32(1 / (32768 * float64(channels)))
}

// downmixInt16 averages interleaved int16 frames into dst and returns the frame
// count converted, min(len(dst), len(src)/channels). channels must be >= 1. The
// frame sum is held in int64, so no channel count overflows it.
func downmixInt16(dst []float32, src []int16, channels int) int {
	n := min(len(dst), len(src)/channels)
	scale := downmixScale(channels)
	for i := range n {
		var sum int64
		for _, v := range src[i*channels : (i+1)*channels] {
			sum += int64(v)
		}
		dst[i] = float32(sum) * scale
	}
	return n
}

// downmixBytesLE is downmixInt16 over little-endian bytes, correct on any host.
// It is the path taken where the zero-copy reinterpret is unsafe (big-endian
// hosts, the MIPS family) and the reference the fast path is tested against.
// len(src) must be even and channels >= 1.
func downmixBytesLE(dst []float32, src []byte, channels int) int {
	n := min(len(dst), len(src)/2/channels)
	scale := downmixScale(channels)
	for i := range n {
		var sum int64
		frame := src[2*i*channels : 2*(i+1)*channels]
		for c := range channels {
			sum += int64(int16(binary.LittleEndian.Uint16(frame[2*c:])))
		}
		dst[i] = float32(sum) * scale
	}
	return n
}
