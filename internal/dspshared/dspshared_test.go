package dspshared

import (
	"errors"
	"math"
	"strings"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/stft"
)

const hopField = "HopSize"

func TestCopyFloor(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	src := []float32{0, -1, nan, inf, EpsPower, 2 * EpsPower, 0.5}
	want := []float32{EpsPower, EpsPower, EpsPower, EpsPower, EpsPower, 2 * EpsPower, 0.5}
	dst := make([]float32, len(src))
	CopyFloor(dst, src)
	for k := range want {
		if dst[k] != want[k] {
			t.Errorf("CopyFloor[%d] = %g, want %g", k, dst[k], want[k])
		}
	}
	CopyFloor(src, src) // in place
	for k := range want {
		if src[k] != want[k] {
			t.Errorf("in-place CopyFloor[%d] = %g, want %g", k, src[k], want[k])
		}
	}
}

func TestTrackWindowFrames(t *testing.T) {
	if f := TrackWindowFrames(2, 48000, 256); f != 375 {
		t.Errorf("TrackWindowFrames(2 s) = %d, want 375", f)
	}
	if f := TrackWindowFrames(0.001, 48000, 256); f != 1 {
		t.Errorf("TrackWindowFrames(tiny) = %d, want 1", f)
	}
}

func TestAutoFrameSize(t *testing.T) {
	cases := map[int]int{8000: 128, 16000: 256, 22050: 512, 32000: 512, 44100: 1024, 48000: 1024, 96000: 2048, 100: MinFrameSize}
	for sr, want := range cases {
		if got := AutoFrameSize(sr); got != want {
			t.Errorf("AutoFrameSize(%d) = %d, want %d", sr, got, want)
		}
	}
}

func TestResolveFrame(t *testing.T) {
	f, h, err := ResolveFrame(48000, 0, 0)
	if err != nil || f != 1024 || h != 256 {
		t.Fatalf("defaults = (%d, %d, %v), want (1024, 256, nil)", f, h, err)
	}
	f, h, err = ResolveFrame(16000, 512, 128)
	if err != nil || f != 512 || h != 128 {
		t.Fatalf("explicit = (%d, %d, %v), want (512, 128, nil)", f, h, err)
	}
	bad := []struct {
		name          string
		sr, frame, hp int
		msg           string
	}{
		{"rate", 0, 0, 0, "SampleRate"},
		{"frame not pow2", 48000, 1000, 0, "FrameSize"},
		{"frame too small", 48000, 32, 0, "FrameSize"},
		{"hop equals frame", 48000, 1024, 1024, hopField},
		{"hop not divisor", 48000, 1024, 300, hopField},
		{"hop negative", 48000, 1024, -4, hopField},
	}
	for _, c := range bad {
		_, _, err := ResolveFrame(c.sr, c.frame, c.hp)
		if !errors.Is(err, dsp.ErrInvalidConfig) || !strings.Contains(err.Error(), c.msg) {
			t.Errorf("%s: err = %v, want ErrInvalidConfig mentioning %s", c.name, err, c.msg)
		}
		if errors.Unwrap(err) != dsp.ErrInvalidConfig { //nolint:errorlint // pins the single-%w chain, not just errors.Is
			t.Errorf("%s: err does not unwrap directly to ErrInvalidConfig", c.name)
		}
	}
}

// pendingRef counts the emitted output by simulating the analyzer fill loop
// sample by sample, the definition PendingOutput must match in closed form.
func pendingRef(inFill, extra, n, hop int, frames, warm int64) int {
	fill, out := inFill, 0
	for range extra {
		fill++
		if fill == n {
			if frames >= warm {
				out += hop
			}
			frames++
			fill -= hop
		}
	}
	return out
}

func TestPendingOutput(t *testing.T) {
	const n, hop = 256, 64
	for _, warm := range []int64{0, 3, 5} {
		for _, frames := range []int64{0, 1, 4, 9} {
			for inFill := 0; inFill < n; inFill += 37 {
				for extra := 0; extra <= 3*n; extra += 23 {
					want := pendingRef(inFill, extra, n, hop, frames, warm)
					if got := PendingOutput(inFill, extra, n, hop, frames, warm); got != want {
						t.Fatalf("warm=%d frames=%d inFill=%d extra=%d: got %d, want %d", warm, frames, inFill, extra, got, want)
					}
				}
			}
		}
	}
}

func TestSTFTConfig(t *testing.T) {
	c := STFTConfig(512, 128)
	if c.FrameSize != 512 || c.HopSize != 128 || c.Window != stft.Hann {
		t.Errorf("STFTConfig = %+v", c)
	}
}

func TestNoEmit(t *testing.T) {
	NoEmit(nil, nil) // must be callable with any arguments and do nothing
}
