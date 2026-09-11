package stft

import (
	"fmt"
	"slices"
)

// Config configures a short-time analysis: the transform size, the frame advance,
// and the analysis window. It is shared by the whole-clip Plan (New) and the
// streaming Analyzer (NewAnalyzer). The framing convention (see PadMode) is not a
// Config field: it is a per-call argument to the whole-clip helpers, and the
// streaming Analyzer is always NoPad (a consumer that wants centered streaming
// feeds its own leading padding).
type Config struct {
	// FrameSize is the FFT length in samples, a power of two >= 2.
	FrameSize int
	// HopSize is the frame advance in samples, in [1, FrameSize]. 0 selects
	// FrameSize/4 (75% overlap), or 1 when FrameSize is 2. Unlike a resynthesizing
	// consumer, analysis does not require HopSize to divide FrameSize; overlap-add
	// reconstruction does, and a consumer that resynthesizes enforces that itself.
	HopSize int
	// Window selects the built-in analysis window shape. Ignored when CustomWindow
	// is non-nil. The zero value is Hann (periodic).
	Window Window
	// CustomWindow, when non-nil, is the exact analysis window and its length must
	// equal FrameSize. It is copied, so the caller may reuse or mutate the slice
	// afterwards.
	CustomWindow []float32
}

const defaultOverlap = 4 // hop = frame/4 when HopSize is 0

// resolve fills defaults and validates, returning the effective Config.
func (c Config) resolve() (Config, error) {
	if c.FrameSize < 2 || c.FrameSize&(c.FrameSize-1) != 0 {
		return c, fmt.Errorf("%w: FrameSize must be a power of two >= 2, got %d", ErrInvalidConfig, c.FrameSize)
	}
	if c.HopSize == 0 {
		c.HopSize = max(1, c.FrameSize/defaultOverlap)
	}
	if c.HopSize < 1 || c.HopSize > c.FrameSize {
		return c, fmt.Errorf("%w: HopSize must be in [1, FrameSize=%d], got %d", ErrInvalidConfig, c.FrameSize, c.HopSize)
	}
	if c.CustomWindow != nil {
		if len(c.CustomWindow) != c.FrameSize {
			return c, fmt.Errorf("%w: CustomWindow length %d must equal FrameSize %d", ErrInvalidConfig, len(c.CustomWindow), c.FrameSize)
		}
	} else if !c.Window.valid() {
		return c, fmt.Errorf("%w: unknown Window %d", ErrInvalidConfig, int(c.Window))
	}
	return c, nil
}

// buildWindow returns the resolved analysis window: a copy of CustomWindow, or a
// freshly generated built-in window.
func (c Config) buildWindow() []float32 {
	if c.CustomWindow != nil {
		return slices.Clone(c.CustomWindow)
	}
	return GenerateWindow(c.Window, c.FrameSize)
}
