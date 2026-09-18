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
	// equal the effective WindowLength (FrameSize when WindowLength is 0). It is
	// copied, so the caller may reuse or mutate the slice afterwards; a window
	// shorter than FrameSize is zero-padded per WindowAlign like a built-in one.
	CustomWindow []float32
	// WindowLength is the analysis window length in samples, in [1, FrameSize]. 0
	// selects FrameSize. A window shorter than FrameSize is zero-padded to FrameSize
	// according to WindowAlign, so the transform size, NumBins and the framing grid
	// stay FrameSize (librosa win_length < n_fft). The window returned by Window()
	// is always FrameSize long. WindowLength and WindowAlign are kept last in the
	// struct so that adding them does not shift the earlier fields for a positional
	// literal.
	WindowLength int
	// WindowAlign places a window shorter than FrameSize inside the frame. The zero
	// value is AlignCenter. It has no effect on placement when the effective
	// WindowLength equals FrameSize (the window already fills the frame), though the
	// value is still validated.
	WindowAlign WindowAlign
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
	if c.WindowLength == 0 {
		c.WindowLength = c.FrameSize
	}
	if c.WindowLength < 1 || c.WindowLength > c.FrameSize {
		return c, fmt.Errorf("%w: WindowLength must be in [1, FrameSize=%d], got %d", ErrInvalidConfig, c.FrameSize, c.WindowLength)
	}
	if !c.WindowAlign.valid() {
		return c, fmt.Errorf("%w: unknown WindowAlign %d", ErrInvalidConfig, int(c.WindowAlign))
	}
	if c.CustomWindow != nil {
		if len(c.CustomWindow) != c.WindowLength {
			return c, fmt.Errorf("%w: CustomWindow length %d must equal WindowLength %d", ErrInvalidConfig, len(c.CustomWindow), c.WindowLength)
		}
	} else if !c.Window.valid() {
		return c, fmt.Errorf("%w: unknown Window %d", ErrInvalidConfig, int(c.Window))
	}
	return c, nil
}

// buildWindow returns the resolved FrameSize-long analysis window: a copy of
// CustomWindow, or a freshly generated built-in window, each of length
// WindowLength. When WindowLength is shorter than FrameSize the window is
// zero-padded to FrameSize at the WindowAlign offset; a short window inside an
// n-point frame is identical to an n-point window that is zero outside the
// placement (x[k]*w_padded[k]). Callers must pass a resolved Config, so
// WindowLength is already set and CustomWindow (if any) already has that length.
func (c Config) buildWindow() []float32 {
	var w []float32
	if c.CustomWindow != nil {
		w = slices.Clone(c.CustomWindow)
	} else {
		w = GenerateWindow(c.Window, c.WindowLength)
	}
	if len(w) >= c.FrameSize {
		return w
	}
	// Embed the short window into a zeroed FrameSize frame. AlignCenter floors the
	// leading pad (librosa util.pad_center); AlignLeft starts at offset 0. This
	// maintains the invariant that the window handed to simd is exactly FrameSize
	// long. The switch carries no default, so a future WindowAlign value trips the
	// exhaustive linter here, the one site where alignment drives behavior.
	framed := make([]float32, c.FrameSize)
	off := 0
	switch c.WindowAlign {
	case AlignCenter:
		off = (c.FrameSize - len(w)) / 2
	case AlignLeft:
		// Window at the frame start; off stays 0.
	}
	copy(framed[off:], w)
	return framed
}
