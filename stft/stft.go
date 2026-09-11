package stft

import (
	"fmt"

	"github.com/tphakala/simd/f32"
)

// PadMode selects the framing/centering convention of the whole-clip helpers,
// mirroring simd's f32.PadMode.
type PadMode int

const (
	// NoPad frames without padding: frame f is signal[f*hop : f*hop+FrameSize],
	// matching librosa stft(center=False). This is also the streaming convention.
	NoPad PadMode = iota
	// PadZero centers each frame with FrameSize/2 zero padding per side
	// (librosa center=True, pad_mode="constant").
	PadZero
	// PadReflect centers each frame with FrameSize/2 reflect padding per side
	// (numpy "reflect" semantics; librosa's pre-0.8.0 default).
	PadReflect
)

// simd maps a PadMode onto the simd framing enum with an explicit case per mode,
// so the mapping lives in one place; any value other than the three defined modes
// falls back to NoPad.
func (m PadMode) simd() f32.PadMode {
	switch m {
	case PadZero:
		return f32.PadZero
	case PadReflect:
		return f32.PadReflect
	default:
		return f32.NoPad
	}
}

// Plan is a whole-clip short-time transform for a fixed Config: a resident simd
// plan plus the resolved analysis window. Reuse one across many Spectrum/PowerInto
// calls to stay allocation-free. Not safe for concurrent use.
type Plan struct {
	p      *f32.STFTPlan
	window []float32
	n, hop int
}

// New builds a whole-clip Plan for cfg. It validates cfg and returns an error
// wrapping ErrInvalidConfig for the first invalid field.
func New(cfg Config) (*Plan, error) {
	rc, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	p, err := f32.NewSTFTPlan(rc.FrameSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidConfig, err)
	}
	return &Plan{p: p, window: rc.buildWindow(), n: rc.FrameSize, hop: rc.HopSize}, nil
}

// FrameSize returns the transform size in samples.
func (p *Plan) FrameSize() int { return p.n }

// HopSize returns the frame advance in samples.
func (p *Plan) HopSize() int { return p.hop }

// NumBins returns the number of output bins per frame, FrameSize/2 + 1 (the
// Hermitian half-spectrum, DC through Nyquist).
func (p *Plan) NumBins() int { return p.p.NumBins() }

// Window returns the resolved analysis window. Do not mutate it; it backs every
// transform this Plan runs.
func (p *Plan) Window() []float32 { return p.window }

// NumFrames reports how many frames Spectrum or PowerInto will write for a signal
// of signalLen samples under pad, so a caller can size dst exactly. For NoPad it
// is 1 + (signalLen-FrameSize)/HopSize (0 when shorter than a frame); for the
// centered modes it is 1 + signalLen/HopSize (0 when signalLen <= 0).
func (p *Plan) NumFrames(signalLen int, pad PadMode) int {
	return p.p.NumFrames(signalLen, p.hop, pad.simd())
}

// Spectrum writes one Hermitian half-spectrum (NumBins complex64 values) per
// frame of signal into dst under pad, and returns the number of frames written
// (min of len(dst) and NumFrames). Per frame it writes min(len(dst[f]), NumBins)
// bins. Allocation-free; dst rows must not overlap signal. The output is
// tolerance-stable, not bit-stable, across build targets and row lengths: compare
// to a tolerance, not bit for bit. The DC and Nyquist bins are exactly real.
func (p *Plan) Spectrum(dst [][]complex64, signal []float32, pad PadMode) int {
	return p.p.STFT(dst, signal, p.window, p.hop, pad.simd())
}

// PowerInto writes the frame-contiguous power spectrum |X|^2 of signal into the
// flat buffer dst under pad: frame f occupies dst[f*NumBins : (f+1)*NumBins],
// ready to pass as the vector argument to a mel-filterbank dot-product batch. It
// writes min(NumFrames, len(dst)/NumBins) whole frames and returns that count.
// Allocation-free; dst must not overlap signal. Tolerance-stable like Spectrum.
func (p *Plan) PowerInto(dst, signal []float32, pad PadMode) int {
	return p.p.STFTPowerInto(dst, signal, p.window, p.hop, pad.simd())
}
