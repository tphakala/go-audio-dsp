package spectrogram

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/stft"

	simdf32 "github.com/tphakala/simd/f32"
)

// maxFitBlock is the most samples per column the geometry accepts, the largest
// value of a 32-bit int, so Hop and BlockSteps fit an int on every target.
const maxFitBlock = math.MaxInt32

// FitConfig configures a Fit.
type FitConfig struct {
	// SampleRate in Hz; required (> 0).
	SampleRate int
	// FrameSize is the transform length in samples, a power of two >= 2. A
	// renderer that draws a picture Height pixels tall uses 2*(Height-1), so the
	// rows (Bins) match its height. A renderer that draws only up to a frequency
	// below Nyquist can use a FrameSize m times larger and set MaxHz to that
	// frequency, so Bins still equals Height.
	FrameSize int
	// Columns is the column limit: Fit never emits more, and picks its step so a
	// span fills about this many columns. Required (>= 1).
	Columns int
	// Window is the analysis window family and its Dolph-Chebyshev attenuation.
	// The zero value is Hann.
	Window stft.WindowShape
	// GainDB is added to every value. It must be finite; BirdNET-Go uses 0.
	GainDB float64
	// MinHz and MaxHz select the output frequency range: rows cover the bins
	// whose centre frequency lies in [MinHz, MaxHz], with the same rounding and
	// validation as Config. MinHz 0 starts at DC. MaxHz 0, or any value at or
	// above Nyquist, means Nyquist; a negative MaxHz is rejected. MinHz must be
	// finite and below the effective MaxHz. Fit still transforms the full
	// FrameSize, so the range selects which rows are kept and does not change
	// their values, within the tolerance the package documents (a range of fewer
	// than 8 bins can differ by one ulp on a SIMD tier); the per-bin work outside
	// the range is skipped. The sox effect has no frequency range; this is an
	// extension.
	MinHz, MaxHz float64
}

// DolphAttenuationDB returns the Dolph-Chebyshev sidelobe attenuation to pair
// with a display range: (dynamicRangeDB + gainDB) * (1.005 + adjust/50) + 6,
// the rule the sox 14.4.2 spectrogram effect applies to its range setting.
// gainDB is FitConfig.GainDB and adjust is the effect's window adjustment (0 by
// default). A range of 80, 100 and 120 dB with no gain and no adjustment gives
// 86.4, 106.5 and 126.6 dB.
func DolphAttenuationDB(dynamicRangeDB, gainDB, adjust float64) float64 {
	return (dynamicRangeDB+gainDB)*(1.005+adjust/50) + 6
}

// Fit renders a clip into at most Columns spectrogram columns the way the sox
// 14.4.2 spectrogram effect does, sox being the behavioural reference. From the
// span the clip should cover it derives the frame step (Hop) and the number of
// frames averaged into one column (BlockSteps), runs the stft.EdgeAnalyzer frame
// schedule and edge-adapted windows, averages the power of each block of frames,
// and converts it to decibels. The first Columns columns are kept and nothing
// past them is produced.
//
// Values are decibels relative to a full-scale sinusoid on a bin centre (0 dB),
// unclamped, with a finite floor near -200 dB plus GainDB for silence. Rows are
// all FrameSize/2+1 bins, DC first, so a renderer drawing the picture top-down
// reads them from the last bin down. Clamping to a display range, colour mapping,
// axes and PNG encoding stay with the renderer, as for Spectrogram.
//
// This is an amplitude normalization, a different level from the energy
// normalization of Spectrogram (see the package documentation).
//
// A Fit runs one stream at a time: Reset starts it, Feed pushes samples, Finish
// ends it and emits the last, possibly shorter, block. Its edge windows come from
// the process-wide cache of stft.EdgeAnalyzer, so a new Fit for a configuration and
// span used before builds none of the windows it keeps while they remain in that
// cache (see stft.EdgeAnalyzer for eviction and the cap). Feed and Finish are
// allocation-free when the callback does not allocate; the only allocations are in
// construction and when a needed window is not in that cache. One Fit is not
// safe for concurrent use; separate Fits may run concurrently.
type Fit struct {
	an    *stft.EdgeAnalyzer
	sc    scaler
	cfg   FitConfig
	n     int
	acc   []float32 // sum of the power of the frames in the pending block; length hi, only [lo, hi) is used
	mean  []float32 // acc divided by the block's frame count; same layout
	col   []float32 // output column handed to the callback, length Bins
	ready bool      // Reset has succeeded at least once

	// Geometry from the last successful Reset.
	hop, block int
	// useful is the number of signal samples after which no kept column can
	// change: the last kept frame ends there.
	useful int64

	// Stream state.
	count    int   // frames in the pending block
	emitted  int   // columns emitted
	consumed int64 // signal samples accepted
	finished bool
}

// NewFit builds a Fit for cfg. It returns an error wrapping ErrInvalidConfig for
// the first invalid field. The new Fit has no geometry yet: call Reset before
// streaming, or use ComputeInto or Compute.
func NewFit(cfg FitConfig) (*Fit, error) {
	if cfg.SampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, cfg.SampleRate)
	}
	if cfg.Columns < 1 {
		return nil, fmt.Errorf("%w: Columns must be >= 1, got %d", ErrInvalidConfig, cfg.Columns)
	}
	if math.IsNaN(cfg.GainDB) || math.IsInf(cfg.GainDB, 0) {
		return nil, fmt.Errorf("%w: GainDB must be finite, got %g", ErrInvalidConfig, cfg.GainDB)
	}
	if err := validateHzRange(cfg.MinHz, cfg.MaxHz); err != nil {
		return nil, err
	}
	an, err := stft.NewEdgeAnalyzer(stft.EdgeConfig{FrameSize: cfg.FrameSize, Shape: cfg.Window})
	if err != nil {
		return nil, err
	}
	lo, hi, err := binRange(cfg.SampleRate, cfg.FrameSize, an.NumBins(), cfg.MinHz, cfg.MaxHz)
	if err != nil {
		return nil, err
	}
	if bins := hi - lo; int64(bins)*int64(cfg.Columns) > math.MaxInt32 {
		return nil, fmt.Errorf("%w: Columns %d with %d bins gives a matrix of more than %d values", ErrInvalidConfig, cfg.Columns, bins, math.MaxInt32)
	}
	sc := newDBScaler(lo, hi, float64(cfg.SampleRate)/float64(cfg.FrameSize), cfg.GainDB)
	return &Fit{
		an:   an,
		sc:   sc,
		cfg:  cfg,
		n:    cfg.FrameSize,
		acc:  make([]float32, hi),
		mean: make([]float32, hi),
		col:  make([]float32, hi-lo),
	}, nil
}

// FrameSize returns the transform size in samples.
func (f *Fit) FrameSize() int { return f.n }

// Bins returns the number of frequency rows in each column: the bins in
// [MinHz, MaxHz], FrameSize/2 + 1 by default.
func (f *Fit) Bins() int { return f.sc.bins() }

// BinHz returns the centre frequency in Hz of row bin. Row 0 is the lowest kept
// bin: DC, unless MinHz selects a higher one.
func (f *Fit) BinHz(bin int) float64 { return f.sc.binHzOf(bin) }

// Hop returns the frame step in samples chosen by the last successful Reset, or 0
// before the first one.
func (f *Fit) Hop() int { return f.hop }

// BlockSteps returns the number of frames averaged into each column, chosen by
// the last successful Reset, or 0 before the first one.
func (f *Fit) BlockSteps() int { return f.block }

// SecondsSpanned returns the time cols columns cover, cols*Hop*BlockSteps/SampleRate
// seconds, for the current geometry. A renderer labels its time axis with it.
func (f *Fit) SecondsSpanned(cols int) float64 {
	return float64(cols) * float64(f.hop) * float64(f.block) / float64(f.cfg.SampleRate)
}

// fitGeometry derives the step, the block size and the sample count after which
// no kept column can change, for a span of spanSamples samples. It changes
// nothing.
//
// The step rule: the columns per second needed to fit the span in Columns
// (capped at 5000) set the samples per column B0; the full window sum S0 is the
// natural step; the step is B0 divided into the fewest equal parts of at most S0
// samples, rounded, and the block is how many steps make up B0, rounded. The
// float64 operations run in this order.
func (f *Fit) fitGeometry(spanSamples int64) (hop, block int, useful int64, err error) {
	if spanSamples <= 0 {
		return 0, 0, 0, fmt.Errorf("%w: span must be > 0 samples, got %d", ErrInvalidConfig, spanSamples)
	}
	rate := float64(f.cfg.SampleRate)
	dur := float64(spanSamples) / rate
	pps := math.Min(5000, float64(f.cfg.Columns)/dur)
	b0 := math.Trunc(rate / pps)
	if !(b0 >= 1) {
		return 0, 0, 0, fmt.Errorf("%w: %d columns over %d samples at %d Hz need more columns per second than the sample rate", ErrInvalidConfig, f.cfg.Columns, spanSamples, f.cfg.SampleRate)
	}
	if b0 > maxFitBlock {
		return 0, 0, 0, fmt.Errorf("%w: %d columns over %d samples at %d Hz need more than %d samples per column", ErrInvalidConfig, f.cfg.Columns, spanSamples, f.cfg.SampleRate, maxFitBlock)
	}
	s0 := math.Max(1, math.Trunc(f.an.FullWindowSum()+0.5))
	hopF := math.Trunc(b0/math.Ceil(b0/s0) + 0.5)
	blockF := math.Floor(b0/hopF + 0.5)
	// b0 >= 1 and s0 >= 1 make hopF in [1, b0], so blockF >= 1 and both fit an int.
	// A step above FrameSize (a window sum above FrameSize + 0.5) is rejected by
	// the analyzer's Reset.
	hop, block = int(hopF), int(blockF)
	// The last kept frame is X*B-1. Its end in signal samples is N - Z0 + (X*B-1)*S
	// for Z0 = ceil((N-S)/2) lead zeros. The product stays far below int64: block*hop
	// is at most B0 plus half a hop, B0 is below 2^31, and Columns is below 2^31
	// (NewFit bounds Bins*Columns by MaxInt32 with Bins >= 1).
	z0 := (f.n - hop + 1) / 2
	useful = int64(f.n-z0) + (int64(f.cfg.Columns)*int64(block)-1)*int64(hop)
	return hop, block, useful, nil
}

// Reset starts a stream for a clip meant to span spanSamples samples (the sox -d
// value converted to samples; a clip shorter than the span fills fewer columns).
// It sets Hop and BlockSteps and reuses the windows of the current geometry; any
// others come from the shared edge window cache (see stft.EdgeAnalyzer).
// spanSamples must be > 0. It returns an error wrapping ErrInvalidConfig if the
// span is not positive, asks for more columns per second than the sample rate, or
// leaves a column covering more than 2^31-1 samples; the Fit and the stream then
// stay as they were.
func (f *Fit) Reset(spanSamples int64) error {
	hop, block, useful, err := f.fitGeometry(spanSamples)
	if err != nil {
		return err
	}
	if err := f.an.Reset(hop); err != nil {
		return err
	}
	f.hop, f.block, f.useful = hop, block, useful
	clear(f.acc)
	f.count, f.emitted, f.consumed = 0, 0, 0
	f.finished = false
	f.ready = true
	return nil
}

// NumColumns returns how many columns a run over signalLen samples emits at the
// current geometry: min(Columns, ceil(frames/BlockSteps)), where frames is the
// stft.EdgeAnalyzer frame count. It is 0 before the first Reset and for a clip
// shorter than Hop/2 samples (integer division), which yields no frame.
func (f *Fit) NumColumns(signalLen int64) int {
	if !f.ready {
		return 0
	}
	frames := f.an.NumFrames(signalLen)
	block := int64(f.block)
	cols := (frames + block - 1) / block
	return int(min(cols, int64(f.cfg.Columns)))
}

// Feed pushes src into the stream and calls onColumn for every column it
// completes, in order, with the column and its index from 0. col has length Bins
// and is Fit-owned: it is valid only until onColumn returns, so a caller that
// keeps a column copies it. Feed does nothing before the first Reset, after
// Finish, or once Columns columns have been emitted; samples that no kept column
// reads are not processed. onColumn must not be nil. Chunk sizes do not change
// the result.
func (f *Fit) Feed(src []float32, onColumn func(col []float32, index int)) {
	if !f.ready || f.finished || f.emitted >= f.cfg.Columns {
		return
	}
	if room := f.useful - f.consumed; int64(len(src)) > room {
		src = src[:room]
	}
	f.consumed += int64(len(src))
	f.an.Feed(src, func(power []float32) { f.frame(power, onColumn) })
}

// Finish ends the stream: it runs the frames of the drain padding and emits the
// last block, which holds fewer than BlockSteps frames when the frame count is
// not a multiple of it. After Finish, Feed and Finish do nothing until Reset.
func (f *Fit) Finish(onColumn func(col []float32, index int)) {
	if !f.ready || f.finished {
		return
	}
	f.finished = true
	if f.emitted >= f.cfg.Columns {
		return
	}
	f.an.Finish(func(power []float32) { f.frame(power, onColumn) })
	if f.count > 0 && f.emitted < f.cfg.Columns {
		f.emit(onColumn)
	}
}

// frame adds one frame's power to the pending block and emits the column when the
// block is full. Frames past the column limit are dropped.
func (f *Fit) frame(power []float32, onColumn func(col []float32, index int)) {
	if f.emitted >= f.cfg.Columns {
		return
	}
	lo, hi := f.sc.lo, f.sc.hi
	simdf32.Add(f.acc[lo:], f.acc[lo:], power[lo:hi])
	f.count++
	if f.count == f.block {
		f.emit(onColumn)
	}
}

// emit converts the pending block's mean power to decibels and hands it out.
func (f *Fit) emit(onColumn func(col []float32, index int)) {
	lo := f.sc.lo
	simdf32.Scale(f.mean[lo:], f.acc[lo:], float32(1/float64(f.count)))
	f.sc.apply(f.col, f.mean)
	onColumn(f.col, f.emitted)
	f.emitted++
	clear(f.acc[lo:])
	f.count = 0
}

// resolveSpan applies the default span of ComputeInto and Compute: the signal
// length, or one second of samples for an empty signal.
func (f *Fit) resolveSpan(signalLen int, spanSamples int64) int64 {
	switch {
	case spanSamples != 0:
		return spanSamples
	case signalLen > 0:
		return int64(signalLen)
	}
	return int64(f.cfg.SampleRate)
}

// ComputeInto writes the columns of signal into dst and returns their number.
// spanSamples is the span as in Reset; 0 selects len(signal), or one second of
// samples when signal is empty. dst.Data must have capacity for
// Bins()*NumColumns(len(signal)) values, or ComputeInto returns ErrBufferTooSmall
// and writes nothing (the Fit is left reset for that span). On success dst.Data
// is resliced to the exact length and dst.Bins and dst.Frames are set. It resets
// the stream first, so calls are independent. Allocation-free once the window
// cache is warm and dst.Data has the capacity.
func (f *Fit) ComputeInto(dst *Matrix, signal []float32, spanSamples int64) (int, error) {
	if err := f.Reset(f.resolveSpan(len(signal), spanSamples)); err != nil {
		return 0, err
	}
	bins := f.Bins()
	need := bins * f.NumColumns(int64(len(signal)))
	if cap(dst.Data) < need {
		return 0, ErrBufferTooSmall
	}
	dst.Data = dst.Data[:need]
	dst.Bins = bins
	onColumn := func(col []float32, index int) {
		copy(dst.Data[index*bins:(index+1)*bins], col)
	}
	f.Feed(signal, onColumn)
	f.Finish(onColumn)
	dst.Frames = f.emitted
	return f.emitted, nil
}

// Compute returns a freshly allocated Matrix for signal; see ComputeInto for the
// span. Reuse ComputeInto with a retained Matrix on a hot path.
func (f *Fit) Compute(signal []float32, spanSamples int64) (Matrix, error) {
	span := f.resolveSpan(len(signal), spanSamples)
	if err := f.Reset(span); err != nil {
		return Matrix{}, err
	}
	m := Matrix{Data: make([]float32, f.Bins()*f.NumColumns(int64(len(signal))))}
	_, err := f.ComputeInto(&m, signal, span)
	return m, err
}
