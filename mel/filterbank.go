package mel

import (
	"fmt"
	"math"
)

// Slaney mel-scale constants (librosa htk=False): linear spacing fsp Hz per mel
// below minLogHz, logarithmic above, meeting continuously at minLogMel.
const (
	melFsp       = 200.0 / 3.0          // Hz per mel in the linear region
	melMinLogHz  = 1000.0               // linear/log breakpoint in Hz
	melMinLogMel = melMinLogHz / melFsp // == 15, the breakpoint in mel
)

// melLogstep is ln(6.4)/27, the mel-scale log slope, computed once.
var melLogstep = math.Log(6.4) / 27

// HzToMel maps a frequency in Hz to the mel scale under scale. Slaney (the zero
// value) is linear below 1 kHz and logarithmic above; HTK is 2595*log10(1+f/700).
func HzToMel(hz float64, scale MelScale) float64 {
	if scale == HTK {
		return 2595 * math.Log10(1+hz/700)
	}
	if hz < melMinLogHz {
		return hz / melFsp
	}
	return melMinLogMel + math.Log(hz/melMinLogHz)/melLogstep
}

// MelToHz maps a mel value back to Hz under scale, the inverse of HzToMel.
func MelToHz(mel float64, scale MelScale) float64 {
	if scale == HTK {
		return 700 * (math.Pow(10, mel/2595) - 1)
	}
	if mel < melMinLogMel {
		return melFsp * mel
	}
	return melMinLogHz * math.Exp(melLogstep*(mel-melMinLogMel))
}

// rowSpan describes one mel row's nonzero weights inside the row-contiguous
// weights slice: the row's first nonzero FFT bin (start), the offset of its
// weights within Filterbank.weights (off), and the number of contiguous weights
// (n). An all-zero row has n == 0.
type rowSpan struct {
	start, off, n int
}

// Filterbank is an immutable mel filterbank: numMels rows over numBins FFT bins,
// stored sparsely as each row's first nonzero bin plus its contiguous nonzero
// weights. It is read-only after construction, so one *Filterbank is safe to
// share between goroutines and between Extractor/ColumnSource instances.
type Filterbank struct {
	numMels, numBins int
	rows             []rowSpan
	weights          []float32
}

// FilterbankConfig parameterizes NewFilterbank. The fields mirror the transform
// and band fields of Config.
type FilterbankConfig struct {
	SampleRate int
	FrameSize  int // NumBins is FrameSize/2 + 1
	NumMels    int
	MinHz      float64
	MaxHz      float64 // 0 means Nyquist
	Scale      MelScale
	Norm       Norm
}

// validate checks a FilterbankConfig, wrapping ErrInvalidConfig with the first
// offending field. FrameSize is validated here (not deferred to stft) so a
// standalone NewFilterbank rejects a bad transform size on its own.
func (c FilterbankConfig) validate() error {
	if c.SampleRate <= 0 {
		return fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, c.SampleRate)
	}
	if c.FrameSize < 2 || c.FrameSize&(c.FrameSize-1) != 0 {
		return fmt.Errorf("%w: FrameSize must be a power of two >= 2, got %d", ErrInvalidConfig, c.FrameSize)
	}
	if c.NumMels <= 0 {
		return fmt.Errorf("%w: NumMels must be > 0, got %d", ErrInvalidConfig, c.NumMels)
	}
	if !c.Scale.valid() {
		return fmt.Errorf("%w: unknown Scale %d", ErrInvalidConfig, int(c.Scale))
	}
	if !c.Norm.valid() {
		return fmt.Errorf("%w: unknown Norm %d", ErrInvalidConfig, int(c.Norm))
	}
	if math.IsNaN(c.MinHz) || math.IsInf(c.MinHz, 0) || c.MinHz < 0 {
		return fmt.Errorf("%w: MinHz must be a finite value >= 0, got %g", ErrInvalidConfig, c.MinHz)
	}
	if math.IsNaN(c.MaxHz) || math.IsInf(c.MaxHz, 0) || c.MaxHz < 0 {
		return fmt.Errorf("%w: MaxHz must be a finite value >= 0 (0 means Nyquist), got %g", ErrInvalidConfig, c.MaxHz)
	}
	nyquist := float64(c.SampleRate) / 2
	maxHz := c.MaxHz
	if maxHz == 0 {
		maxHz = nyquist
	} else if maxHz > nyquist {
		// A model front end silently clamped to a different band would produce
		// features that do not match training, so this is an error, not a clamp.
		return fmt.Errorf("%w: MaxHz %g exceeds Nyquist %g", ErrInvalidConfig, c.MaxHz, nyquist)
	}
	if c.MinHz >= maxHz {
		return fmt.Errorf("%w: MinHz %g must be < effective MaxHz %g", ErrInvalidConfig, c.MinHz, maxHz)
	}
	return nil
}

// NewFilterbank builds a generated (Slaney or HTK) mel filterbank for cfg. The
// triangles follow librosa filters.mel: NumMels+2 mel-spaced edge frequencies,
// each band a triangle from edge m to m+2 peaking at m+1, optionally area
// normalized (Norm). Rows too narrow to cover a bin come back empty and project
// to 0, the same result librosa produces (librosa also logs a warning; this
// package, being pure DSP, does not).
func NewFilterbank(cfg FilterbankConfig) (*Filterbank, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	numBins := cfg.FrameSize/2 + 1
	nyquist := float64(cfg.SampleRate) / 2
	maxHz := cfg.MaxHz
	if maxHz == 0 {
		maxHz = nyquist
	}
	binHz := float64(cfg.SampleRate) / float64(cfg.FrameSize)

	// Mel-spaced edge frequencies in Hz. The grid is strictly increasing (maxHz >
	// MinHz from validation, NumMels >= 1, MelToHz strictly increasing), so every
	// ctr-lo and hi-ctr below is > 0 and no ramp divides by zero.
	melMin := HzToMel(cfg.MinHz, cfg.Scale)
	melMax := HzToMel(maxHz, cfg.Scale)
	pts := make([]float64, cfg.NumMels+2)
	for i := range pts {
		pts[i] = MelToHz(melMin+(melMax-melMin)*float64(i)/float64(cfg.NumMels+1), cfg.Scale)
	}

	fb := &Filterbank{
		numMels: cfg.NumMels,
		numBins: numBins,
		rows:    make([]rowSpan, 0, cfg.NumMels),
	}
	row := make([]float64, numBins)
	for m := range cfg.NumMels {
		lo, ctr, hi := pts[m], pts[m+1], pts[m+2]
		norm := 1.0
		if cfg.Norm == NormSlaney {
			norm = 2 / (hi - lo)
		}
		for k := range numBins {
			f := float64(k) * binHz
			lower := (f - lo) / (ctr - lo)
			upper := (hi - f) / (hi - ctr)
			row[k] = math.Max(0, math.Min(lower, upper)) * norm
		}
		fb.appendRow(row)
	}
	return fb, nil
}

// appendRow trims a dense float64 row to its nonzero span and appends the span,
// cast to float32, to the filterbank's row-contiguous weights.
func (fb *Filterbank) appendRow(row []float64) {
	first, last := -1, -1
	for k, v := range row {
		if v != 0 {
			if first < 0 {
				first = k
			}
			last = k
		}
	}
	span := rowSpan{off: len(fb.weights)}
	if first >= 0 {
		span.start = first
		span.n = last - first + 1
		for k := first; k <= last; k++ {
			fb.weights = append(fb.weights, float32(row[k]))
		}
	}
	fb.rows = append(fb.rows, span)
}

// FilterbankFromRows builds a filterbank from dense rows, a NumMels x NumBins
// matrix (each inner slice one mel row over the FFT bins). It requires at least
// one row, every row the same length NumBins >= 2, and finite, non-negative
// values; it strips each row's leading and trailing zeros into a sparse span and
// keeps interior zeros, so any custom shape (non-triangular or overlapping
// filters) projects exactly as its dense form. Weights must be non-negative
// because a mel filterbank projects non-negative spectral energy: a negative
// weight could drive a projected value below zero and then to NaN through the
// log stage. The rows are copied; the caller may reuse the input.
func FilterbankFromRows(rows [][]float32) (*Filterbank, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("%w: rows must have at least one row", ErrInvalidConfig)
	}
	numBins := len(rows[0])
	if numBins < 2 {
		return nil, fmt.Errorf("%w: each row must have at least 2 bins, got %d", ErrInvalidConfig, numBins)
	}
	fb := &Filterbank{
		numMels: len(rows),
		numBins: numBins,
		rows:    make([]rowSpan, 0, len(rows)),
	}
	for m, r := range rows {
		if len(r) != numBins {
			return nil, fmt.Errorf("%w: row %d has %d bins, want %d (all rows must match)", ErrInvalidConfig, m, len(r), numBins)
		}
		first, last := -1, -1
		for k, v := range r {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("%w: row %d bin %d is not finite (%g)", ErrInvalidConfig, m, k, v)
			}
			if v < 0 {
				return nil, fmt.Errorf("%w: row %d bin %d is negative (%g); mel filterbank weights must be non-negative", ErrInvalidConfig, m, k, v)
			}
			if v != 0 {
				if first < 0 {
					first = k
				}
				last = k
			}
		}
		span := rowSpan{off: len(fb.weights)}
		if first >= 0 {
			span.start = first
			span.n = last - first + 1
			fb.weights = append(fb.weights, r[first:last+1]...) // append copies, so the caller may reuse r
		}
		fb.rows = append(fb.rows, span)
	}
	return fb, nil
}

// NumMels returns the number of mel rows.
func (fb *Filterbank) NumMels() int { return fb.numMels }

// NumBins returns the number of FFT bins each row spans, FrameSize/2 + 1.
func (fb *Filterbank) NumBins() int { return fb.numBins }

// Row returns the sparse view of mel row m: its first nonzero bin and its
// contiguous weights (leading and trailing zeros trimmed, interior zeros kept).
// An empty row returns (0, nil). Do not mutate the returned slice; it backs the
// filterbank.
func (fb *Filterbank) Row(m int) (start int, weights []float32) {
	r := fb.rows[m]
	return r.start, fb.weights[r.off : r.off+r.n]
}

// Dense returns a freshly allocated dense copy of the filterbank, NumMels rows of
// NumBins values each, with the trimmed spans placed back at their start bin. It
// is for tests and inspection, not the hot path.
func (fb *Filterbank) Dense() [][]float32 {
	out := make([][]float32, fb.numMels)
	for m := range out {
		out[m] = make([]float32, fb.numBins)
		r := fb.rows[m]
		copy(out[m][r.start:], fb.weights[r.off:r.off+r.n])
	}
	return out
}
