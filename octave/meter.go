package octave

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/equalizer"
)

// Defaults applied to zero Config fields.
const (
	defaultMinFreq       = 25.0
	defaultMaxFreq       = 20000.0
	defaultNyquistMargin = 0.95
	defaultOrder         = 1
	maxOrder             = 8

	// chunk bounds the float64 scratch the meter filters through per step, so its
	// memory is O(1) regardless of how much audio one Process call is given.
	chunk = 4096

	// dbFloor and dbCeil bound DB so a silent or overflowing band reads a finite
	// level: 10*log10(1e-20) = -200 dB.
	dbFloor = 1e-20
	dbCeil  = 1e20
)

// Config configures a Meter. The zero value of every field except SampleRate and
// PeriodSamples selects its default.
type Config struct {
	// SampleRate is the input rate in Hz, required.
	SampleRate int
	// PeriodSamples is the integration period in samples, required and > 0. A
	// caller that wants one-second levels passes SampleRate.
	PeriodSamples int
	// MinFreq and MaxFreq select the bands whose nominal centre lies in
	// [MinFreq, MaxFreq] Hz. Defaults 25 and 20000.
	MinFreq, MaxFreq float64
	// NyquistMargin is the fraction of the Nyquist frequency that a band's upper
	// edge must stay below for the band to be included, in (0, 1]. Default 0.95.
	NyquistMargin float64
	// Order is the number of identical biquad sections cascaded per band, 0 meaning
	// 1, at most 8. See the package documentation for what Order does to the
	// level and the bandwidth.
	Order int
}

// Meter is a streaming 1/3-octave band level meter. It is not a dsp.Processor:
// its output is band energy per period, not audio. A Meter is not safe for
// concurrent use; run one per stream.
type Meter struct {
	period  int
	bands   []Band
	filters []*equalizer.Filter
	acc     []float64 // per band: running sum of squares in the current period
	ms      []float64 // per band: the mean square handed to emit
	filled  int       // samples accumulated in the current period
	x, y    []float64 // chunk scratch: converted input, and its per-band copy
}

// New builds a Meter. It validates cfg and returns an error wrapping
// ErrInvalidConfig for the first invalid field, including a configuration for
// which no band fits below the Nyquist margin.
func New(cfg Config) (*Meter, error) {
	if cfg.SampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, cfg.SampleRate)
	}
	if cfg.PeriodSamples <= 0 {
		return nil, fmt.Errorf("%w: PeriodSamples must be > 0, got %d", ErrInvalidConfig, cfg.PeriodSamples)
	}
	minF, maxF, margin, order := cfg.MinFreq, cfg.MaxFreq, cfg.NyquistMargin, cfg.Order
	if minF == 0 {
		minF = defaultMinFreq
	}
	if maxF == 0 {
		maxF = defaultMaxFreq
	}
	if margin == 0 {
		margin = defaultNyquistMargin
	}
	if order == 0 {
		order = defaultOrder
	}
	// Negated comparisons so NaN (every comparison false) is rejected too.
	if !(minF > 0) || math.IsInf(minF, 0) {
		return nil, fmt.Errorf("%w: MinFreq must be finite and > 0, got %g", ErrInvalidConfig, cfg.MinFreq)
	}
	if !(maxF >= minF) || math.IsInf(maxF, 0) {
		return nil, fmt.Errorf("%w: MaxFreq must be finite and >= MinFreq (%g), got %g", ErrInvalidConfig, minF, cfg.MaxFreq)
	}
	if !(margin > 0 && margin <= 1) {
		return nil, fmt.Errorf("%w: NyquistMargin must be in (0, 1], got %g", ErrInvalidConfig, cfg.NyquistMargin)
	}
	if order < 1 || order > maxOrder {
		return nil, fmt.Errorf("%w: Order must be in [1, %d], got %d", ErrInvalidConfig, maxOrder, cfg.Order)
	}

	bands := selectBands(minF, maxF, margin, float64(cfg.SampleRate)/2)
	if len(bands) == 0 {
		return nil, fmt.Errorf("%w: no band in [%g, %g] Hz fits below %g of the Nyquist frequency at %d Hz", ErrInvalidConfig, minF, maxF, margin, cfg.SampleRate)
	}

	// The equalizer BandPass takes its width in Hz and converts it to octaves as
	// log2((f+W/2)/(f-W/2)). Choose W so that this equals log2(Upper/Lower), the
	// band's own octave width, instead of Upper-Lower (which is off by about 0.2%).
	r := math.Pow(10, 0.1)
	filters := make([]*equalizer.Filter, len(bands))
	for i, b := range bands {
		f, err := equalizer.NewFilter(equalizer.Band{
			Type:      equalizer.BandPass,
			Frequency: b.Exact,
			WidthHz:   2 * b.Exact * (r - 1) / (r + 1),
			Passes:    order,
		}, cfg.SampleRate)
		if err != nil {
			return nil, fmt.Errorf("octave: band %g Hz: %w", b.Nominal, err)
		}
		filters[i] = f
	}
	return &Meter{
		period:  cfg.PeriodSamples,
		bands:   bands,
		filters: filters,
		acc:     make([]float64, len(bands)),
		ms:      make([]float64, len(bands)),
		x:       make([]float64, chunk),
		y:       make([]float64, chunk),
	}, nil
}

// Bands returns the included bands in ascending order. The slice is the meter's
// own and must not be modified.
func (m *Meter) Bands() []Band { return m.bands }

// NumBands returns the number of included bands, the length of the slice Process
// passes to emit.
func (m *Meter) NumBands() int { return len(m.bands) }

// Process filters in through the band bank and calls emit once for every period
// of PeriodSamples samples it completes, with the mean square of each band
// ordered as Bands. The slice passed to emit is reused and valid only during the
// call. A chunk may complete several periods or none; the partial period carries
// over to the next call, and the emitted values do not depend on how the input
// is split into chunks. A nil emit discards the periods.
//
// A non-finite sample is treated as 0, so one bad sample cannot poison the
// filter state until the next Reset. Process allocates nothing when emit does
// not escape.
func (m *Meter) Process(in []float32, emit func(meanSquare []float64)) {
	for len(in) > 0 {
		n := min(len(in), chunk, m.period-m.filled)
		x := m.x[:n]
		for i, s := range in[:n] {
			v := float64(s)
			if v-v != 0 { // NaN or Inf
				v = 0
			}
			x[i] = v
		}
		y := m.y[:n]
		for b, f := range m.filters {
			copy(y, x)
			f.ApplyFloat64(y)
			// Sequential accumulation, so the sum does not depend on where chunk
			// boundaries fall.
			acc := m.acc[b]
			for _, v := range y {
				acc += v * v
			}
			m.acc[b] = acc
		}
		m.filled += n
		in = in[n:]
		if m.filled == m.period {
			period := float64(m.period)
			for b := range m.acc {
				m.ms[b] = m.acc[b] / period
				m.acc[b] = 0
			}
			m.filled = 0
			if emit != nil {
				emit(m.ms)
			}
		}
	}
}

// Reset clears the filter state and the partial period, keeping the
// configuration, so the next Process call starts a new stream.
func (m *Meter) Reset() {
	for _, f := range m.filters {
		f.Reset()
	}
	clear(m.acc)
	m.filled = 0
}

// DB converts a mean square to decibels, 10*log10(meanSquare), relative to a
// full scale amplitude of 1.0 (a full scale sine has mean square 0.5, -3.01 dB).
// The result is always finite: a zero, negative or NaN input reads -200 dB and
// anything above 1e20 reads +200 dB.
func DB(meanSquare float64) float64 {
	if !(meanSquare > dbFloor) {
		return 10 * math.Log10(dbFloor)
	}
	return 10 * math.Log10(min(meanSquare, dbCeil))
}
