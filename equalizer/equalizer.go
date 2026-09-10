package equalizer

import "fmt"

// Equalizer is a cascade of RBJ biquad filters that runs as a streaming
// dsp.Processor over mono float32 audio. It has zero latency (output aligns
// sample-for-sample with input) and no tail, so it does not implement
// dsp.Flusher; Reset clears the filter state between streams. The bands are
// fixed at construction. An Equalizer is not safe for concurrent use; run one
// instance per stream (for example one per channel).
type Equalizer struct {
	sampleRate int
	sections   []section
}

// New builds an Equalizer from cfg. It validates SampleRate and every band and
// returns an error wrapping ErrInvalidConfig for the first invalid field. An
// empty Bands list is valid and yields an identity block.
func New(cfg Config) (*Equalizer, error) {
	if cfg.SampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, cfg.SampleRate)
	}
	// One section per pass. Pre-size to the band count, which is the exact total
	// for the common single-pass-per-band case, and let append grow it for a band
	// that cascades more passes. newCoeffs validates Passes (into [0, maxPasses])
	// before the inner loop, so the cascade is built in a single validation pass
	// with no intermediate plan slice.
	sections := make([]section, 0, len(cfg.Bands))
	for i := range cfg.Bands {
		b := cfg.Bands[i]
		c, err := newCoeffs(b, cfg.SampleRate)
		if err != nil {
			return nil, fmt.Errorf("band %d: %w", i, err)
		}
		passes := b.Passes
		if passes == 0 {
			passes = 1
		}
		for range passes {
			sections = append(sections, section{c: c})
		}
	}
	return &Equalizer{sampleRate: cfg.SampleRate, sections: sections}, nil
}

// ProcessInto writes in filtered through the cascade into out and returns the
// number of samples written (len(in)). out must have room for len(in) samples
// (see MaxOutputLen); a shorter out returns ErrBufferTooSmall and leaves the
// filter state unchanged so the caller can retry. out may be exactly in for
// in-place filtering; a shifted overlap of out and in is not supported.
func (e *Equalizer) ProcessInto(in, out []float32) (int, error) {
	if len(out) < len(in) {
		return 0, ErrBufferTooSmall
	}
	n := len(in)
	if n == 0 {
		return 0, nil
	}
	copy(out[:n], in) // a no-op when out aliases in; the cascade then runs in place
	for i := range e.sections {
		e.sections[i].run(out[:n])
	}
	return n, nil
}

// MaxOutputLen returns inputLen: the equalizer writes one output sample per
// input sample.
func (e *Equalizer) MaxOutputLen(inputLen int) int { return inputLen }

// Latency returns 0: a biquad's output sample aligns with its input sample.
func (e *Equalizer) Latency() int { return 0 }

// Reset clears the filter state of every section so the next ProcessInto starts
// a new stream. The configured bands are kept.
func (e *Equalizer) Reset() {
	for i := range e.sections {
		e.sections[i].reset()
	}
}

// NumSections returns the total number of biquad sections, counting each band's
// Passes. An empty (identity) equalizer returns 0.
func (e *Equalizer) NumSections() int { return len(e.sections) }
