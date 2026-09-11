package equalizer

import (
	"fmt"

	dsp "github.com/tphakala/go-audio-dsp"
)

// A Filter is a streaming dsp.Processor with zero latency and no tail, so it
// does not implement dsp.Flusher. This assertion fails to compile if the
// contract drifts.
var _ dsp.Processor = (*Filter)(nil)

// Filter is one RBJ biquad band, cascaded over its Passes, as a streaming
// dsp.Processor over mono float32 audio. Like Equalizer it uses float64
// coefficients and float64 state with float32 I/O, has zero latency (output
// aligns sample-for-sample with input) and no tail (so it does not implement
// dsp.Flusher), and Reset clears the state between streams. Its float32 path
// (ProcessInto) is bit-identical to an Equalizer of the same single band, so a
// FilterChain of Filters equals an Equalizer of the same bands. Unlike an
// Equalizer, whose bands are fixed at construction, a Filter is a single
// reusable stage a caller composes into a FilterChain. A Filter is not safe for
// concurrent use; run one instance per stream (for example one per channel).
type Filter struct {
	sampleRate int // the rate the coefficients were computed for; a FilterChain enforces one rate across its filters
	sections   []section
}

// NewFilter builds a Filter for one band at sampleRate. It validates sampleRate
// and the band and returns an error wrapping ErrInvalidConfig for the first
// invalid field. Passes == 0 means one section, matching Band's documented
// default.
func NewFilter(b Band, sampleRate int) (*Filter, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("%w: SampleRate must be > 0, got %d", ErrInvalidConfig, sampleRate)
	}
	c, err := newCoeffs(b, sampleRate)
	if err != nil {
		return nil, err
	}
	passes := b.Passes
	if passes == 0 {
		passes = 1
	}
	sections := make([]section, passes)
	for i := range sections {
		sections[i].c = c
	}
	return &Filter{sampleRate: sampleRate, sections: sections}, nil
}

// ProcessInto writes in filtered through this band's cascade into out and
// returns the number of samples written (len(in)). out must have room for
// len(in) samples (see MaxOutputLen); a shorter out returns ErrBufferTooSmall
// and leaves the filter state unchanged so the caller can retry. out may be
// exactly in for in-place filtering; a shifted overlap of out and in is not
// supported. Between sections a sample is the float32 output of the previous
// pass, matching Equalizer; use ApplyFloat64 for a float64-precision cascade.
func (f *Filter) ProcessInto(in, out []float32) (int, error) {
	if len(out) < len(in) {
		return 0, ErrBufferTooSmall
	}
	n := len(in)
	if n == 0 {
		return 0, nil
	}
	copy(out[:n], in) // a no-op when out aliases in; the cascade then runs in place
	f.applyInPlace32(out[:n])
	return n, nil
}

// ApplyFloat64 filters buf in place through this band's cascade, keeping full
// float64 precision between passes (no float32 truncation between sections).
// This is a numerically DIFFERENT filter from ProcessInto, which truncates to
// float32 at each stage: use ApplyFloat64 when the caller carries a float64
// buffer and wants the extra precision through the cascade, and ProcessInto for
// the float32 streaming path. The two share the same filter state, so use one
// per stream and Reset between switching.
func (f *Filter) ApplyFloat64(buf []float64) {
	for i := range f.sections {
		f.sections[i].run64(buf)
	}
}

// applyInPlace32 runs every section over buf in float32, without the bounds
// check or copy ProcessInto does; FilterChain uses it to run one shared buffer
// through each filter in turn.
func (f *Filter) applyInPlace32(buf []float32) {
	for i := range f.sections {
		f.sections[i].run(buf)
	}
}

// MaxOutputLen returns inputLen: the filter writes one output sample per input
// sample.
func (f *Filter) MaxOutputLen(inputLen int) int { return inputLen }

// Latency returns 0: a biquad's output sample aligns with its input sample.
func (f *Filter) Latency() int { return 0 }

// Reset clears the filter state of every section so the next call starts a new
// stream. The configured band is kept.
func (f *Filter) Reset() {
	for i := range f.sections {
		f.sections[i].reset()
	}
}

// NumSections returns the number of biquad sections (this band's Passes, with 0
// counted as 1).
func (f *Filter) NumSections() int { return len(f.sections) }
