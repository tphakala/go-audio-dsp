package equalizer

import (
	"fmt"

	dsp "github.com/tphakala/go-audio-dsp"
)

// A FilterChain is a streaming dsp.Processor with zero latency and no tail, so
// it does not implement dsp.Flusher.
var _ dsp.Processor = (*FilterChain)(nil)

// FilterChain is an ordered, growable cascade of Filters as a streaming
// dsp.Processor over mono float32 audio. Unlike Equalizer, whose bands are fixed
// at construction, a FilterChain is built by appending Filters, so a caller can
// assemble it incrementally (for example from per-source settings) and rebuild
// it per sample rate with independent state. Running the same bands as one
// Filter each through a FilterChain is bit-identical to an Equalizer of those
// bands. Every Filter in one chain must be built at the same sample rate, since
// mixing rates would yield an incorrect combined response; AddFilter enforces
// this by rejecting a filter whose rate differs from the chain's first. A
// FilterChain is not safe for concurrent use; run one instance per stream (for
// example one per channel). Response and ResponseInto report the chain's
// frequency response, and Marks the frequencies a plot of it should sample.
type FilterChain struct {
	sampleRate int // the rate of the first filter added; every later filter must match
	filters    []*Filter
}

// NewFilterChain returns an empty FilterChain. An empty chain is a valid
// identity block: it copies input to output unchanged.
func NewFilterChain() *FilterChain { return &FilterChain{} }

// AddFilter appends f to the chain, in application order. It returns an error
// wrapping ErrInvalidConfig for a nil filter (rather than deferring to a nil
// panic in ProcessInto) or for a filter built at a different sample rate than
// the chain's first, which would cascade incompatible coefficients. The filter
// is not copied; do not mutate it, reuse it in another chain, or add the same
// *Filter more than once (to this or any chain), since its state is shared
// through the pointer: sharing one filter's state across two positions corrupts
// the output.
func (c *FilterChain) AddFilter(f *Filter) error {
	if f == nil {
		return fmt.Errorf("%w: nil filter", ErrInvalidConfig)
	}
	if len(c.filters) == 0 {
		c.sampleRate = f.sampleRate
	} else if f.sampleRate != c.sampleRate {
		return fmt.Errorf("%w: filter sample rate %d differs from the chain's %d", ErrInvalidConfig, f.sampleRate, c.sampleRate)
	}
	c.filters = append(c.filters, f)
	return nil
}

// Marks returns the union of the marks of the filters AddFilter accepted, sorted
// and duplicate-free, in a new slice; nil when no filter has a mark. It is
// derived from the chain on each call, so a rejected filter never contributes.
// Pass it to LogSweepWithMarks so a narrow filter is not missed between sweep
// points; see Filter.Marks for the rule per constructor.
func (c *FilterChain) Marks() []float64 {
	marks := make([]float64, 0, len(c.filters))
	for _, f := range c.filters {
		marks = append(marks, f.marks...)
	}
	if len(marks) == 0 {
		return nil
	}
	return sortedUnique(marks)
}

// SampleRate returns the sample rate of the chain's filters, which is the rate
// of the first filter added and is shared by every later one; 0 for an empty
// chain. The usable response range is [0, SampleRate()/2].
func (c *FilterChain) SampleRate() int { return c.sampleRate }

// Len returns the number of filters in the chain.
func (c *FilterChain) Len() int { return len(c.filters) }

// ProcessInto writes in filtered through every filter in order into out and
// returns the number of samples written (len(in)). out must have room for
// len(in) samples (see MaxOutputLen); a shorter out returns ErrBufferTooSmall
// and leaves every filter's state unchanged so the caller can retry. out may be
// exactly in for in-place filtering; a shifted overlap of out and in is not
// supported. Between filters a sample is the float32 output of the previous one,
// matching Equalizer; use ApplyFloat64 for a float64-precision cascade.
func (c *FilterChain) ProcessInto(in, out []float32) (int, error) {
	if len(out) < len(in) {
		return 0, ErrBufferTooSmall
	}
	n := len(in)
	if n == 0 {
		return 0, nil
	}
	copy(out[:n], in) // a no-op when out aliases in; the cascade then runs in place
	for _, f := range c.filters {
		f.applyInPlace32(out[:n])
	}
	return n, nil
}

// ApplyFloat64 filters buf in place through every filter in order, keeping full
// float64 precision between filters and between their passes (no float32
// truncation). Like Filter.ApplyFloat64 this is a numerically different filter
// from ProcessInto, which truncates to float32 at each stage; use it when the
// caller carries a float64 buffer and wants the extra precision through the
// cascade. Use one path per stream and Reset between switching.
func (c *FilterChain) ApplyFloat64(buf []float64) {
	for _, f := range c.filters {
		f.ApplyFloat64(buf)
	}
}

// MaxOutputLen returns inputLen: the chain writes one output sample per input
// sample.
func (c *FilterChain) MaxOutputLen(inputLen int) int { return inputLen }

// Latency returns 0: every biquad's output sample aligns with its input sample.
func (c *FilterChain) Latency() int { return 0 }

// Reset clears the state of every filter so the next call starts a new stream.
// The filters and their order are kept.
func (c *FilterChain) Reset() {
	for _, f := range c.filters {
		f.Reset()
	}
}

// NumSections returns the total number of biquad sections across all filters,
// counting every filter's sections.
func (c *FilterChain) NumSections() int {
	n := 0
	for _, f := range c.filters {
		n += f.NumSections()
	}
	return n
}
