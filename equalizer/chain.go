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
// bands. Every Filter added to one chain must be built at the same sample rate;
// the chain does not verify this, so mixing rates yields an incorrect combined
// response. A FilterChain is not safe for concurrent use; run one instance per
// stream (for example one per channel).
type FilterChain struct {
	filters []*Filter
}

// NewFilterChain returns an empty FilterChain. An empty chain is a valid
// identity block: it copies input to output unchanged.
func NewFilterChain() *FilterChain { return &FilterChain{} }

// AddFilter appends f to the chain, in application order. A nil f is rejected
// with an error wrapping ErrInvalidConfig rather than deferred to a nil panic in
// ProcessInto. The filter is not copied; do not mutate it, reuse it in another
// chain, or add the same *Filter more than once (to this or any chain), since its
// state is shared through the pointer: sharing one filter's state across two
// positions corrupts the output.
func (c *FilterChain) AddFilter(f *Filter) error {
	if f == nil {
		return fmt.Errorf("%w: nil filter", ErrInvalidConfig)
	}
	c.filters = append(c.filters, f)
	return nil
}

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
// counting each filter's Passes.
func (c *FilterChain) NumSections() int {
	n := 0
	for _, f := range c.filters {
		n += len(f.sections)
	}
	return n
}
