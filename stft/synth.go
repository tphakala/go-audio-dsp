package stft

import (
	"fmt"

	"github.com/tphakala/simd/f32"
)

// minWOLANorm is the smallest overlap-add normalization NewSynthesizer accepts at
// any hop position; below it the reciprocal is unbounded and reconstruction is
// meaningless. It matches the floor the simd ISTFT uses.
const minWOLANorm = 1e-8

// Synthesizer is the weighted overlap-add (WOLA) counterpart of an Analyzer: it
// inverse-transforms frame spectra, applies the synthesis window, accumulates
// them into an overlap-add buffer, and releases each hop block once every
// overlapping frame has been added, divided by the WOLA normalization so an
// unmodified analyze/synthesize pass reconstructs the input.
//
// Use it from the Analyzer's Feed callback: for every frame, call Add with the
// (possibly modified) spectrum, then exactly one of Finish (keep the block) or
// Discard (drop it). The stream is modelled as FrameSize-HopSize leading zeros
// followed by the input, so for a HopSize that divides FrameSize the first FrameSize/HopSize-1 blocks are incomplete
// leading zeros; a caller feeds that preroll to the Analyzer and discards those
// blocks. Finish with an out shorter than a hop drops the rest of the block, which
// is how a final flush is clipped. Reset clears the accumulator and must be paired
// with the Analyzer's Reset.
//
// A Synthesizer shares its Analyzer's window and inverse-transform scratch, so
// both belong to one goroutine and Add must not run concurrently with Feed on
// that Analyzer outside its callback. Not safe for concurrent use.
type Synthesizer struct {
	an      *Analyzer
	invNorm []float32 // len hop: 1 / WOLA normalization per position in a block
	synth   []float32 // inverse frame, len n
	ola     []float32 // overlap-add accumulator, len n
}

// NewSynthesizer builds a Synthesizer for an, using an's window and frame
// geometry. It returns an error wrapping ErrInvalidConfig when the window's
// overlap sum vanishes at some position of the hop (for example HopSize equal to
// FrameSize, or a window much shorter than the hop), since no normalization can
// reconstruct those samples. A HopSize that does not divide FrameSize is accepted
// as long as the overlap sum stays positive; the leading-zero preroll then ends part way through a block, so only floor((FrameSize-HopSize)/HopSize) blocks are entirely leading zeros.
func NewSynthesizer(an *Analyzer) (*Synthesizer, error) {
	norm := WOLANorm(an.window, an.window, an.hop)
	for i, v := range norm {
		if !(v > minWOLANorm) {
			return nil, fmt.Errorf("%w: window overlap sum at hop position %d is %g, too small to reconstruct", ErrInvalidConfig, i, v)
		}
	}
	inv := make([]float32, an.hop)
	f32.Reciprocal(inv, norm) // full-precision division, not approximate rcp
	return &Synthesizer{
		an:      an,
		invNorm: inv,
		synth:   make([]float32, an.n),
		ola:     make([]float32, an.n),
	}, nil
}

// Add inverse-transforms spec (an Analyzer half-spectrum, possibly modified) and
// overlap-adds the windowed frame into the accumulator. It does not modify spec
// and is allocation-free.
func (s *Synthesizer) Add(spec []complex64) {
	s.an.Inverse(s.synth, spec)
	// Fuse the synthesis window and overlap-add: ola += synth * window. synth is
	// only the inverse-transform output and is not read after this. MulAdd uses a
	// hardware FMA where available, so it is tolerance-stable across CPU tiers,
	// not bit-identical.
	f32.MulAdd(s.ola, s.synth, s.an.window)
}

// Finish finalizes the oldest hop block: it writes the block divided by the WOLA
// normalization into out, clipped to len(out), advances the accumulator by one
// hop, and returns the samples written.
func (s *Synthesizer) Finish(out []float32) int {
	m := min(s.an.hop, len(out))
	f32.Mul(out[:m], s.ola[:m], s.invNorm[:m])
	s.advance()
	return m
}

// Discard drops the oldest hop block (for example a leading-zero block) and
// advances the accumulator by one hop.
func (s *Synthesizer) Discard() { s.advance() }

func (s *Synthesizer) advance() {
	copy(s.ola, s.ola[s.an.hop:])
	clear(s.ola[s.an.n-s.an.hop:])
}

// Reset clears the overlap-add accumulator so the next Add starts a new stream.
func (s *Synthesizer) Reset() { clear(s.ola) }
