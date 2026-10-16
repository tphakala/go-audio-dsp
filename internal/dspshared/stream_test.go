package dspshared

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/stft"
)

const (
	testFrame = 256
	testHop   = 64
)

// hookRec is a stand-in denoise method: it records every call the Stream makes
// and returns each frame's spectrum unchanged, held back by delay frames
// through a ring, so an unmodified analyze, delay, synthesize pass reconstructs
// the input. Its methods are bound as method values, as the production blocks
// bind theirs.
type hookRec struct {
	delay    int
	bins     int
	ring     []complex64 // (delay+1) rows of bins
	idx      []int64
	flushing []bool
	resets   int
}

func newStream(tb testing.TB, delay int) (*Stream, *hookRec) {
	tb.Helper()
	an, err := stft.NewAnalyzer(STFTConfig(testFrame, testHop))
	if err != nil {
		tb.Fatal(err)
	}
	h := &hookRec{delay: delay, bins: an.NumBins()}
	h.ring = make([]complex64, (delay+1)*h.bins)
	s := new(Stream)
	if err := s.Init(an, delay, h.frame, h.resetState); err != nil {
		tb.Fatal(err)
	}
	s.Reset()
	h.resets = 0 // the construction Reset is not part of what the tests count
	return s, h
}

func (h *hookRec) row(f int64) []complex64 {
	m := int64(h.delay + 1)
	s := int(f%m) * h.bins
	return h.ring[s : s+h.bins]
}

func (h *hookRec) frame(f int64, spec []complex64, _ []float32, flushing bool) []complex64 {
	h.idx = append(h.idx, f)
	h.flushing = append(h.flushing, flushing)
	copy(h.row(f), spec)
	if f < int64(h.delay) {
		return nil
	}
	return h.row(f - int64(h.delay))
}

func (h *hookRec) resetState() { h.resets++ }

// testSignal is white noise with a slow chirp on top, so a misaligned or
// mis-scaled output differs from the input.
func testSignal(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	x := make([]float32, n)
	for i := range x {
		ph := 2 * math.Pi * (0.02*float64(i) + 1e-5*float64(i)*float64(i))
		x[i] = float32(0.3*math.Sin(ph) + 0.2*(r.Float64()*2-1))
	}
	return x
}

// chunkings returns the chunk-length sequences the tests cover for an input of
// n samples: one chunk, a fixed irregular list repeated, and a seeded shuffle.
func chunkings(n int) map[string][]int {
	fixed := []int{1, 7, 100, 999, 64, 256, 5}
	r := rand.New(rand.NewPCG(7, 11))
	var shuffled []int
	for sum := 0; sum < n; {
		c := 1 + r.IntN(700)
		shuffled = append(shuffled, c)
		sum += c
	}
	repeated := []int{}
	for sum := 0; sum < n; {
		for _, c := range fixed {
			repeated = append(repeated, c)
			sum += c
		}
	}
	return map[string][]int{"whole": {n}, "fixed": repeated, "shuffled": shuffled}
}

// streamAll feeds x in the given chunk lengths (clipped to the input) and
// flushes, asserting the buffer-sizing contract at every call, and returns the
// concatenated output.
func streamAll(t *testing.T, s *Stream, x []float32, chunks []int) []float32 {
	t.Helper()
	var got []float32
	pos := 0
	for _, c := range chunks {
		if pos >= len(x) {
			break
		}
		in := x[pos:min(pos+c, len(x))]
		pos += len(in)
		want := s.pending(len(in))
		out := make([]float32, s.MaxOutputLen(len(in)))
		n, err := s.ProcessInto(in, out)
		if err != nil {
			t.Fatalf("ProcessInto(%d samples) with MaxOutputLen buffer: %v", len(in), err)
		}
		if n != want {
			t.Fatalf("ProcessInto returned %d samples, pending predicted %d", n, want)
		}
		got = append(got, out[:n]...)
	}
	if rem := int(s.totalIn - s.totalOut); rem > s.Latency()+s.hop {
		t.Fatalf("remainder %d before FlushInto exceeds Latency()+hop = %d", rem, s.Latency()+s.hop)
	}
	fb := make([]float32, s.Latency()+s.hop)
	n, err := s.FlushInto(fb)
	if err != nil {
		t.Fatalf("FlushInto with Latency()+hop buffer: %v", err)
	}
	return append(got, fb[:n]...)
}

// TestStreamFrameIndexAndFlushFlag pins what the hook sees: frame indices count
// 0..F-1 with no gap or repeat whatever the chunking, flushing is false for
// frames fed by ProcessInto and true for frames fed by FlushInto, the reset
// hook runs once per FlushInto and once per Reset, and a new stream after
// FlushInto restarts at frame 0.
func TestStreamFrameIndexAndFlushFlag(t *testing.T) {
	x := testSignal(3000, 1)
	for _, delay := range []int{0, 1, 2} {
		for name, chunks := range chunkings(len(x)) {
			s, h := newStream(t, delay)
			streamAll(t, s, x, chunks)
			if h.resets != 1 {
				t.Errorf("delay %d %s: reset hook ran %d times after one FlushInto, want 1", delay, name, h.resets)
			}
			if len(h.idx) == 0 {
				t.Fatalf("delay %d %s: hook never ran", delay, name)
			}
			firstFlush := slices.Index(h.flushing, true)
			if firstFlush < 0 {
				t.Fatalf("delay %d %s: no frame was fed with flushing set", delay, name)
			}
			for i, f := range h.idx {
				if f != int64(i) {
					t.Fatalf("delay %d %s: call %d got frame index %d, want %d", delay, name, i, f, i)
				}
				if want := i >= firstFlush; h.flushing[i] != want {
					t.Fatalf("delay %d %s: frame %d flushing = %v, want %v", delay, name, i, h.flushing[i], want)
				}
			}
			// A second stream on the same Stream restarts the frame index at 0.
			h.idx, h.flushing = nil, nil
			streamAll(t, s, x, chunks)
			if len(h.idx) == 0 || h.idx[0] != 0 {
				t.Errorf("delay %d %s: second stream first frame index = %v, want 0", delay, name, h.idx[:min(1, len(h.idx))])
			}
			s.Reset()
			if h.resets != 3 {
				t.Errorf("delay %d %s: reset hook ran %d times after two FlushInto and one Reset, want 3", delay, name, h.resets)
			}
		}
	}
}

// TestStreamDelayAlignment pins that for every delay the output is the input
// sample for sample: a wrong warm count shifts the output by a hop and a wrong
// emission delay shifts it by delay hops. The tolerance is the one
// stft/synth_test.go uses for an unmodified analyze and synthesize pass.
func TestStreamDelayAlignment(t *testing.T) {
	x := testSignal(4000, 2)
	for _, delay := range []int{0, 1, 2} {
		s, _ := newStream(t, delay)
		got := streamAll(t, s, x, []int{len(x)})
		if len(got) != len(x) {
			t.Fatalf("delay %d: %d samples out, want %d", delay, len(got), len(x))
		}
		var maxErr float64
		for i := range x {
			maxErr = math.Max(maxErr, math.Abs(float64(got[i]-x[i])))
		}
		if maxErr > 1e-4 {
			t.Errorf("delay %d: max error %g against the input, want <= 1e-4", delay, maxErr)
		}
		if want := (testFrame - testHop) + delay*testHop; s.Latency() != want {
			t.Errorf("delay %d: Latency() = %d, want %d", delay, s.Latency(), want)
		}
	}
}

// TestStreamCountsAndBounds pins the counting contract across transitions: each
// ProcessInto returns exactly the pending count and no more than MaxOutputLen,
// totals match after flush, a rejected call runs no hook and changes nothing,
// and the output is bit-identical whatever the chunking, across a second stream
// on the same Stream and a stream after a mid-stream Reset.
func TestStreamCountsAndBounds(t *testing.T) {
	x := testSignal(3000, 3)
	for _, delay := range []int{0, 1, 2} {
		s, h := newStream(t, delay)
		ref := streamAll(t, s, x, []int{len(x)})
		if len(ref) != len(x) {
			t.Fatalf("delay %d: %d samples out, want %d", delay, len(ref), len(x))
		}
		for name, chunks := range chunkings(len(x)) {
			if got := streamAll(t, s, x, chunks); !slices.Equal(got, ref) {
				t.Errorf("delay %d %s: output differs from the whole-input stream", delay, name)
			}
		}

		// Reset mid-stream, then a full stream.
		if _, err := s.Process(x[:1000]); err != nil {
			t.Fatal(err)
		}
		s.Reset()
		if got := streamAll(t, s, x, []int{300, 300, 5000}); !slices.Equal(got, ref) {
			t.Errorf("delay %d: stream after a mid-stream Reset differs from a fresh stream", delay)
		}

		// A ProcessInto one sample short consumes nothing and runs no hook.
		if _, err := s.Process(x[:2000]); err != nil {
			t.Fatal(err)
		}
		in := x[2000:3000]
		need := s.pending(len(in))
		if need == 0 {
			t.Fatalf("delay %d: test input yields no output, cannot exercise ErrBufferTooSmall", delay)
		}
		calls, totalIn, frames := len(h.idx), s.totalIn, s.frames
		n, err := s.ProcessInto(in, make([]float32, need-1))
		if n != 0 || !errors.Is(err, dsp.ErrBufferTooSmall) {
			t.Fatalf("delay %d: ProcessInto(short) = (%d, %v), want (0, ErrBufferTooSmall)", delay, n, err)
		}
		if len(h.idx) != calls || s.totalIn != totalIn || s.frames != frames || s.pending(len(in)) != need {
			t.Errorf("delay %d: rejected ProcessInto changed the stream", delay)
		}
		if n, err = s.ProcessInto(in, make([]float32, need)); n != need || err != nil {
			t.Fatalf("delay %d: sized retry = (%d, %v), want (%d, nil)", delay, n, err, need)
		}

		// A FlushInto one sample short does nothing; the sized retry succeeds.
		rem := int(s.totalIn - s.totalOut)
		if rem == 0 {
			t.Fatalf("delay %d: nothing left to flush, cannot exercise ErrBufferTooSmall", delay)
		}
		calls, resets := len(h.idx), h.resets
		n, err = s.FlushInto(make([]float32, rem-1))
		if n != 0 || !errors.Is(err, dsp.ErrBufferTooSmall) {
			t.Fatalf("delay %d: FlushInto(short) = (%d, %v), want (0, ErrBufferTooSmall)", delay, n, err)
		}
		if len(h.idx) != calls || h.resets != resets || int(s.totalIn-s.totalOut) != rem {
			t.Errorf("delay %d: rejected FlushInto changed the stream", delay)
		}
		if n, err = s.FlushInto(make([]float32, rem)); n != rem || err != nil {
			t.Fatalf("delay %d: sized FlushInto retry = (%d, %v), want (%d, nil)", delay, n, err, rem)
		}
		if h.resets != resets+1 {
			t.Errorf("delay %d: FlushInto ran the reset hook %d times, want 1", delay, h.resets-resets)
		}
	}
}

// TestStreamEmptyFlush pins that flushing a fresh stream writes nothing and
// still resets, so a block can be flushed unconditionally.
func TestStreamEmptyFlush(t *testing.T) {
	s, h := newStream(t, 1)
	n, err := s.FlushInto(nil)
	if n != 0 || err != nil {
		t.Fatalf("FlushInto on a fresh stream = (%d, %v), want (0, nil)", n, err)
	}
	if h.resets != 1 || len(h.idx) != 0 {
		t.Errorf("fresh FlushInto: resets=%d hook calls=%d, want 1 and 0", h.resets, len(h.idx))
	}
}
