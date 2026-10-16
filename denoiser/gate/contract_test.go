package gate

import (
	"errors"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestMaxOutputLenGuarantee checks the contract promise for every lookahead: an
// out buffer sized with MaxOutputLen never yields ErrBufferTooSmall across chunk
// sizes and stream fill states, and a Latency()+HopSize() buffer always holds
// the FlushInto tail. MaxOutputLen is the input length plus one hop.
func TestMaxOutputLenGuarantee(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	x := whiteNoise(4000, dbToLin(-30), 21) // crosses many frames
	for _, tsf := range []int{1, 3, 5} {    // L = 0, 1, 2
		p := ParamsFor(denoiser.Medium)
		p.TimeSmoothFrames = tsf
		g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
		if err != nil {
			t.Fatal(err)
		}
		if got, want := g.MaxOutputLen(1000), 1000+g.HopSize(); got != want {
			t.Errorf("TimeSmoothFrames=%d: MaxOutputLen(1000) = %d, want %d", tsf, got, want)
		}
		for chunk := 1; chunk <= 1500; chunk += 97 {
			g.Reset()
			for i := 0; i < len(x); i += chunk {
				in := x[i:min(i+chunk, len(x))]
				out := make([]float32, g.MaxOutputLen(len(in)))
				if _, err := g.ProcessInto(in, out); err != nil {
					t.Fatalf("TimeSmoothFrames=%d chunk %d at %d: ProcessInto with MaxOutputLen buffer: %v", tsf, chunk, i, err)
				}
			}
			flush := make([]float32, g.Latency()+g.HopSize())
			if _, err := g.FlushInto(flush); err != nil {
				t.Fatalf("TimeSmoothFrames=%d chunk %d: FlushInto with Latency()+HopSize() buffer: %v", tsf, chunk, err)
			}
		}
	}
}

// TestFlushAndReuseBitExact pins that Flush and Reset return the whole block to
// its start state: a second stream after Flush, and a stream after a mid-stream
// Reset, are bit-identical to a fresh block's. The learned floor leaves masks
// below 1, so a mask ring left unprimed changes the start of the next stream;
// the blind clip is longer than a quarter of FloorWindowSec so the tracker has
// published a floor that a skipped tracker reset would carry over.
func TestFlushAndReuseBitExact(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	x := whiteNoise(20000, dbToLin(-30), 31)
	tn := tone(len(x), sr, 3000, dbToLin(-18))
	for i := range x {
		x[i] += tn[i]
	}
	noise := whiteNoise(frame*8, dbToLin(-30), 32)
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}
	for _, m := range []struct {
		name  string
		setup func(*Gate)
	}{
		{modeBlind, nil},
		{modeLearned, func(g *Gate) {
			if err := g.LearnNoise(noise); err != nil {
				t.Fatalf("LearnNoise: %v", err)
			}
		}},
	} {
		fresh := func() *Gate {
			g, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if m.setup != nil {
				m.setup(g)
			}
			return g
		}
		ref := streamAll(t, fresh(), x)

		g := fresh()
		assertBitExact(t, m.name+": first stream vs fresh", streamAll(t, g, x), ref)
		assertBitExact(t, m.name+": second stream after Flush vs fresh", streamAll(t, g, x), ref)
		if _, err := g.Process(x[:9000]); err != nil {
			t.Fatal(err)
		}
		g.Reset()
		assertBitExact(t, m.name+": stream after mid-stream Reset vs fresh", streamAll(t, g, x), ref)
	}
}

// TestErrBufferTooSmallIsShared checks the gate's undersize error chains to the
// shared dsp sentinel, so a consumer can test every block uniformly.
func TestErrBufferTooSmallIsShared(t *testing.T) {
	g, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	_, err = g.ProcessInto(whiteNoise(4096, dbToLin(-30), 41), make([]float32, 1))
	if !errors.Is(err, dsp.ErrBufferTooSmall) {
		t.Fatalf("ProcessInto err = %v, want dsp.ErrBufferTooSmall", err)
	}
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("ProcessInto err = %v, want gate.ErrBufferTooSmall", err)
	}
}

// TestErrInvalidConfigIsShared checks the gate's construction error chains to
// the shared dsp sentinel, so a consumer can test any block's config failure
// uniformly.
func TestErrInvalidConfigIsShared(t *testing.T) {
	_, err := New(Config{SampleRate: 0}) // SampleRate must be > 0
	if !errors.Is(err, dsp.ErrInvalidConfig) {
		t.Fatalf("New(bad config) err = %v, want dsp.ErrInvalidConfig", err)
	}
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("New(bad config) err = %v, want gate.ErrInvalidConfig", err)
	}
}
