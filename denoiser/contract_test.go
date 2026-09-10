package denoiser

import (
	"errors"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

func TestFrameHopAccessors(t *testing.T) {
	cases := []struct {
		cfg   Config
		frame int
		hop   int
	}{
		{Config{SampleRate: 48000}, 1024, 256},                // auto: ~21.3 ms, 75% overlap
		{Config{SampleRate: 16000}, 256, 64},                  // auto at 16 kHz
		{Config{SampleRate: 48000, FrameSize: 512}, 512, 128}, // explicit frame, auto hop
		{Config{SampleRate: 48000, FrameSize: 512, HopSize: 128}, 512, 128},
	}
	for _, c := range cases {
		d, err := New(c.cfg)
		if err != nil {
			t.Fatalf("New(%+v): %v", c.cfg, err)
		}
		if d.FrameSize() != c.frame {
			t.Errorf("FrameSize() = %d, want %d for %+v", d.FrameSize(), c.frame, c.cfg)
		}
		if d.HopSize() != c.hop {
			t.Errorf("HopSize() = %d, want %d for %+v", d.HopSize(), c.hop, c.cfg)
		}
		if got, want := d.MaxOutputLen(1000), 1000+c.hop; got != want {
			t.Errorf("MaxOutputLen(1000) = %d, want %d for %+v", got, want, c.cfg)
		}
		// Latency() cannot recover HopSize; the accessors must.
		if d.FrameSize()-d.HopSize() != d.Latency() {
			t.Errorf("FrameSize-HopSize = %d, Latency = %d", d.FrameSize()-d.HopSize(), d.Latency())
		}
	}
}

// TestMaxOutputLenGuarantee checks the contract promise: an out buffer sized
// with MaxOutputLen never yields ErrBufferTooSmall, across a range of chunk
// sizes and stream fill states.
func TestMaxOutputLenGuarantee(t *testing.T) {
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	x := testTone(4000) // enough to cross several frames
	for chunk := 1; chunk <= 1500; chunk += 97 {
		d.Reset()
		for i := 0; i < len(x); i += chunk {
			end := min(i+chunk, len(x))
			in := x[i:end]
			out := make([]float32, d.MaxOutputLen(len(in)))
			if _, err := d.ProcessInto(in, out); err != nil {
				t.Fatalf("chunk %d at %d: ProcessInto with MaxOutputLen buffer: %v", chunk, i, err)
			}
		}
		flush := make([]float32, d.FrameSize())
		if _, err := d.FlushInto(flush); err != nil {
			t.Fatalf("chunk %d: FlushInto with FrameSize buffer: %v", chunk, err)
		}
	}
}

// TestErrBufferTooSmallIsShared checks the denoiser's undersize error chains to
// the shared dsp sentinel, so a consumer can test every block uniformly.
func TestErrBufferTooSmallIsShared(t *testing.T) {
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.ProcessInto(testTone(2048), make([]float32, 1))
	if !errors.Is(err, dsp.ErrBufferTooSmall) {
		t.Fatalf("ProcessInto err = %v, want dsp.ErrBufferTooSmall", err)
	}
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("ProcessInto err = %v, want denoiser.ErrBufferTooSmall", err)
	}
}
