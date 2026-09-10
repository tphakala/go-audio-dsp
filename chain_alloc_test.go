//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. This zero-allocation assertion is
// therefore only meaningful without -race.

package dsp_test

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/equalizer"
	"github.com/tphakala/go-audio-dsp/gain"
	"github.com/tphakala/go-audio-dsp/pcm"
)

// TestChainSteadyStateZeroAlloc checks that a full decode -> denoise -> equalize
// -> gain -> encode chain, run over reused buffers, allocates nothing per chunk
// in steady state (past the denoiser's initial latency).
func TestChainSteadyStateZeroAlloc(t *testing.T) {
	const (
		sr    = 48000
		chunk = 4800
	)
	d, err := denoiser.New(denoiser.Config{SampleRate: sr})
	if err != nil {
		t.Fatal(err)
	}
	eq, err := equalizer.New(equalizer.Config{SampleRate: sr, Bands: []equalizer.Band{
		{Type: equalizer.HighPass, Frequency: 80, Q: 0.7071},
		{Type: equalizer.Peaking, Frequency: 3000, WidthHz: 800, GainDB: 3},
	}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := gain.New(-2)
	if err != nil {
		t.Fatal(err)
	}

	inBytes := make([]byte, chunk*2)
	for i := range inBytes {
		inBytes[i] = byte(i * 3)
	}
	decoded := make([]float32, chunk)
	work := make([]float32, d.MaxOutputLen(chunk))
	outBytes := make([]byte, len(work)*2)

	step := func() {
		if _, err := pcm.BytesToFloat32(decoded, inBytes); err != nil {
			t.Fatal(err)
		}
		m, err := d.ProcessInto(decoded, work)
		if err != nil {
			t.Fatal(err)
		}
		seg := work[:m]
		if _, err := eq.ProcessInto(seg, seg); err != nil {
			t.Fatal(err)
		}
		if _, err := g.ProcessInto(seg, seg); err != nil {
			t.Fatal(err)
		}
		if _, err := pcm.Float32ToBytes(outBytes[:m*2], seg); err != nil {
			t.Fatal(err)
		}
	}

	step() // warm past the denoiser's initial latency
	step()
	if a := testing.AllocsPerRun(50, step); a != 0 {
		t.Errorf("chain step allocated %v times, want 0", a)
	}
}
