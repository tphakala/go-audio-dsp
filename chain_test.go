package dsp_test

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/equalizer"
	"github.com/tphakala/go-audio-dsp/gain"
	"github.com/tphakala/go-audio-dsp/pcm"
)

// BenchmarkChain measures the steady-state cost of the full decode -> denoise ->
// equalize -> gain -> encode chain over reused buffers.
func BenchmarkChain(b *testing.B) {
	const (
		sr    = 48000
		chunk = 4800 // 100 ms
	)
	d, err := denoiser.New(denoiser.Config{SampleRate: sr})
	if err != nil {
		b.Fatal(err)
	}
	eq, err := equalizer.New(equalizer.Config{SampleRate: sr, Bands: []equalizer.Band{
		{Type: equalizer.HighPass, Frequency: 80, Q: 0.7071},
		{Type: equalizer.Peaking, Frequency: 3000, WidthHz: 800, GainDB: 3},
	}})
	if err != nil {
		b.Fatal(err)
	}
	g, err := gain.New(-2)
	if err != nil {
		b.Fatal(err)
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
			b.Fatal(err)
		}
		m, err := d.ProcessInto(decoded, work)
		if err != nil {
			b.Fatal(err)
		}
		seg := work[:m]
		if _, err := eq.ProcessInto(seg, seg); err != nil {
			b.Fatal(err)
		}
		if _, err := g.ProcessInto(seg, seg); err != nil {
			b.Fatal(err)
		}
		if _, err := pcm.Float32ToBytes(outBytes[:m*2], seg); err != nil {
			b.Fatal(err)
		}
	}
	step() // warm up

	b.SetBytes(int64(chunk * 2))
	b.ReportAllocs()
	for b.Loop() {
		step()
	}
}
