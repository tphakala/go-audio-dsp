// Command streaming demonstrates a full DSP chain over the streaming contract:
// interleaved little-endian int16 PCM bytes are decoded to float32, denoised,
// equalized and gained, then re-encoded to int16 bytes, all over reused buffers
// so the steady state allocates nothing.
package main

import (
	"fmt"
	"math"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/equalizer"
	"github.com/tphakala/go-audio-dsp/gain"
	"github.com/tphakala/go-audio-dsp/pcm"
)

const sampleRate = 48000

// RunChain streams input (interleaved little-endian int16 PCM) through a
// denoiser, an equalizer and a gain block in fixed chunks, returning the
// processed PCM. The output has the same number of samples as the input: the
// denoiser is drained with FlushInto at the end, and the equalizer and gain are
// zero-latency.
func RunChain(input []byte, chunkSamples int) ([]byte, error) {
	if chunkSamples <= 0 {
		return nil, fmt.Errorf("chunkSamples must be > 0, got %d", chunkSamples)
	}
	if len(input)%2 != 0 {
		return nil, pcm.ErrOddByteLength // whole int16 samples only
	}
	d, err := denoiser.New(denoiser.Config{SampleRate: sampleRate})
	if err != nil {
		return nil, err
	}
	eq, err := equalizer.New(equalizer.Config{SampleRate: sampleRate, Bands: []equalizer.Band{
		{Type: equalizer.HighPass, Frequency: 80, Q: 0.7071},
		{Type: equalizer.Peaking, Frequency: 3000, WidthHz: 800, GainDB: 3},
	}})
	if err != nil {
		return nil, err
	}
	g, err := gain.New(-2)
	if err != nil {
		return nil, err
	}

	// Reused scratch: decode target, the denoiser's output (also the in-place
	// buffer for the equalizer and gain), and the byte encode target. work must
	// hold both a chunk's ProcessInto output and the FlushInto tail (up to
	// FrameSize), so size it to the larger of the two.
	decoded := make([]float32, chunkSamples)
	work := make([]float32, max(d.MaxOutputLen(chunkSamples), d.FrameSize()))
	encoded := make([]byte, len(work)*2)
	out := make([]byte, 0, len(input))

	// emit runs the m samples in work[:m] through eq -> gain, encodes them to
	// little-endian int16 bytes and appends them to out. m may be 0 (a no-op), so
	// the per-chunk and flush paths share one encode-and-append routine.
	emit := func(m int) error {
		if m == 0 {
			return nil
		}
		seg := work[:m]
		if _, err := eq.ProcessInto(seg, seg); err != nil { // in place
			return err
		}
		if _, err := g.ProcessInto(seg, seg); err != nil { // in place
			return err
		}
		nb, err := pcm.Float32ToBytes(encoded[:m*2], seg)
		if err != nil {
			return err
		}
		out = append(out, encoded[:nb*2]...) // nb is samples; two bytes each
		return nil
	}

	// step runs one block of decoded samples through the denoiser, then emits.
	step := func(samples []float32) error {
		m, err := d.ProcessInto(samples, work)
		if err != nil {
			return err
		}
		return emit(m)
	}

	chunkBytes := chunkSamples * 2
	for off := 0; off < len(input); off += chunkBytes {
		end := min(off+chunkBytes, len(input))
		ns := (end - off) / 2
		if _, err := pcm.BytesToFloat32(decoded[:ns], input[off:off+ns*2]); err != nil {
			return nil, err
		}
		if err := step(decoded[:ns]); err != nil {
			return nil, err
		}
	}

	// Drain the denoiser's tail through the rest of the chain.
	m, err := d.FlushInto(work)
	if err != nil {
		return nil, err
	}
	if err := emit(m); err != nil {
		return nil, err
	}
	return out, nil
}

// synthClip returns a 2 s noisy tone as interleaved little-endian int16 bytes.
func synthClip() []byte {
	const n = sampleRate * 2
	f := make([]float32, n)
	for i := range f {
		t := float64(i) / sampleRate
		tone := 0.3 * math.Sin(2*math.Pi*1000*t)
		hiss := 0.02 * math.Sin(float64(i)*12.9898)
		f[i] = float32(tone + hiss)
	}
	b := make([]byte, n*2)
	if _, err := pcm.Float32ToBytes(b, f); err != nil {
		panic(err)
	}
	return b
}

func main() {
	in := synthClip()
	out, err := RunChain(in, 4800) // 100 ms chunks
	if err != nil {
		panic(err)
	}
	fmt.Printf("in %d bytes, out %d bytes\n", len(in), len(out))
}
