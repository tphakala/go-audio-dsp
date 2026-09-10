package denoiser_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// rms returns the root-mean-square level of x.
func rms(x []float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)))
}

// noisyTone builds seconds of a 1 kHz tone buried in white noise at 48 kHz. A
// fixed seed keeps the samples reproducible so the example output is stable.
func noisyTone(seconds int) []float32 {
	rng := rand.New(rand.NewSource(1))
	n := 48000 * seconds
	x := make([]float32, n)
	for i := range x {
		tone := 0.1 * math.Sin(2*math.Pi*1000*float64(i)/48000)
		noise := 0.2 * (2*rng.Float64() - 1)
		x[i] = float32(tone + noise)
	}
	return x
}

// Denoise a whole clip held in memory. It measures the noise profile from the
// clip's quietest window and falls back to an adaptive tracker when there is no
// distinct quiet region, so the caller supplies nothing but the audio and a
// sample rate.
func Example() {
	x := noisyTone(1)

	out, err := denoiser.Denoise(x, denoiser.Config{SampleRate: 48000})
	if err != nil {
		panic(err)
	}

	fmt.Printf("samples in/out: %d/%d\n", len(x), len(out))
	fmt.Printf("noise reduced:  %t\n", rms(out) < rms(x))
	// Output:
	// samples in/out: 48000/48000
	// noise reduced:  true
}

// Stream audio through a Denoiser in arbitrary chunks and Flush at the end. The
// output is aligned sample-for-sample with the input and lags by Latency()
// during the stream; Flush returns the tail so total output equals total input.
func Example_streaming() {
	d, err := denoiser.New(denoiser.Config{SampleRate: 48000, Strength: denoiser.Light})
	if err != nil {
		panic(err)
	}

	x := noisyTone(1)
	var total int
	for off := 0; off < len(x); off += 4000 { // feed 4000-sample chunks
		out, err := d.Process(x[off:min(off+4000, len(x))])
		if err != nil {
			panic(err)
		}
		total += len(out)
	}
	tail, err := d.Flush()
	if err != nil {
		panic(err)
	}
	total += len(tail)

	fmt.Printf("latency:        %d samples\n", d.Latency())
	fmt.Printf("samples in/out: %d/%d\n", len(x), total)
	// Output:
	// latency:        768 samples
	// samples in/out: 48000/48000
}

// When a noise-only excerpt is available (for example a region the user marked
// on a spectrogram), pass it to DenoiseWithNoise to measure the profile
// directly instead of searching the clip for a quiet window.
func ExampleDenoiseWithNoise() {
	noise := noisyTone(1)[:24000] // pretend the first half-second is noise only
	clip := noisyTone(2)

	out, err := denoiser.DenoiseWithNoise(clip, noise, denoiser.Config{SampleRate: 48000})
	if err != nil {
		panic(err)
	}

	fmt.Printf("samples in/out: %d/%d\n", len(clip), len(out))
	fmt.Printf("noise reduced:  %t\n", rms(out) < rms(clip))
	// Output:
	// samples in/out: 96000/96000
	// noise reduced:  true
}

// A measured NoiseProfile can be saved and restored: Spectrum returns the
// per-bin noise power and NewNoiseProfile rebuilds an equivalent profile from
// it, so a profile can be cached across runs. The rebuilt profile reports its
// source as "external".
func ExampleNewNoiseProfile() {
	d, err := denoiser.New(denoiser.Config{SampleRate: 48000})
	if err != nil {
		panic(err)
	}

	measured, err := d.NoiseProfileFromSamples(noisyTone(1))
	if err != nil {
		panic(err)
	}

	spectrum := measured.Spectrum() // persist these floats, then later:
	restored, err := denoiser.NewNoiseProfile(spectrum)
	if err != nil {
		panic(err)
	}

	fmt.Printf("measured source: %s\n", measured.Info().Source)
	fmt.Printf("restored source: %s\n", restored.Info().Source)
	fmt.Printf("bins:            %d\n", len(spectrum))
	// Output:
	// measured source: samples
	// restored source: external
	// bins:            513
}
