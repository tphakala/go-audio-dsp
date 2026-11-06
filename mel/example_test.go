package mel

import (
	"fmt"
	"slices"
	"sync"

	"github.com/tphakala/go-audio-dsp/stft"
)

// ExampleExtractor computes a whole-clip log-mel matrix and reports its shape.
func ExampleExtractor() {
	ex, err := New(Config{
		SampleRate: 16000, FrameSize: 512, HopSize: 128, NumMels: 40,
		MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	})
	if err != nil {
		panic(err)
	}
	sig := toneSig(4096, 16000, []float64{1000}, 0.8)
	m := ex.Compute(sig, stft.NoPad)
	fmt.Printf("%d mels x %d frames\n", m.Mels, m.Frames)
	// Output:
	// 40 mels x 29 frames
}

// ExampleColumnSource streams the same audio and counts the emitted columns.
func ExampleColumnSource() {
	cs, err := NewColumnSource(Config{
		SampleRate: 16000, FrameSize: 512, HopSize: 128, NumMels: 40,
		MinHz: 0, MaxHz: 0, Input: InputPower, Log: Log10, LogOffset: 1e-6,
	})
	if err != nil {
		panic(err)
	}
	sig := toneSig(4096, 16000, []float64{1000}, 0.8)
	columns := 0
	cs.Feed(sig, func(col []float32, centerSample int64) {
		columns++
	})
	fmt.Printf("%d columns of %d mels\n", columns, cs.NumMels())
	// Output:
	// 29 columns of 40 mels
}

// ExampleExtractor_parallel computes one signed real-part, power-law matrix
// serially and again with four goroutines. Each worker owns an Extractor (an
// Extractor is not safe for concurrent use) that shares one immutable Filterbank,
// and ComputeInto on a frame-aligned sub-slice of the signal under NoPad writes
// exactly those columns: frame f spans signal[f*hop : f*hop+FrameSize), so frames
// [f0, f0+c) need signal[f0*hop : (f0+c-1)*hop+FrameSize).
func ExampleExtractor_parallel() {
	const (
		sr, n, hop, mels = 48000, 1024, 280, 96
		lo, hi           = 11, 320
	)
	// A synthetic bank of 96 triangles over bins [lo, hi).
	rows := make([][]float32, mels)
	for m := range rows {
		rows[m] = make([]float32, n/2+1)
		start := lo + m*(hi-lo-6)/mels
		for k := range 6 {
			rows[m][start+k] = 1 - float32(k)/6
		}
	}
	fb, err := FilterbankFromRows(rows)
	if err != nil {
		panic(err)
	}
	cfg := Config{
		SampleRate: sr, FrameSize: n, HopSize: hop, Filterbank: fb,
		Input: InputReal, Log: PowerLaw, PowerExponent: 0.4,
	}
	signal := toneSig(3*sr, sr, []float64{1200, 4300}, 0.4)

	ex, err := New(cfg)
	if err != nil {
		panic(err)
	}
	frames := ex.NumFrames(len(signal), stft.NoPad)
	serial := ex.Compute(signal, stft.NoPad)

	whole := Matrix{Data: make([]float32, frames*mels)}
	var wg sync.WaitGroup
	f0 := 0
	for _, c := range []int{128, 128, 128, frames - 3*128} {
		wg.Add(1)
		go func(f0, c int) {
			defer wg.Done()
			w, err := New(cfg)
			if err != nil {
				panic(err)
			}
			part := Matrix{Data: whole.Data[f0*mels : (f0+c)*mels : (f0+c)*mels]}
			sub := signal[f0*hop : (f0+c-1)*hop+n]
			if _, err := w.ComputeInto(&part, sub, stft.NoPad); err != nil {
				panic(err)
			}
		}(f0, c)
		f0 += c
	}
	wg.Wait()
	fmt.Printf("%d frames x %d mels, parallel equals serial: %v\n", frames, mels, slices.Equal(whole.Data, serial.Data))
	// Output:
	// 511 frames x 96 mels, parallel equals serial: true
}
