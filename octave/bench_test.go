package octave

import "testing"

func benchProcess(b *testing.B, rate, order int) {
	b.Helper()
	m, err := New(Config{SampleRate: rate, PeriodSamples: rate, Order: order})
	if err != nil {
		b.Fatal(err)
	}
	in := noise(rate/10, 1, 0.5) // 100 ms chunks, the shape of a streaming consumer
	b.SetBytes(int64(len(in) * 4))
	emit := func([]float64) {}
	for b.Loop() {
		m.Process(in, emit)
	}
}

func BenchmarkProcess48k(b *testing.B)       { benchProcess(b, 48000, 1) }
func BenchmarkProcess256k(b *testing.B)      { benchProcess(b, 256000, 1) }
func BenchmarkProcess48kOrder3(b *testing.B) { benchProcess(b, 48000, 3) }
