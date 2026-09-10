package equalizer

import (
	"fmt"
	"testing"
)

// BenchmarkProcessInto measures the scalar cascade at several section counts (a
// single peaking band cascaded via Passes), for the steady-state streaming path.
func BenchmarkProcessInto(b *testing.B) {
	const block = 4800 // 100 ms at 48 kHz
	in := eqSignal(block)
	out := make([]float32, block)
	for _, sections := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("sections=%d", sections), func(b *testing.B) {
			e, err := New(Config{SampleRate: 48000, Bands: []Band{
				{Type: Peaking, Frequency: 1000, WidthHz: 400, GainDB: 6, Passes: sections},
			}})
			if err != nil {
				b.Fatal(err)
			}
			if _, err := e.ProcessInto(in, out); err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(block * 4))
			b.ReportAllocs()
			for b.Loop() {
				if _, err := e.ProcessInto(in, out); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkResponse(b *testing.B) {
	e, err := New(testConfig())
	if err != nil {
		b.Fatal(err)
	}
	freqs, err := LogSweep(20, 20000, 512)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = e.Response(freqs)
	}
}
