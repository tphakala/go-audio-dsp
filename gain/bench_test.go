package gain

import "testing"

func BenchmarkProcessInto(b *testing.B) {
	g, err := New(6)
	if err != nil {
		b.Fatal(err)
	}
	in := make([]float32, 4800)
	out := make([]float32, 4800)
	for i := range in {
		in[i] = float32(i%200-100) / 100
	}
	b.SetBytes(int64(len(in) * 4))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := g.ProcessInto(in, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyInt16(b *testing.B) {
	g, err := New(6)
	if err != nil {
		b.Fatal(err)
	}
	s := make([]int16, 4800)
	for i := range s {
		s[i] = int16(i%2000 - 1000)
	}
	b.SetBytes(int64(len(s) * 2))
	b.ReportAllocs()
	for b.Loop() {
		g.ApplyInt16(s)
	}
}
