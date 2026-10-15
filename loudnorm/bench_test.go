package loudnorm

import "testing"

// A 3-second 48 kHz mono clip, the typical BirdNET-Go export size.
func benchClipInt16() []int16 { return sineInt16(-12, 1000, 3, 48000) }

func BenchmarkMeasureInt16Mono48k(b *testing.B) {
	pcm := benchClipInt16()
	b.SetBytes(int64(len(pcm) * 2))
	for b.Loop() {
		if _, err := MeasureInt16(pcm, 48000, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMeterReuseMono48k measures the explicit zero-allocation reuse path: a
// single meter reset and reused across clips.
func BenchmarkMeterReuseMono48k(b *testing.B) {
	pcm := benchClipInt16()
	m := NewMeter(48000, 1)
	b.SetBytes(int64(len(pcm) * 2))
	for b.Loop() {
		m.Reset()
		m.AddInt16(pcm)
		_ = m.IntegratedLoudness()
		_ = m.TruePeakDBTP()
	}
}

func BenchmarkNormalizeInt16Mono48k(b *testing.B) {
	src := benchClipInt16()
	pcm := make([]int16, len(src))
	opts := DefaultOptions()
	b.SetBytes(int64(len(pcm) * 2))
	for b.Loop() {
		copy(pcm, src) // fresh buffer each iteration (normalize mutates in place)
		if _, err := NormalizeInt16(pcm, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPlanBytes plans a 5-minute 48 kHz mono clip (28.8 MB of int16 bytes),
// the shape of a long field recording. The sub-gate case pays the second,
// scaled measurement pass of the gate fallback.
func BenchmarkPlanBytes(b *testing.B) {
	for _, c := range []struct {
		name string
		dbfs float64
	}{{"measurable", -30}, {"gate-lifted", -80}} {
		b.Run(c.name, func(b *testing.B) {
			pcm := bytesLE(sineInt16(c.dbfs, 1000, 300, 48000))
			opts := PlanOptions{Options: DefaultOptions(), MaxGainDB: 60, GateFallback: true}
			b.SetBytes(int64(len(pcm)))
			for b.Loop() {
				if _, err := PlanBytes(b.Context(), pcm, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
