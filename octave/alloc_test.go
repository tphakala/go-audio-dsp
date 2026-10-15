//go:build !race

// The race detector adds allocations to instrumented code, so these
// zero-allocation assertions are only meaningful without -race.

package octave

import "testing"

func TestProcessZeroAlloc(t *testing.T) {
	const rate = 48000
	m := mustMeter(t, Config{SampleRate: rate, PeriodSamples: rate / 10})
	in := noise(rate/2, 1, 0.5)
	var sum float64
	var emits int
	emit := func(ms []float64) { sum += ms[0]; emits++ }
	m.Process(in, emit) // warm up and confirm the call does real work
	if emits == 0 || sum == 0 {
		t.Fatalf("warm-up emitted %d periods, sum %v", emits, sum)
	}
	if a := testing.AllocsPerRun(20, func() { m.Process(in, emit) }); a != 0 {
		t.Errorf("Process allocated %v times, want 0", a)
	}
}
