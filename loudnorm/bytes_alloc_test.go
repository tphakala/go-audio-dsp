//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package loudnorm

import (
	"encoding/binary"
	"testing"
)

// TestBytesZeroAlloc checks the byte entry points allocate nothing in steady
// state on little-endian hosts: the samples are read and written in place. The
// buffer is refilled by copy (no allocation) each iteration so the gain
// write-back runs every time.
func TestBytesZeroAlloc(t *testing.T) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		t.Skip("zero-copy byte path is little-endian only")
	}
	opts := DefaultOptions()
	src := bytesLE(sineInt16(-20, 1000, 1.0, 48000))
	scratch := make([]byte, len(src))
	copy(scratch, src)
	// Warm the meter pool and confirm the calls actually do their work: a version
	// that errored out early would allocate nothing and pass this check falsely.
	if r, err := NormalizeBytes(scratch, opts); err != nil || r.GainDB == 0 {
		t.Fatalf("NormalizeBytes warm-up: gain=%v err=%v, want non-zero gain and nil", r.GainDB, err)
	}
	if _, err := MeasureBytes(src, 48000, 1); err != nil {
		t.Fatalf("MeasureBytes warm-up: %v", err)
	}

	if a := testing.AllocsPerRun(20, func() {
		copy(scratch, src)
		_, _ = NormalizeBytes(scratch, opts)
	}); a != 0 {
		t.Errorf("NormalizeBytes allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(20, func() {
		_, _ = MeasureBytes(src, 48000, 1)
	}); a != 0 {
		t.Errorf("MeasureBytes allocated %v times, want 0", a)
	}
}
