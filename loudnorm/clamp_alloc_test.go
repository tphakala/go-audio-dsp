//go:build !race

// The race detector adds allocations to instrumented code, so these
// zero-allocation assertions are only meaningful without -race.

package loudnorm

import (
	"encoding/binary"
	"testing"
)

// TestPlanClampedGainBytesZeroAlloc checks the plan path allocates nothing in
// steady state on little-endian hosts: it measures the bytes in place and
// returns value types, so it needs no scratch buffer and no boxing.
func TestPlanClampedGainBytesZeroAlloc(t *testing.T) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		t.Skip("zero-copy byte path is little-endian only")
	}
	opts := DefaultOptions()
	b := bytesLE(sineInt16(-20, 1000, 1.0, 48000))
	// Warm the meter pool and confirm the call does real work: a version that
	// errored out early would allocate nothing and pass this check falsely.
	if _, res, _, err := PlanClampedGainBytes(b, opts, DefaultMaxGainDB); err != nil || res.GainDB == 0 {
		t.Fatalf("warm-up: res.GainDB=%v err=%v, want non-zero gain and nil", res.GainDB, err)
	}
	if a := testing.AllocsPerRun(20, func() {
		_, _, _, _ = PlanClampedGainBytes(b, opts, DefaultMaxGainDB)
	}); a != 0 {
		t.Errorf("PlanClampedGainBytes allocated %v times, want 0", a)
	}
}
