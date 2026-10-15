//go:build !race

// The race detector adds allocations to instrumented code, so these
// zero-allocation assertions are only meaningful without -race.

package loudnorm

import (
	"testing"
)

// TestPlanBytesZeroAlloc checks PlanBytes allocates nothing in steady state on
// the normal path and on the gate-lifted path (which re-measures into the
// pooled scratch rather than copying the clip).
func TestPlanBytesZeroAlloc(t *testing.T) {
	ctx := t.Context()
	for _, c := range []struct {
		name string
		dbfs float64
		lift bool
	}{
		{"measurable", audibleDBFS, false},
		{"gate-lifted", subGateDBFS, true},
	} {
		b := tonePCM(c.dbfs, 2, planTestRate, 1)
		opts := planOpts(planTestRate, 1, true)
		// Warm the pools and confirm the call does real work: a version that
		// errored out early would allocate nothing and pass this falsely.
		plan, err := PlanBytes(ctx, b, opts)
		if err != nil || plan.GateLifted != c.lift || plan.GainDB == 0 {
			t.Fatalf("%s warm-up: %+v, %v", c.name, plan, err)
		}
		if a := testing.AllocsPerRun(20, func() { _, _ = PlanBytes(ctx, b, opts) }); a != 0 {
			t.Errorf("%s: PlanBytes allocated %v times, want 0", c.name, a)
		}
	}
	b := tonePCM(audibleDBFS, 2, planTestRate, 1)
	if a := testing.AllocsPerRun(20, func() { _, _ = MeasureBytes(b, planTestRate, 1) }); a != 0 {
		t.Errorf("MeasureBytes allocated %v times, want 0", a)
	}
}
