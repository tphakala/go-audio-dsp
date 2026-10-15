package loudnorm

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/gain"
)

const (
	subGateDBFS  = -80.0 // a 1 kHz tone this quiet measures -Inf (under the absolute gate)
	audibleDBFS  = -30.0
	planTestRate = 48000
)

func planOpts(rate, channels int, fallback bool) PlanOptions {
	return PlanOptions{
		Options:      Options{SampleRate: rate, Channels: channels, TargetLUFS: -23, TruePeakDBTP: -2},
		MaxGainDB:    60,
		GateFallback: fallback,
	}
}

// interleave repeats mono into channels interleaved channels.
func interleave(mono []int16, channels int) []int16 {
	out := make([]int16, 0, len(mono)*channels)
	for _, v := range mono {
		for range channels {
			out = append(out, v)
		}
	}
	return out
}

func tonePCM(dbfs, seconds float64, rate, channels int) []byte {
	return bytesLE(interleave(sineInt16(dbfs, 1000, seconds, float64(rate)), channels))
}

// noisePCM is steady uniform noise: sub-gate loudness, low crest factor.
func noisePCM(amp int16, seconds float64, rate int) []byte {
	r := rand.New(rand.NewPCG(1, 2))
	s := make([]int16, int(seconds*float64(rate)))
	for i := range s {
		s[i] = int16(r.IntN(2*int(amp)+1) - int(amp))
	}
	return bytesLE(s)
}

func mustMeasure(t *testing.T, b []byte, rate, channels int) Measurement {
	t.Helper()
	m, err := MeasureBytes(b, rate, channels)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func applyGain(t *testing.T, b []byte, db float64) []byte {
	t.Helper()
	g, err := gain.New(db)
	if err != nil {
		t.Fatal(err)
	}
	out := slices.Clone(b)
	if err := g.ApplyBytes(out); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestPlanBytesMeasurementMatchesMeasureInt16 pins chunk independence: the
// planner's measurement equals an independent one-shot MeasureInt16 on a decoded
// copy, bit for bit, at lengths that sit on, below and across chunk edges and at
// channel counts that do not divide the chunk size (a frame-splitting bug fails
// on 3 and 6 channels).
func TestPlanBytesMeasurementMatchesMeasureInt16(t *testing.T) {
	for _, rate := range []int{8000, 48000, 96000} {
		for _, ch := range []int{1, 2, 3, 6} {
			chunkFrames := planChunkSamples / ch
			for _, frames := range []int{0, 1, 1000, chunkFrames, chunkFrames + 777, 3*chunkFrames + 5} {
				mono := make([]int16, frames)
				for i := range mono {
					mono[i] = int16(8000 * math.Sin(float64(i)*0.05) * math.Sin(float64(i)*0.0003))
				}
				il := interleave(mono, ch)
				want, err := MeasureInt16(il, rate, ch)
				if err != nil {
					t.Fatal(err)
				}
				b := bytesLE(il)
				plan, err := PlanBytes(t.Context(), b, planOpts(rate, ch, false))
				if err != nil {
					t.Fatal(err)
				}
				if plan.Result.Input != want {
					t.Fatalf("rate %d ch %d frames %d: planner measured %+v, one-shot %+v", rate, ch, frames, plan.Result.Input, want)
				}
				if got := mustMeasure(t, b, rate, ch); got != want {
					t.Fatalf("rate %d ch %d frames %d: MeasureBytes %+v, one-shot %+v", rate, ch, frames, got, want)
				}
			}
		}
	}
}

// TestPlanBytesLiftedMeasurementMatchesFullCopy pins the no-copy refinement
// against its copy-based definition: apply the lift to a clone with gain,
// measure, plan. The results must agree exactly.
func TestPlanBytesLiftedMeasurementMatchesFullCopy(t *testing.T) {
	for _, ch := range []int{1, 6} {
		b := tonePCM(subGateDBFS, 5, planTestRate, ch)
		opts := planOpts(planTestRate, ch, true)
		plan, err := PlanBytes(t.Context(), b, opts)
		if err != nil {
			t.Fatal(err)
		}
		meas := mustMeasure(t, b, planTestRate, ch)
		lift, ok := GateFallbackGainDB(meas, opts.TargetLUFS, opts.TruePeakDBTP)
		if !ok || !plan.GateLifted {
			t.Fatalf("ch %d: fixture must take the gate fallback", ch)
		}
		lifted := mustMeasure(t, applyGain(t, b, lift), planTestRate, ch)
		want := lift
		if !math.IsInf(lifted.IntegratedLUFS, -1) {
			want += PlanGain(lifted, opts.Options).GainDB
		}
		if plan.PlannedGainDB != want {
			t.Errorf("ch %d: planned %v, copy-based %v", ch, plan.PlannedGainDB, want)
		}
	}
}

func TestGateFallbackGainDB(t *testing.T) {
	negInf := math.Inf(-1)
	cases := []struct {
		name   string
		meas   Measurement
		want   float64
		wantOK bool
	}{
		{"gate distance binds for a very quiet peak", Measurement{negInf, -60}, -23 - AbsoluteGateLUFS, true},
		{"peak headroom binds for a louder peak", Measurement{negInf, -30}, -2 - -30, true},
		{"true silence is left alone", Measurement{negInf, negInf}, 0, false},
		{"a measurable clip needs no fallback", Measurement{-30, -10}, 0, false},
	}
	for _, c := range cases {
		got, ok := GateFallbackGainDB(c.meas, -23, -2)
		if ok != c.wantOK || math.Abs(got-c.want) > 1e-9 {
			t.Errorf("%s: got (%v, %v), want (%v, %v)", c.name, got, ok, c.want, c.wantOK)
		}
	}
}

func TestPlanBytesGateFallback(t *testing.T) {
	ctx := t.Context()
	opts := planOpts(planTestRate, 1, true)

	t.Run("sub-gate clip is lifted to target", func(t *testing.T) {
		b := tonePCM(subGateDBFS, 5, planTestRate, 1)
		plan, err := PlanBytes(ctx, b, opts)
		if err != nil {
			t.Fatal(err)
		}
		if !plan.GateLifted || plan.Clamped {
			t.Fatalf("GateLifted=%v Clamped=%v, want true, false", plan.GateLifted, plan.Clamped)
		}
		if !math.IsInf(plan.Result.Input.IntegratedLUFS, -1) || plan.Result.GainDB != 0 {
			t.Errorf("Result must be the pre-fallback plan, got %+v", plan.Result)
		}
		got := mustMeasure(t, applyGain(t, b, plan.GainDB), planTestRate, 1)
		if math.Abs(got.IntegratedLUFS-opts.TargetLUFS) > 1 {
			t.Errorf("lifted clip measures %.2f LUFS, want %.0f +/-1", got.IntegratedLUFS, opts.TargetLUFS)
		}
	})

	t.Run("still sub-gate keeps the conservative lift", func(t *testing.T) {
		// 150 ms is shorter than one 400 ms block, so even the lifted clip is
		// unmeasurable and the plan is the lift alone.
		b := tonePCM(subGateDBFS, 0.15, planTestRate, 1)
		plan, err := PlanBytes(ctx, b, opts)
		if err != nil {
			t.Fatal(err)
		}
		meas := mustMeasure(t, b, planTestRate, 1)
		lift, _ := GateFallbackGainDB(meas, opts.TargetLUFS, opts.TruePeakDBTP)
		if !plan.GateLifted || plan.PlannedGainDB != lift {
			t.Errorf("planned %v lifted %v, want the bare lift %v", plan.PlannedGainDB, plan.GateLifted, lift)
		}
	})

	t.Run("measurable clip matches the plain plan", func(t *testing.T) {
		b := tonePCM(audibleDBFS, 5, planTestRate, 1)
		plan, err := PlanBytes(ctx, b, opts)
		if err != nil {
			t.Fatal(err)
		}
		want := PlanGain(mustMeasure(t, b, planTestRate, 1), opts.Options)
		if plan.GateLifted || plan.Clamped || plan.Result != want || plan.GainDB != want.GainDB {
			t.Errorf("plan %+v, want plain %+v", plan, want)
		}
	})

	t.Run("silence is left alone", func(t *testing.T) {
		plan, err := PlanBytes(ctx, make([]byte, planTestRate*2*5), opts)
		if err != nil {
			t.Fatal(err)
		}
		if plan.GateLifted || plan.GainDB != 0 || plan.PlannedGainDB != 0 {
			t.Errorf("silence plan %+v", plan)
		}
	})

	t.Run("empty input is silent", func(t *testing.T) {
		plan, err := PlanBytes(ctx, nil, opts)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := MeasureInt16(nil, planTestRate, 1)
		if plan.Result.Input != want || plan.GainDB != 0 || plan.GateLifted {
			t.Errorf("empty plan %+v, silent measurement %+v", plan, want)
		}
	})

	// The property the loudness bound exists for: a flat sub-gate noise floor
	// (a dead or muted microphone) must never be lifted past the target.
	t.Run("flat hiss never louder than target", func(t *testing.T) {
		b := noisePCM(8, 8, planTestRate)
		meas := mustMeasure(t, b, planTestRate, 1)
		if !math.IsInf(meas.IntegratedLUFS, -1) || math.IsInf(meas.TruePeakDBTP, -1) {
			t.Fatalf("fixture must be sub-gate with a finite true peak, got %+v", meas)
		}
		plan, err := PlanBytes(ctx, b, opts)
		if err != nil {
			t.Fatal(err)
		}
		if plan.GainDB <= 0 {
			t.Fatalf("the clip must still be lifted out of inaudibility, gain %v", plan.GainDB)
		}
		after := mustMeasure(t, applyGain(t, b, plan.GainDB), planTestRate, 1)
		if after.IntegratedLUFS > opts.TargetLUFS+0.5 {
			t.Errorf("lifted to %.2f LUFS, past the %.0f target", after.IntegratedLUFS, opts.TargetLUFS)
		}
		if math.Abs(after.IntegratedLUFS-opts.TargetLUFS) > 0.5 {
			t.Errorf("lifted to %.2f LUFS, should reach %.0f", after.IntegratedLUFS, opts.TargetLUFS)
		}
	})
}

func TestPlanBytesGateFallbackOff(t *testing.T) {
	b := tonePCM(subGateDBFS, 5, planTestRate, 1)
	plan, err := PlanBytes(t.Context(), b, planOpts(planTestRate, 1, false))
	if err != nil {
		t.Fatal(err)
	}
	if plan.GainDB != 0 || plan.PlannedGainDB != 0 || plan.GateLifted || plan.Clamped {
		t.Errorf("sub-gate clip without the fallback must plan nothing, got %+v", plan)
	}
}

func TestPlanBytesClamp(t *testing.T) {
	ctx := t.Context()
	t.Run("quiet measurable clip clamped positive", func(t *testing.T) {
		opts := planOpts(planTestRate, 1, false)
		opts.MaxGainDB = 3
		plan, err := PlanBytes(ctx, tonePCM(-45, 5, planTestRate, 1), opts)
		if err != nil {
			t.Fatal(err)
		}
		if !plan.Clamped || plan.GainDB != 3 || plan.PlannedGainDB <= 3 {
			t.Errorf("plan %+v, want GainDB 3 with a larger pre-clamp gain", plan)
		}
	})
	t.Run("hot clip clamped negative", func(t *testing.T) {
		opts := planOpts(planTestRate, 1, false)
		opts.MaxGainDB = 3
		plan, err := PlanBytes(ctx, tonePCM(-3, 5, planTestRate, 1), opts)
		if err != nil {
			t.Fatal(err)
		}
		if !plan.Clamped || plan.GainDB != -3 || plan.PlannedGainDB >= -3 {
			t.Errorf("plan %+v, want GainDB -3 with a smaller pre-clamp gain", plan)
		}
	})
	t.Run("gate-lifted clip clamped", func(t *testing.T) {
		opts := planOpts(planTestRate, 1, true)
		opts.MaxGainDB = 5
		plan, err := PlanBytes(ctx, tonePCM(subGateDBFS, 5, planTestRate, 1), opts)
		if err != nil {
			t.Fatal(err)
		}
		if !plan.GateLifted || !plan.Clamped || plan.GainDB != 5 || plan.PlannedGainDB <= 5 {
			t.Errorf("plan %+v, want GateLifted and clamped to 5", plan)
		}
	})
}

func TestPlanBytesValidation(t *testing.T) {
	good := tonePCM(audibleDBFS, 1, planTestRate, 1)
	cases := []struct {
		name string
		b    []byte
		mod  func(*PlanOptions)
		want error
	}{
		{"odd length", good[:len(good)-1], nil, ErrOddByteLength},
		{"rate too low", good, func(o *PlanOptions) { o.SampleRate = 100 }, ErrSampleRateTooLow},
		{"zero channels", good, func(o *PlanOptions) { o.Channels = 0 }, ErrInvalidChannels},
		{"length not a multiple of channels", good[:2*(len(good)/2-1)], func(o *PlanOptions) { o.Channels = 2 }, ErrLengthNotMultiple},
		{"target out of range", good, func(o *PlanOptions) { o.TargetLUFS = -80 }, ErrTargetOutOfRange},
		{"ceiling above zero", good, func(o *PlanOptions) { o.TruePeakDBTP = 1 }, ErrCeilingInvalid},
		{"MaxGainDB zero", good, func(o *PlanOptions) { o.MaxGainDB = 0 }, dsp.ErrInvalidConfig},
		{"MaxGainDB negative", good, func(o *PlanOptions) { o.MaxGainDB = -1 }, dsp.ErrInvalidConfig},
		{"MaxGainDB NaN", good, func(o *PlanOptions) { o.MaxGainDB = math.NaN() }, dsp.ErrInvalidConfig},
		{"MaxGainDB Inf", good, func(o *PlanOptions) { o.MaxGainDB = math.Inf(1) }, dsp.ErrInvalidConfig},
	}
	for _, c := range cases {
		opts := planOpts(planTestRate, 1, true)
		if c.mod != nil {
			c.mod(&opts)
		}
		plan, err := PlanBytes(t.Context(), c.b, opts)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
		if plan != (Plan{}) {
			t.Errorf("%s: plan must be zero on error, got %+v", c.name, plan)
		}
	}
}

// countdownCtx reports cancellation from its (limit+1)th Err call on, so a test
// can cancel at an exact point inside the planner without sleeps.
type countdownCtx struct {
	context.Context
	left  int
	calls int
}

func (c *countdownCtx) Err() error {
	c.calls++
	if c.left <= 0 {
		return context.Canceled
	}
	c.left--
	return nil
}

func TestPlanBytesCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	b := tonePCM(audibleDBFS, 1, planTestRate, 1)
	plan, err := PlanBytes(ctx, b, planOpts(planTestRate, 1, true))
	if err != context.Canceled || plan != (Plan{}) { //nolint:errorlint // pins that the context error is returned unwrapped
		t.Errorf("got (%+v, %v), want a zero Plan and context.Canceled itself", plan, err)
	}
	// Cancellation is reported even when the options are also invalid.
	bad := planOpts(100, 0, true)
	if _, err := PlanBytes(ctx, b[:len(b)-1], bad); err != context.Canceled { //nolint:errorlint // pins that the context error is returned unwrapped
		t.Errorf("invalid options on a cancelled ctx: err = %v, want context.Canceled", err)
	}
}

// TestPlanBytesCancelledMidway cancels at every context check, inside the
// measurement and inside the refinement, and checks that the next plan on the
// same config equals a fresh one (a cancelled run leaves the pooled meter clean).
func TestPlanBytesCancelledMidway(t *testing.T) {
	b := tonePCM(subGateDBFS, 4, planTestRate, 1) // several chunks, takes the fallback
	opts := planOpts(planTestRate, 1, true)

	counter := &countdownCtx{Context: t.Context(), left: math.MaxInt}
	want, err := PlanBytes(counter, b, opts)
	if err != nil || !want.GateLifted {
		t.Fatalf("baseline: %+v, %v", want, err)
	}
	if counter.calls < 5 {
		t.Fatalf("fixture too short to cancel midway: %d checks", counter.calls)
	}
	for left := range counter.calls {
		ctx := &countdownCtx{Context: t.Context(), left: left}
		plan, err := PlanBytes(ctx, b, opts)
		if !errors.Is(err, context.Canceled) || plan != (Plan{}) {
			t.Fatalf("left=%d: got (%+v, %v), want a zero Plan and context.Canceled", left, plan, err)
		}
		again, err := PlanBytes(t.Context(), b, opts)
		if err != nil || again != want {
			t.Fatalf("left=%d: plan after a cancelled run = %+v, %v; want %+v", left, again, err, want)
		}
	}
}

func TestPlanBytesDoesNotModifyInput(t *testing.T) {
	for _, dbfs := range []float64{audibleDBFS, subGateDBFS} {
		b := tonePCM(dbfs, 2, planTestRate, 2)
		orig := slices.Clone(b)
		if _, err := PlanBytes(t.Context(), b, planOpts(planTestRate, 2, true)); err != nil {
			t.Fatal(err)
		}
		if _, err := MeasureBytes(b, planTestRate, 2); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(b, orig) {
			t.Errorf("%g dBFS: input was modified", dbfs)
		}
	}
}

// TestMeasureBytesConcurrentReaders runs readers over one shared buffer under
// -race: a measurement that wrote back into the buffer (as pcm.InPlaceInt16 does
// on byte-wise hosts) would be a data race here.
func TestMeasureBytesConcurrentReaders(t *testing.T) {
	b := tonePCM(audibleDBFS, 2, planTestRate, 1)
	want := mustMeasure(t, b, planTestRate, 1)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, err := MeasureBytes(b, planTestRate, 1)
			if err != nil || got != want {
				t.Errorf("concurrent MeasureBytes = %+v, %v; want %+v", got, err, want)
			}
		})
	}
	wg.Wait()
}

func TestPlanBytesConcurrentMixedConfigs(t *testing.T) {
	type cfg struct{ rate, ch int }
	cfgs := []cfg{{48000, 1}, {44100, 2}, {96000, 1}, {48000, 6}, {8000, 3}}
	want := make([]Plan, len(cfgs))
	bufs := make([][]byte, len(cfgs))
	for i, c := range cfgs {
		bufs[i] = tonePCM(subGateDBFS, 2, c.rate, c.ch)
		p, err := PlanBytes(t.Context(), bufs[i], planOpts(c.rate, c.ch, true))
		if err != nil {
			t.Fatal(err)
		}
		want[i] = p
	}
	var wg sync.WaitGroup
	for range 4 {
		for i, c := range cfgs {
			wg.Go(func() {
				got, err := PlanBytes(t.Context(), bufs[i], planOpts(c.rate, c.ch, true))
				if err != nil || got != want[i] {
					t.Errorf("%+v: concurrent plan %+v, %v; want %+v", c, got, err, want[i])
				}
			})
		}
	}
	wg.Wait()
}

// TestMeasureBytesFrameWiderThanChunk covers the one-off scratch for a frame
// with more samples than a chunk holds: the frame must still never be split.
func TestMeasureBytesFrameWiderThanChunk(t *testing.T) {
	const ch = planChunkSamples + 1
	il := make([]int16, 2*ch)
	for i := range il {
		il[i] = int16(i%2000 - 1000)
	}
	want, err := MeasureInt16(il, 8000, ch)
	if err != nil {
		t.Fatal(err)
	}
	if got := mustMeasure(t, bytesLE(il), 8000, ch); got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// quietBedWithBurst is a bed of +/-2 LSB noise (about -89 LUFS, under the absolute
// gate) with a short tone burst in the middle. Planned from its measurement, which
// sees only the burst blocks, the gain lifts the whole bed above the gate.
func quietBedWithBurst(seconds, burstSeconds, burstDBFS float64, bedLSB int) []int16 {
	r := rand.New(rand.NewPCG(7, 11))
	n := int(seconds * planTestRate)
	pcm := make([]int16, n)
	for i := range pcm {
		pcm[i] = int16(r.IntN(2*bedLSB+1) - bedLSB)
	}
	tone := sineInt16(burstDBFS, 1000, burstSeconds, planTestRate)
	copy(pcm[n/2:], tone)
	return pcm
}

// TestPlanBytesAccountsForBlocksLiftedAcrossTheGate pins that a boost which lifts
// quiet blocks over the absolute gate is planned from the loudness the boosted clip
// will really have. Gating is not linear in gain: planning target - I from the
// unboosted measurement left such a clip about 20 LU under the target.
func TestPlanBytesAccountsForBlocksLiftedAcrossTheGate(t *testing.T) {
	pcm := quietBedWithBurst(20, 2, -60, 8)
	b := bytesLE(pcm)
	plan, err := PlanBytes(t.Context(), b, planOpts(planTestRate, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	if plan.PeakLimited || plan.Clamped || plan.GateLifted {
		t.Fatalf("fixture must be unbound, got %+v", plan)
	}
	g, err := gain.New(plan.GainDB)
	if err != nil {
		t.Fatal(err)
	}
	out := slices.Clone(b)
	if err := g.ApplyBytes(out); err != nil {
		t.Fatal(err)
	}
	got, err := MeasureBytes(out, planTestRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.IntegratedLUFS-(-23)) > 0.5 {
		t.Errorf("planned gain %.2f dB leaves the clip at %.2f LUFS, want -23 +/-0.5", plan.GainDB, got.IntegratedLUFS)
	}
	if got.TruePeakDBTP > -2+0.1 {
		t.Errorf("true peak %.2f dBTP above the -2 ceiling", got.TruePeakDBTP)
	}
}

// TestLoudnessAtGainMatchesScaledMeasurement checks the block-energy solver
// against measuring the actually scaled audio, across a boost that lifts the bed
// over the absolute gate, no change, and an attenuation.
func TestLoudnessAtGainMatchesScaledMeasurement(t *testing.T) {
	pcm := quietBedWithBurst(20, 2, -60, 8)
	m := NewMeter(planTestRate, 1)
	m.AddInt16(pcm)
	if got, want := m.loudnessAtGain(0), m.IntegratedLoudness(); got != want {
		t.Fatalf("loudnessAtGain(0) = %v, IntegratedLoudness = %v", got, want)
	}
	for _, g := range []float64{-6, 10, 25, 33, 41} {
		scaled := slices.Clone(pcm)
		pcmScale(t, scaled, g)
		want, err := MeasureInt16(scaled, planTestRate, 1)
		if err != nil {
			t.Fatal(err)
		}
		if got := m.loudnessAtGain(g); math.Abs(got-want.IntegratedLUFS) > 0.1 { // the scaled clip is rounded to int16, which a +/-8 LSB bed feels
			t.Errorf("gain %+.0f dB: block energies give %.3f LUFS, measuring the scaled clip %.3f", g, got, want.IntegratedLUFS)
		}
	}
}

// pcmScale applies gainDB to pcm the way gain.ApplyBytes does.
func pcmScale(t *testing.T, pcm []int16, gainDB float64) {
	t.Helper()
	g, err := gain.New(gainDB)
	if err != nil {
		t.Fatal(err)
	}
	b := bytesLE(pcm)
	if err := g.ApplyBytes(b); err != nil {
		t.Fatal(err)
	}
	for i := range pcm {
		pcm[i] = int16(uint16(b[2*i]) | uint16(b[2*i+1])<<8)
	}
}

// TestPlanBytesGateSolveClamped pins that a boost the clamp cuts anyway
// still reports the clamp and applies the clamped gain.
func TestPlanBytesGateSolveClamped(t *testing.T) {
	opts := planOpts(planTestRate, 1, false)
	opts.MaxGainDB = 10
	plan, err := PlanBytes(t.Context(), bytesLE(quietBedWithBurst(20, 2, -60, 8)), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Clamped || plan.GainDB != 10 || plan.PlannedGainDB <= plan.Result.GainDB+1 {
		t.Errorf("plan %+v, want a clamp to 10 dB of a gain the gate correction raised above the linear one", plan)
	}
}

// TestPlanBytesReportsPeakLimitFromRefinement pins Plan.PeakLimited for a clip the
// pass-one plan leaves unlimited but the gate-crossing correction pushes into the
// true-peak ceiling: Result.PeakLimited stays false, Plan.PeakLimited reports it.
func TestPlanBytesReportsPeakLimitFromRefinement(t *testing.T) {
	pcm := quietBedWithBurst(20, 2, -60, 8)
	pcm[len(pcm)/2+100] = 200 // a click about -44 dBFS: headroom for +42 dB, not for the corrected gain
	plan, err := PlanBytes(t.Context(), bytesLE(pcm), planOpts(planTestRate, 1, true))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Result.PeakLimited || !plan.PeakLimited || plan.PlannedGainDB <= plan.Result.GainDB {
		t.Errorf("plan %+v, want a refined gain above the pass-one gain, limited by the peak", plan)
	}
	g, err := gain.New(plan.GainDB)
	if err != nil {
		t.Fatal(err)
	}
	out := bytesLE(pcm)
	if err := g.ApplyBytes(out); err != nil {
		t.Fatal(err)
	}
	got, err := MeasureBytes(out, planTestRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.TruePeakDBTP > -2+0.1 {
		t.Errorf("true peak %.2f dBTP above the -2 ceiling", got.TruePeakDBTP)
	}
}

// TestNormalizeInt16QuietBedLandsOnTarget pins that the in-place Normalize path
// solves the gain through the loudness gates like PlanBytes: a quiet bed under a
// call lands on the target, and Result reports the loudness the clip really has.
func TestNormalizeInt16QuietBedLandsOnTarget(t *testing.T) {
	pcm := quietBedWithBurst(20, 2, -60, 8)
	want, err := PlanBytes(t.Context(), bytesLE(pcm), planOpts(planTestRate, 1, false))
	if err != nil {
		t.Fatal(err)
	}
	f32 := make([]float32, len(pcm))
	for i, v := range pcm {
		f32[i] = float32(v) / 32768
	}
	resF, err := NormalizeFloat32(f32, Options{SampleRate: planTestRate, Channels: 1, TargetLUFS: -23, TruePeakDBTP: -2})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(resF.GainDB-want.GainDB) > 1e-3 {
		t.Errorf("NormalizeFloat32 applied %.3f dB, PlanBytes plans %.3f dB", resF.GainDB, want.GainDB)
	}
	res, err := NormalizeInt16(pcm, Options{SampleRate: planTestRate, Channels: 1, TargetLUFS: -23, TruePeakDBTP: -2})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.GainDB-want.GainDB) > 1e-9 {
		t.Errorf("NormalizeInt16 applied %.3f dB, PlanBytes plans %.3f dB", res.GainDB, want.GainDB)
	}
	got, err := MeasureInt16(pcm, planTestRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.IntegratedLUFS-(-23)) > 0.5 {
		t.Errorf("normalized clip measures %.2f LUFS, want -23 +/-0.5", got.IntegratedLUFS)
	}
	if math.Abs(res.OutputLUFS-got.IntegratedLUFS) > 0.1 {
		t.Errorf("Result.OutputLUFS %.2f, measured %.2f", res.OutputLUFS, got.IntegratedLUFS)
	}
}

// TestPlanBytesCorrectsAttenuationAcrossTheGate pins the attenuation side of the
// gate correction: at a -69 LUFS target the cut drops the quiet bulk of a clip
// under the absolute gate, so the linear plan leaves it too loud.
func TestPlanBytesCorrectsAttenuationAcrossTheGate(t *testing.T) {
	pcm := sineInt16(-60.5, 1000, 18, planTestRate)
	pcm = append(pcm, sineInt16(-54, 1000, 2, planTestRate)...)
	opts := planOpts(planTestRate, 1, false)
	opts.TargetLUFS = -69
	plan, err := PlanBytes(t.Context(), bytesLE(pcm), opts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.PlannedGainDB >= plan.Result.GainDB-2 || plan.Clamped || plan.PeakLimited {
		t.Fatalf("plan %+v, want a deeper cut than the linear gain", plan)
	}
	out := slices.Clone(pcm)
	pcmScale(t, out, plan.GainDB)
	got, err := MeasureInt16(out, planTestRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.IntegratedLUFS-(-69)) > 0.5 {
		t.Errorf("cut by %.2f dB leaves the clip at %.2f LUFS, want -69 +/-0.5", plan.GainDB, got.IntegratedLUFS)
	}
}

// TestPlanBytesFallbackReportsPeakLimit pins Plan.PeakLimited on the gate fallback
// path: a sub-gate clip with one sharp sample is lifted into the true-peak
// ceiling, which Result (the pass-one plan) cannot report.
func TestPlanBytesFallbackReportsPeakLimit(t *testing.T) {
	for _, tc := range []struct {
		name  string
		spike int16
		want  bool
	}{{"flat bed", 0, false}, {"sharp sample", 20, true}} {
		t.Run(tc.name, func(t *testing.T) {
			pcm := quietBedWithBurst(20, 0.3, -90, 1)
			if tc.spike != 0 {
				pcm[len(pcm)/2+50] = tc.spike
			}
			plan, err := PlanBytes(t.Context(), bytesLE(pcm), planOpts(planTestRate, 1, true))
			if err != nil {
				t.Fatal(err)
			}
			if !plan.GateLifted || plan.Result.PeakLimited || plan.PeakLimited != tc.want {
				t.Fatalf("plan %+v, want GateLifted with PeakLimited %v and Result.PeakLimited false", plan, tc.want)
			}
		})
	}
}

// TestSolveGateAwareGainDBConverges runs the solver over a grid of quiet beds under
// a call and checks, on the block energies, that the solved gain leaves under
// 0.05 dB of correction unless the ceiling limits it, and is never worse than the
// linear gain it started from.
func TestSolveGateAwareGainDBConverges(t *testing.T) {
	opts := planOpts(planTestRate, 1, false).Options
	checked := 0
	for _, secs := range []float64{6, 20} {
		for _, burst := range []float64{-70, -65, -60, -55, -50} {
			for _, bed := range []int{2, 5, 8, 12, 24} {
				m := NewMeter(planTestRate, 1)
				m.AddInt16(quietBedWithBurst(secs, 1.5, burst, bed))
				start := PlanGain(result(m), opts)
				if math.IsInf(start.Input.IntegratedLUFS, -1) {
					continue
				}
				residual := func(g float64) float64 {
					return math.Abs(PlanGain(Measurement{IntegratedLUFS: m.loudnessAtGain(g), TruePeakDBTP: m.TruePeakDBTP() + g}, opts).GainDB)
				}
				got, limited := solveGateAwareGainDB(m, start.GainDB, opts)
				if r := residual(got); r > residual(start.GainDB) || (r > 0.05 && !limited) {
					t.Errorf("secs %v burst %v bed %d: solved %.3f dB leaves %.3f dB (start %.3f dB left %.3f dB), limited %v", secs, burst, bed, got, r, start.GainDB, residual(start.GainDB), limited)
				}
				checked++
			}
		}
	}
	if checked < 30 {
		t.Fatalf("only %d fixtures measured", checked)
	}
}

// TestSolveGateAwareGainDBSilence pins that a silent clip keeps the gain it was
// given and reports no peak limit.
func TestSolveGateAwareGainDBSilence(t *testing.T) {
	m := NewMeter(planTestRate, 1)
	m.AddInt16(make([]int16, 2*planTestRate))
	for _, planned := range []float64{0, 12} {
		if got, limited := solveGateAwareGainDB(m, planned, planOpts(planTestRate, 1, false).Options); got != planned || limited {
			t.Errorf("silent clip planned %v: got (%v, %v), want the gain unchanged and not limited", planned, got, limited)
		}
	}
}

// TestNormalizeInt16ReportsGateSolvePeakLimit pins Result.PeakLimited from the gate
// solve: the clip's linear plan is unlimited, the corrected gain reaches the
// true-peak ceiling, and Normalize reports it as PlanBytes does, with Input and
// TargetGainDB left at the pass-one values.
func TestNormalizeInt16ReportsGateSolvePeakLimit(t *testing.T) {
	pcm := quietBedWithBurst(20, 2, -60, 8)
	pcm[len(pcm)/2+100] = 200
	want, err := PlanBytes(t.Context(), bytesLE(pcm), planOpts(planTestRate, 1, false))
	if err != nil {
		t.Fatal(err)
	}
	if want.Result.PeakLimited || !want.PeakLimited {
		t.Fatalf("fixture must be limited only after the gate correction, got %+v", want)
	}
	res, err := NormalizeInt16(pcm, Options{SampleRate: planTestRate, Channels: 1, TargetLUFS: -23, TruePeakDBTP: -2})
	if err != nil {
		t.Fatal(err)
	}
	if !res.PeakLimited || math.Abs(res.GainDB-want.GainDB) > 1e-9 {
		t.Errorf("Normalize result %+v, want PeakLimited with the gain %.3f dB PlanBytes plans", res, want.GainDB)
	}
	if res.Input != want.Result.Input || res.TargetGainDB != want.Result.TargetGainDB {
		t.Errorf("Input %+v and TargetGainDB %.3f changed, want the pass-one %+v and %.3f", res.Input, res.TargetGainDB, want.Result.Input, want.Result.TargetGainDB)
	}
}
