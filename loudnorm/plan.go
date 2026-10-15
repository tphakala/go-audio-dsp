package loudnorm

import (
	"context"
	"fmt"
	"math"
	"sync"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/pcm"
)

// planChunkSamples is how many interleaved int16 samples PlanBytes and
// MeasureBytes decode and meter between context checks. A chunk is always a
// whole number of frames, so the meter never sees a split frame.
const planChunkSamples = 1 << 16

// planScratchPool holds the fixed-size decode scratch so steady-state planning
// allocates nothing. Inputs with more channels than planChunkSamples (never in
// practice) get a one-off scratch instead.
var planScratchPool = sync.Pool{New: func() any { return new([planChunkSamples]int16) }}

// PlanOptions configures PlanBytes.
type PlanOptions struct {
	// Options carries the sample rate, channel count, target loudness and
	// true-peak ceiling. All of them are validated.
	Options
	// MaxGainDB bounds the magnitude of the applied gain: the planned gain is
	// clamped to [-MaxGainDB, +MaxGainDB]. It must be finite and > 0.
	MaxGainDB float64
	// GateFallback lifts a clip the absolute gate leaves unmeasured; see
	// GateFallbackGainDB. With it off such a clip plans no gain.
	GateFallback bool
}

// Plan is the outcome of PlanBytes.
type Plan struct {
	// GainDB is the gain to apply: PlannedGainDB clamped to +/-MaxGainDB.
	GainDB float64
	// PlannedGainDB is the gain before the clamp, after the gate fallback when
	// it fired.
	PlannedGainDB float64
	// Result is PlanGain on the pass-one measurement, before the gate fallback
	// and the clamp. Result.Input is the measurement of the unmodified clip.
	Result Result
	// GateLifted reports whether the clip sat under the absolute gate and the
	// gate fallback (and its refinement) decided PlannedGainDB.
	GateLifted bool
	// PeakLimited reports whether the true-peak ceiling limited PlannedGainDB. It
	// is Result.PeakLimited for the pass-one plan, and the final pass's verdict
	// when a refinement (the gate fallback or the gate-crossing correction)
	// replaced it, which Result does not carry.
	PeakLimited bool
	// Clamped reports whether the clamp changed the gain.
	Clamped bool
}

// PlanBytes plans the single gain that brings interleaved little-endian int16
// PCM carried as bytes to the target loudness without its true peak exceeding
// the ceiling, for a caller that applies the gain itself (for example while
// encoding, to avoid a second buffer pass). It measures b with the BS.1770 meter
// (integrated loudness and true peak), plans with PlanGain, runs the gate
// fallback when opts.GateFallback is set, and clamps the result to
// +/-opts.MaxGainDB with ClampGainDB.
//
// b is only read: it is decoded in chunks of whole frames into a small scratch,
// so no copy of the clip is made, and the result is bit-identical to measuring
// the whole clip at once. ctx is checked before any work and between chunks;
// when it is done PlanBytes returns ctx.Err() unwrapped with a zero Plan, so a
// long recording can be abandoned promptly. A cancelled context is reported as
// cancellation even when the options are also invalid.
//
// Gate fallback. A clip whose every 400 ms block sits under the absolute gate
// measures -Inf and PlanGain plans no gain for it. With GateFallback set and a
// finite true peak, the clip is lifted by GateFallbackGainDB, the lifted signal
// is measured again (scaled chunk by chunk into the scratch, never copied
// whole), and once it is measurable a normal PlanGain finishes the job from
// that measurement, so a very quiet clip lands on the target instead of merely
// somewhere audible. A clip that is still under the gate after the lift keeps
// the conservative lift. Plan.GateLifted reports the fallback. This extra pass
// runs only for sub-gate clips.
//
// Loudness gating is not linear in gain either: a boost can lift quiet blocks
// from under the absolute gate to over it, and the boosted clip is then quieter
// than the unboosted measurement plus the gain says. When the planned boost would
// do that, the gain is corrected from the block energies of the one measuring pass,
// so a clip with a quiet bed under a call lands on the target instead of several
// LU below it, at no extra pass over the audio.
//
// Plan.PlannedGainDB is the gain before the clamp and Plan.GainDB the gain to
// apply; Plan.Result.OutputLUFS and TargetGainDB project the pre-fallback,
// pre-clamp plan, so once Clamped or GateLifted is set they no longer describe
// the output. Silent or sub-400 ms input without the fallback plans a gain of 0,
// leaving a quiet clip unchanged rather than boosting its noise floor.
//
// Errors: ErrOddByteLength for an odd len(b); the sentinel of the first invalid
// option (sample rate, channels, length not a multiple of channels, target,
// ceiling); and an error wrapping dsp.ErrInvalidConfig for a MaxGainDB that is
// not finite and > 0. Every error returns a zero Plan.
func PlanBytes(ctx context.Context, b []byte, opts PlanOptions) (Plan, error) {
	if err := ctx.Err(); err != nil {
		return Plan{}, err
	}
	if len(b)%2 != 0 {
		return Plan{}, ErrOddByteLength
	}
	if err := opts.validate(len(b) / 2); err != nil {
		return Plan{}, err
	}
	if math.IsNaN(opts.MaxGainDB) || math.IsInf(opts.MaxGainDB, 0) || opts.MaxGainDB <= 0 {
		return Plan{}, fmt.Errorf("%w: MaxGainDB must be finite and > 0, got %g", dsp.ErrInvalidConfig, opts.MaxGainDB)
	}

	meas := Measurement{IntegratedLUFS: math.Inf(-1), TruePeakDBTP: math.Inf(-1)}
	var m *Meter
	if len(b) > 0 {
		m = acquireMeter(opts.SampleRate, opts.Channels)
		defer releaseMeter(m)
		if err := feedBytes(ctx, m, b, opts.Channels, 1); err != nil {
			return Plan{}, err
		}
		meas = result(m)
	}

	plan := Plan{Result: PlanGain(meas, opts.Options)}
	planned := plan.Result.GainDB
	plan.PeakLimited = plan.Result.PeakLimited
	if opts.GateFallback {
		if lift, ok := GateFallbackGainDB(meas, opts.TargetLUFS, opts.TruePeakDBTP); ok {
			refined, limited, err := refineLiftedGainDB(ctx, m, b, lift, opts.Options)
			if err != nil {
				return Plan{}, err
			}
			// The lift is deliberately conservative, because a sub-gate clip's
			// real loudness is unknown and the bound has to assume the worst
			// case. Lifting usually raises the clip above the gate, though, at
			// which point it can be measured: add the refinement from that real
			// measurement so a very quiet clip lands on target.
			planned = lift + refined
			plan.PeakLimited = limited
			plan.GateLifted = true
		}
	}
	if !plan.GateLifted && m != nil {
		planned, plan.PeakLimited = solveGateAwareGainDB(m, planned, plan.PeakLimited, opts.Options)
	}
	plan.PlannedGainDB = planned
	plan.GainDB, plan.Clamped = ClampGainDB(planned, opts.MaxGainDB)
	return plan, nil
}

// GateFallbackGainDB handles the clip EBU R128 gating cannot measure. A clip
// whose gated integrated loudness falls below the -70 LUFS absolute gate yields
// -Inf, and PlanGain plans no gain at all for it, so the clip would be exported
// untouched and inaudible. When such a clip still has a finite true peak there
// is enough signal to work with, so the gain is anchored to the true-peak
// ceiling, which is what ffmpeg loudnorm does for the same case.
//
// That anchor alone is not enough. Lifting purely by true-peak headroom leaves
// the resulting loudness at (ceiling - crest factor), so a FLAT signal (steady
// hiss from a dead or muted microphone, whose crest factor is only a few dB)
// lands far LOUDER than the loudness target: measured, a sub-gate noise floor
// came out at -13.8 LUFS against a -23 target. The second bound closes that.
// Gating failing is itself information: the clip's true loudness must be below
// AbsoluteGateLUFS, so lifting by no more than (target - gate) cannot
// overshoot the target however peaky or flat the clip turns out to be.
//
// That premise is worth stating precisely, since the bound rests on it.
// The meter reports -Inf on exactly three paths: every 400 ms block sat
// under the absolute gate (the premise holds by construction); the relative gate
// eliminated every block (unreachable, because the loudest gated block always
// exceeds a gate set 10 LU below their mean); or the clip is shorter than one
// 400 ms block, where loudness is undefined at ANY level. Only that third path
// breaks the premise. A short window is lifted to its true-peak ceiling, by at
// most (target - gate) dB, and is attenuated instead when its peak is already
// above the ceiling.
//
// Genuine silence (a true peak of -Inf too) reports false and is left alone;
// there is nothing to lift.
func GateFallbackGainDB(meas Measurement, targetLUFS, truePeakDBTP float64) (gainDB float64, ok bool) {
	if !math.IsInf(meas.IntegratedLUFS, -1) || math.IsInf(meas.TruePeakDBTP, -1) {
		return 0, false
	}
	peakBound := truePeakDBTP - meas.TruePeakDBTP
	loudnessBound := targetLUFS - AbsoluteGateLUFS
	return math.Min(peakBound, loudnessBound), true
}

// refineLiftedGainDB returns the extra gain to add on top of a gate-fallback
// lift. Loudness gating is not linear in gain, so the loudness of the lifted
// signal cannot be derived arithmetically: blocks that sat under the absolute
// gate before the lift pass it afterwards. The only way to know is to measure
// the lifted signal. It reuses m (reset), scaling each chunk in the scratch
// with the same factor and kernel gain.ApplyBytes uses, so the measured signal
// is bit-identical to what applying the lift would produce, with no copy of the
// clip.
//
// Once the clip is measurable a normal PlanGain finishes the job, landing it on
// the target and respecting the true-peak ceiling; peakLimited reports whether
// the ceiling limited it. A clip still under the gate after the lift yields 0,
// leaving the conservative lift as the final answer rather than guessing. The only
// error is the context's.
func refineLiftedGainDB(ctx context.Context, m *Meter, b []byte, liftDB float64, opts Options) (gainDB float64, peakLimited bool, err error) {
	m.Reset()
	if err := feedBytes(ctx, m, b, opts.Channels, float32(dsp.FactorFromDB(liftDB))); err != nil {
		return 0, false, err
	}
	lifted := result(m)
	if math.IsInf(lifted.IntegratedLUFS, -1) {
		return 0, false, nil
	}
	res := PlanGain(lifted, opts)
	return res.GainDB, res.PeakLimited, nil
}

// gateSolveIters bounds the iterations of solveGateAwareGainDB. Each step costs
// one pass over the block energies, no audio, and lands within a fraction of a dB
// of the target, so the bound is only a backstop.
const gateSolveIters = 8

// gateSolveTolDB is the correction under which solveGateAwareGainDB stops: below
// it the plan is as accurate as the meter.
const gateSolveTolDB = 0.005

// solveGateAwareGainDB corrects a planned gain for loudness gating. PlanGain
// assumes loudness moves one for one with gain, but the gates are fixed levels: a
// boost lifts quiet blocks from under the -70 LUFS absolute gate to over it, which
// adds them to the integrated loudness, and the boosted clip lands under the
// target (4 to 7 LU on a quiet bed under a call; ffmpeg loudnorm does the same).
// An attenuation drops blocks under the gate the same way. m holds the
// measurement of the unscaled clip and planned the gain PlanGain chose for it.
//
// The loudness at a trial gain follows from the stored block energies
// (loudnessAtGain) and the true peak from the measured one plus the gain, so each
// iteration re-plans with PlanGain on the loudness and peak the clip would have at
// the current gain and adds the correction, until it is below gateSolveTolDB or
// gateSolveIters have run. A clip no block of which crosses a gate needs no
// correction and keeps planned unchanged. limited reports whether the true-peak
// ceiling limited the final plan, starting from peakLimited.
func solveGateAwareGainDB(m *Meter, planned float64, peakLimited bool, opts Options) (gainDB float64, limited bool) {
	peak := m.TruePeakDBTP()
	for range gateSolveIters {
		res := PlanGain(Measurement{IntegratedLUFS: m.loudnessAtGain(planned), TruePeakDBTP: peak + planned}, opts)
		peakLimited = res.PeakLimited
		if math.Abs(res.GainDB) < gateSolveTolDB {
			break
		}
		planned += res.GainDB
	}
	return planned, peakLimited
}

// feedBytes feeds m the little-endian int16 PCM in b, scaled by factor, without
// writing b. It decodes whole-frame chunks into a pooled scratch with
// pcm.BytesToInt16 (a read-only decode on every host, unlike pcm.InPlaceInt16,
// which writes back through b on byte-wise hosts), scales the scratch in place,
// and checks ctx before each chunk. len(b) must be even and a whole number of
// frames of the given channel count; callers validate that first. On a done
// context it returns ctx.Err() unwrapped.
func feedBytes(ctx context.Context, m *Meter, b []byte, channels int, factor float32) error {
	chunkFrames := max(1, planChunkSamples/channels)
	// Whole frames only: AddInt16 drops a trailing partial frame. Bytes, with
	// the frame count bounded by planChunkSamples, cannot overflow int.
	chunkBytes := chunkFrames * channels * 2
	var scratch []int16
	if chunkFrames*channels <= planChunkSamples {
		arr := planScratchPool.Get().(*[planChunkSamples]int16)
		defer planScratchPool.Put(arr)
		scratch = arr[:]
	} else {
		scratch = make([]int16, chunkFrames*channels)
	}
	for off := 0; off < len(b); {
		if err := ctx.Err(); err != nil {
			return err
		}
		take := min(chunkBytes, len(b)-off)
		n, _ := pcm.BytesToInt16(scratch[:take/2], b[off:off+take]) // take is even
		samples := scratch[:n]
		pcm.ScaleInt16(samples, factor)
		m.AddInt16(samples)
		off += take
	}
	return nil
}
