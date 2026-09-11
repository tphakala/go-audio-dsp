package loudnorm

import "github.com/tphakala/go-audio-dsp/pcm"

// MeasureBytes measures interleaved little-endian int16 PCM carried as bytes,
// the transport form of the signal. len(b) must be even (two bytes per sample);
// otherwise ErrOddByteLength is returned. On little-endian hosts the bytes are
// read as int16 with no copy.
func MeasureBytes(b []byte, sampleRate, channels int) (Measurement, error) {
	if len(b)%2 != 0 {
		return Measurement{}, ErrOddByteLength
	}
	if len(b) == 0 {
		// InPlaceInt16 skips the closure on empty input, so validate and
		// measure directly to stay identical to MeasureInt16 (which rejects a
		// bad sample rate or channel count and reports -Inf for no samples).
		return MeasureInt16(nil, sampleRate, channels)
	}
	var (
		meas Measurement
		err  error
	)
	pcm.InPlaceInt16(b, func(s []int16) {
		meas, err = MeasureInt16(s, sampleRate, channels)
	})
	return meas, err
}

// NormalizeBytes normalizes interleaved little-endian int16 PCM carried as bytes
// in place, writing the gained samples back into b, and returns the outcome.
// len(b) must be even; otherwise ErrOddByteLength is returned. On little-endian
// hosts the normalization is zero-copy: it needs no scratch buffer the size of
// the clip. Gain is applied with rounding and saturation so a boost never wraps;
// silent input is left untouched (GainDB = 0).
func NormalizeBytes(b []byte, opts Options) (Result, error) {
	if len(b)%2 != 0 {
		return Result{}, ErrOddByteLength
	}
	if len(b) == 0 {
		// InPlaceInt16 skips the closure on empty input, so validate and
		// normalize directly to stay identical to NormalizeInt16 (which rejects
		// invalid options and is a no-op for no samples).
		return NormalizeInt16(nil, opts)
	}
	var (
		res Result
		err error
	)
	pcm.InPlaceInt16(b, func(s []int16) {
		res, err = NormalizeInt16(s, opts)
	})
	return res, err
}

// PlanClampedGainBytes runs the measure, plan, clamp sequence for a caller that
// applies the gain itself (for example while encoding, to avoid a second buffer
// pass). It measures the integrated loudness and true peak of interleaved
// little-endian int16 PCM bytes with MeasureBytes (which reads them in place
// without mutating b), plans the single gain that brings the clip to
// opts.TargetLUFS without its true peak exceeding opts.TruePeakDBTP with
// PlanGain, then clamps that gain to [-|maxAbsGainDB|, +|maxAbsGainDB|] with
// ClampGainDB.
//
// clampedGainDB is the gain to apply. res is PlanGain's untouched planning
// Result: res.Input is the pass-one measurement, res.GainDB is the pre-clamp
// planned gain (equal to clampedGainDB when the clamp did not fire), and
// res.PeakLimited reports true-peak limiting; limited reports whether the clamp
// took effect. res.OutputLUFS and res.TargetGainDB are PlanGain's projections for
// the pre-clamp gain (res.GainDB); once limited is true the applied gain is
// clampedGainDB, so those two projections no longer describe the output. Silent
// or sub-400 ms input yields clampedGainDB == 0, leaving a
// quiet clip unchanged rather than boosting its noise floor. len(b) must be
// even; an odd length returns ErrOddByteLength and zeroes every other return value.
//
// Only the sample rate and channel count are validated (by MeasureBytes). Like
// PlanGain, opts.TargetLUFS and opts.TruePeakDBTP are not range-checked here, so
// a caller that bypasses Normalize* must pass a target in (-70, 0) and a ceiling
// <= 0. A NaN target makes PlanGain produce a NaN gain, which ClampGainDB then
// clamps to 0 (no change) rather than letting it corrupt the signal.
func PlanClampedGainBytes(b []byte, opts Options, maxAbsGainDB float64) (clampedGainDB float64, res Result, limited bool, err error) {
	meas, err := MeasureBytes(b, opts.SampleRate, opts.Channels)
	if err != nil {
		return 0, Result{}, false, err
	}
	res = PlanGain(meas, opts)
	clampedGainDB, limited = ClampGainDB(res.GainDB, maxAbsGainDB)
	return clampedGainDB, res, limited, nil
}
