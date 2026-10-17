package equalizer

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
)

// SectionFloorDB is the least dB one biquad section contributes to GainDB. An
// exact null, or a section deeper than this, reads exactly SectionFloorDB, so
// GainDB is finite at every finite frequency and encodes as JSON. Float64
// round-off alone puts an RBJ notch's value at its own centre anywhere from
// -Inf to about -300 dB, so the floor also makes that reading the same on every
// platform. It is amplitude dB (a magnitude of 1e-10), matching the -200 dB
// floor of the octave and spectrogram packages, and applies per section, not to
// the total: k sections at the floor read k*SectionFloorDB.
const SectionFloorDB = -200.0

// ResponsePoint is a cascade's steady-state response at one frequency: the
// magnitude in dB and the phase in radians. GainDB is the sum of the dB gains of
// the cascade's sections, each at least SectionFloorDB, so it is finite for any
// finite frequency and a deep cascade stays finite. An exact null, or a section
// deeper than the floor, reads SectionFloorDB per section. A non-finite
// frequency yields NaN for both GainDB and PhaseRad. PhaseRad is the summed
// section phase wrapped to [-pi, pi] and is unspecified where a section's
// magnitude is at or below the floor.
type ResponsePoint struct {
	Hz       float64
	GainDB   float64
	PhaseRad float64
}

// at returns the complex frequency response H(e^jw) of one section at angular
// frequency w in radians per sample.
func (c coeffs) at(w float64) complex128 {
	z1 := cmplx.Rect(1, -w)   // e^{-jw}
	z2 := cmplx.Rect(1, -2*w) // e^{-2jw}
	num := complex(c.b0, 0) + complex(c.b1, 0)*z1 + complex(c.b2, 0)*z2
	den := complex(1, 0) + complex(c.a1, 0)*z1 + complex(c.a2, 0)*z2
	return num / den
}

// Response evaluates the equalizer's magnitude and phase at each frequency in
// freqs (Hz), returning one ResponsePoint per input frequency in order. It is a
// visualization helper (for example to draw a response curve), not an
// audio-path method: it allocates the returned slice and does not touch the
// filter state, so it is safe to call while streaming. Only frequencies in
// [0, SampleRate/2] are meaningful; the response is periodic, so a frequency
// outside that range is evaluated modulo SampleRate rather than rejected.
func (e *Equalizer) Response(freqs []float64) []ResponsePoint {
	return sectionsResponse(e.sections, e.sampleRate, freqs)
}

// Response evaluates the filter's magnitude and phase at each frequency in
// freqs (Hz), with the same conventions as Equalizer.Response: one
// ResponsePoint per input frequency in order, each section floored at
// SectionFloorDB and NaN for a non-finite frequency. It is a visualization and
// measurement helper, not an audio-path method: it allocates the returned slice
// and does not touch the filter state, so it is safe to call while streaming.
func (f *Filter) Response(freqs []float64) []ResponsePoint {
	return sectionsResponse(f.sections, f.sampleRate, freqs)
}

// omega converts a frequency in Hz to radians per sample at sampleRate. The
// response is periodic in sampleRate, so f is reduced first (math.Mod is exact
// and returns f unchanged below sampleRate); without it a huge finite f
// overflows 2*pi*f and the response turns NaN. A non-finite f stays NaN.
func omega(f float64, sampleRate int) float64 {
	rate := float64(sampleRate)
	if math.Abs(f) >= rate {
		f = math.Mod(f, rate)
	}
	return 2 * math.Pi * f / rate
}

// addSections adds the response of each section at w, in order, to the running
// gain in dB and phase in radians. Summing per section instead of multiplying
// the complex responses keeps a deep cascade finite: a product of many
// high-gain (or deep-attenuation) sections overflows (or underflows to zero)
// long before its dB sum is large, whereas the sum of dB values only grows
// linearly. Each section's dB is floored at SectionFloorDB, which maps an exact
// zero magnitude (log10 gives -Inf) to a finite value; a NaN magnitude still
// makes the gain NaN, because the builtin max propagates NaN.
func addSections(gainDB, phase float64, sections []section, w float64) (gainOut, phaseOut float64) {
	for j := range sections {
		h := sections[j].c.at(w)
		gainDB += max(20*math.Log10(cmplx.Abs(h)), SectionFloorDB)
		phase += cmplx.Phase(h)
	}
	return gainDB, phase
}

// responsePoint builds a ResponsePoint from the summed gain and phase of a
// cascade at f Hz, wrapping the phase to [-pi, pi]. A non-finite f is mapped to
// NaN explicitly so the documented rule holds even for an empty cascade, which
// has no section to produce a NaN.
func responsePoint(f, gainDB, phase float64) ResponsePoint {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return ResponsePoint{Hz: f, GainDB: math.NaN(), PhaseRad: math.NaN()}
	}
	return ResponsePoint{Hz: f, GainDB: gainDB, PhaseRad: math.Remainder(phase, 2*math.Pi)}
}

// sectionsResponse evaluates the cascade of sections at sampleRate at each
// frequency in freqs, through the shared omega, addSections and responsePoint
// helpers that FilterChain.ResponseInto also uses.
func sectionsResponse(sections []section, sampleRate int, freqs []float64) []ResponsePoint {
	pts := make([]ResponsePoint, len(freqs))
	for i, f := range freqs {
		gainDB, phase := addSections(0, 0, sections, omega(f, sampleRate))
		pts[i] = responsePoint(f, gainDB, phase)
	}
	return pts
}

// Response evaluates the chain's magnitude (GainDB, in dB) and phase (PhaseRad,
// in radians) at each frequency in freqs (Hz), returning one ResponsePoint per
// input frequency in order. The response is that of every filter's sections in
// application order, the same cascade ProcessInto and ApplyFloat64 run. The dB
// and phase of each section are summed rather than multiplying complex
// responses, so a deep cascade stays finite (see SectionFloorDB for the
// per-section floor). An empty chain is the identity: 0 dB and 0 rad at every
// finite frequency. A non-finite frequency yields NaN. PhaseRad is the summed
// phase wrapped to [-pi, pi], so unwrapping is the caller's job. Only
// frequencies in [0, SampleRate()/2] are meaningful; others are evaluated
// modulo SampleRate(). It allocates the returned slice (use ResponseInto to
// avoid that) and does not touch the filter state, so it is safe to call while
// streaming.
func (c *FilterChain) Response(freqs []float64) []ResponsePoint {
	pts := make([]ResponsePoint, len(freqs))
	_, _ = c.ResponseInto(pts, freqs) // cannot fail: pts is sized exactly
	return pts
}

// ResponseInto is the allocation-free form of Response: it writes one
// ResponsePoint per frequency into dst[:len(freqs)] and returns len(freqs).
// When len(dst) < len(freqs) it returns ErrBufferTooSmall and writes nothing;
// elements of dst beyond len(freqs) are left untouched. Units, conventions and
// state handling are those of Response.
func (c *FilterChain) ResponseInto(dst []ResponsePoint, freqs []float64) (int, error) {
	if len(dst) < len(freqs) {
		return 0, ErrBufferTooSmall
	}
	for i, f := range freqs {
		var gainDB, phase float64
		if len(c.filters) > 0 { // an empty chain has no rate to divide by; it stays 0 dB, 0 rad
			w := omega(f, c.sampleRate)
			for _, flt := range c.filters {
				gainDB, phase = addSections(gainDB, phase, flt.sections, w)
			}
		}
		dst[i] = responsePoint(f, gainDB, phase)
	}
	return len(freqs), nil
}

// LogSweep returns n frequencies spaced logarithmically from fLow to fHigh
// inclusive, the usual x-axis for a response plot. It returns an error wrapping
// ErrInvalidConfig if fLow <= 0, fHigh <= fLow, fHigh is +Inf, n < 2, or the
// range is too narrow for n distinct float64 points. The first and last values
// are exactly fLow and fHigh, and the result is finite and strictly increasing.
//
// A narrow feature such as a deep notch or a high-Q peak can fall between grid
// points; LogSweepWithMarks adds the filters' own frequencies to the grid.
func LogSweep(fLow, fHigh float64, n int) ([]float64, error) {
	return logGrid(fLow, fHigh, n)
}

// LogSweepWithMarks returns LogSweep(fLow, fHigh, n) with every mark in
// [fLow, fHigh] merged in: the result is ascending, has no duplicates, and
// holds every LogSweep point and every in-range mark exactly. Its length is
// between n and n+len(marks), so a caller reusing a ResponseInto buffer sizes it
// by len(result). Marks outside the range, and NaN or infinite marks, are
// dropped rather than rejected, so a filter above the plotted range is fine.
// Pass FilterChain.Marks, Filter.Marks or Equalizer.Marks, plus any frequencies
// of a caller-designed filter. Errors are those of LogSweep.
func LogSweepWithMarks(fLow, fHigh float64, n int, marks []float64) ([]float64, error) {
	freqs, err := logGrid(fLow, fHigh, n)
	if err != nil {
		return nil, err
	}
	freqs = slices.Grow(freqs, len(marks))
	for _, m := range marks {
		if fLow <= m && m <= fHigh { // false for NaN; +/-Inf is out of range
			freqs = append(freqs, m)
		}
	}
	return sortedUnique(freqs), nil
}

// sortedUnique sorts s in place and drops duplicates; it is the one definition
// of "sorted and duplicate-free" for every marks list and sweep.
func sortedUnique(s []float64) []float64 {
	slices.Sort(s)
	return slices.Compact(s)
}

// logGrid builds the LogSweep grid.
func logGrid(fLow, fHigh float64, n int) ([]float64, error) {
	if !(fLow > 0) {
		return nil, fmt.Errorf("%w: fLow must be > 0, got %g", ErrInvalidConfig, fLow)
	}
	if !(fHigh > fLow) || math.IsInf(fHigh, 1) {
		return nil, fmt.Errorf("%w: fHigh must be finite and > fLow (%g), got %g", ErrInvalidConfig, fLow, fHigh)
	}
	if n < 2 {
		return nil, fmt.Errorf("%w: n must be >= 2, got %d", ErrInvalidConfig, n)
	}
	freqs := make([]float64, n)
	logLow := math.Log(fLow)
	step := (math.Log(fHigh) - logLow) / float64(n-1)
	for i := range freqs {
		// Clamp: exp round-off must not leave [fLow, fHigh].
		freqs[i] = min(max(math.Exp(logLow+step*float64(i)), fLow), fHigh)
	}
	freqs[0] = fLow // pin the endpoints exactly, free of exp/log round-off
	freqs[n-1] = fHigh
	for i := 1; i < n; i++ {
		if !(freqs[i] > freqs[i-1]) {
			return nil, fmt.Errorf("%w: range %g to %g is too narrow for %d points", ErrInvalidConfig, fLow, fHigh, n)
		}
	}
	return freqs, nil
}
