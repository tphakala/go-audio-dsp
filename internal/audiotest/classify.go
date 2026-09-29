package audiotest

import (
	"cmp"
	"math"
	"slices"
)

const (
	// corpusSampleRate is the mono rate every clip is decoded to. Both the Go
	// denoiser and afftdn see this identical buffer, so the A/B is apples to
	// apples. 48 kHz is BirdNET's rate and matches the synthetic harness.
	corpusSampleRate = 48000
	// corpusWin is the energy-classification window (~21 ms at 48 kHz).
	corpusWin = 1024
	// corpusQuietFrac is the fraction of windows (lowest energy) taken as the
	// noise floor.
	corpusQuietFrac = 0.15
	// corpusSignalMarginDB selects signal-dominant windows: those whose level is
	// at least this far above the pooled noise floor. A window ~12 dB above the
	// floor is roughly 12 dB SNR, at which removing the noise moves its level by
	// under ~0.3 dB, so its level drop reflects signal damage, not noise removal.
	// A relative threshold (rather than a fixed top percentile) keeps sparse-signal
	// clips from pulling noise-only windows into the signal set.
	corpusSignalMarginDB = 12
	// corpusMinWindows is the fewest classification windows a clip needs for a
	// stable floor estimate (~0.43 s at 48 kHz).
	corpusMinWindows = 20
	// corpusMaxLag bounds the afftdn delay search (samples). Alignment shifts the
	// reference left by the measured delay, zero-filling its last lag samples, so
	// measurement windows are kept clear of the final corpusMaxLag samples.
	corpusMaxLag = 4096
	// silenceFloorDB treats a pooled level at or below this as silence: spanRMSDB
	// returns a -200 dBFS sentinel for an all-zero span, and this sits above it
	// with margin. A strength floors attenuation at MaxAttenuationDB (<= 20 dB), so
	// real output never lands between here and the sentinel.
	silenceFloorDB = -190.0
	// abToleranceDB is how far ours may trail afftdn on either metric before the
	// A/B fails (the spec's within-1-dB-of-afftdn bar).
	abToleranceDB = 1.0
	// floorRiseTolDB absorbs the small apparent noise-floor rise WOLA overlap-add
	// can leak into a quiet window near a burst; a gain <= 1 denoiser stays within
	// it.
	floorRiseTolDB = 0.5
)

// windowMS returns the mean-square energy of the corpusWin-sample window at s.
func windowMS(x []float32, s int) float64 {
	var acc float64
	for _, v := range x[s : s+corpusWin] {
		acc += float64(v) * float64(v)
	}
	return acc / float64(corpusWin)
}

// ClassifyWindows splits x into non-overlapping corpusWin-sample windows and
// returns the quietest corpusQuietFrac as noise-floor spans and, as signal spans,
// every window at least corpusSignalMarginDB above the pooled noise floor. ok is
// false when the floor is silent or no window rises above the signal threshold
// (nothing to measure).
//
// Exact digital-silence windows (recorder pre-roll, a dropout: a run of exact
// zeros) are excluded from the noise-floor candidates. Without that a long run
// of zeros fills the quietest set and pegs the estimated floor to the -200 dB
// silence sentinel, skipping an otherwise usable clip.
func ClassifyWindows(x []float32) (quiet, signal [][2]int, ok bool) {
	nWin := len(x) / corpusWin
	if nWin < 1 {
		return nil, nil, false
	}
	// Per-window mean-square, computed once and reused for both the quiet-set
	// sort and the signal-threshold pass. Recomputing it inside the sort
	// comparator (and again per window below) made classification do O(n log n)
	// passes over the audio.
	ms := make([]float64, nWin)
	for w := range ms {
		ms[w] = windowMS(x, w*corpusWin)
	}
	// Noise-floor candidates are the non-silent windows, ordered quietest first.
	order := make([]int, 0, nWin)
	for w := range ms {
		if ms[w] > 0 {
			order = append(order, w)
		}
	}
	if len(order) == 0 {
		return nil, nil, false // clip is entirely digital silence
	}
	slices.SortStableFunc(order, func(a, b int) int {
		return cmp.Compare(ms[a], ms[b])
	})
	nq := max(1, int(float64(len(order))*corpusQuietFrac))
	for _, w := range order[:nq] {
		s := w * corpusWin
		quiet = append(quiet, [2]int{s, s + corpusWin})
	}
	floor := SpanRMSDB(x, quiet)
	if floor <= silenceFloorDB { // silent clip: at/below the -200 sentinel, nothing to reduce
		return nil, nil, false
	}
	threshold := floor + corpusSignalMarginDB
	for w := range nWin {
		if windowLevelFromMS(ms[w]) > threshold {
			s := w * corpusWin
			signal = append(signal, [2]int{s, s + corpusWin})
		}
	}
	if len(signal) == 0 {
		return nil, nil, false
	}
	return quiet, signal, true
}

// windowLevelFromMS is a window's level in the same dB scale SpanRMSDB uses
// (10*log10 of the mean square), from its precomputed mean-square. A fully
// silent window (ms 0) returns the -200 sentinel, which sits below any real
// floor+margin, so it is correctly excluded from the signal set.
func windowLevelFromMS(ms float64) float64 {
	if ms == 0 {
		return -200
	}
	return 10 * math.Log10(ms)
}
