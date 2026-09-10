package denoiser

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"
)

// TestCorpusAgainstAfftdn is the real-corpus A/B comparison behind the Taskfile
// `ab` target. It runs every recording in the corpus through the Go denoiser and
// through ffmpeg's afftdn (the BirdNET-Go baseline this library replaces) at each
// preset, and compares two reference-free metrics per clip per preset:
//
//   - noise reduction: the level drop in the clip's quietest windows (its noise
//     floor). Higher is better; ours should match or beat afftdn.
//   - signal retention: the level drop in the loudest, high-SNR windows. Those
//     windows are signal-dominated, so their level barely moves when only noise
//     is removed; a large drop means the denoiser is eating the signal. Ours
//     should not lose more than afftdn.
//
// Both are pooled RMS levels (spanRMSDB), not spectral distances: real clips have
// no clean reference, so a distance-from-input metric would just re-measure noise
// removal and penalize the denoiser for working. Spectral quality (musical noise)
// is a listening judgment this test cannot make.
//
// The corpus is user-supplied and gitignored (denoiser/testdata/corpus/*.wav), so
// this test never runs in CI: it skips when the corpus or ffmpeg is absent. It is
// a manual tuning tool, so a miss against afftdn fails loudly rather than skipping
// quietly. Authoritative numbers need a pinned ffmpeg (see issue #6); an unpinned
// host afftdn can drift the comparison.

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
	// with margin. A preset floors attenuation at MaxAttenuationDB (<= 20 dB), so
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

// corpusDir returns the directory holding the user-supplied A/B corpus. It is
// gitignored (testdata/corpus) and absent in CI; DENOISER_CORPUS_DIR overrides
// the location for a corpus kept elsewhere.
func corpusDir() string {
	if d := os.Getenv("DENOISER_CORPUS_DIR"); d != "" {
		return d
	}
	return filepath.Join("testdata", "corpus")
}

// decodeToF32Mono decodes a corpus clip to mono float32 at sr Hz with ffmpeg,
// which the afftdn reference already requires; this avoids a WAV parser and any
// go-wav dependency. ffmpeg accepts any WAV encoding, channel count, or source
// rate (the corpus glob admits *.wav only; see TestCorpusAgainstAfftdn).
func decodeToF32Mono(t *testing.T, bin, path string, sr int) []float32 {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "-hide_banner", "-nostats", "-v", "error",
		"-i", path, "-f", "f32le", "-ac", "1", "-ar", strconv.Itoa(sr), "-")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("ffmpeg decode %s failed: %v\n%s", filepath.Base(path), err, stderr.String())
	}
	return f32leSamples(stdout.Bytes())
}

// windowMS returns the mean-square energy of the corpusWin-sample window at s.
func windowMS(x []float32, s int) float64 {
	var acc float64
	for _, v := range x[s : s+corpusWin] {
		acc += float64(v) * float64(v)
	}
	return acc / float64(corpusWin)
}

// classifyWindows splits x into non-overlapping corpusWin-sample windows and
// returns the quietest corpusQuietFrac as noise-floor spans and, as signal spans,
// every window at least corpusSignalMarginDB above the pooled noise floor. ok is
// false when the floor is silent or no window rises above the signal threshold
// (nothing to measure).
func classifyWindows(x []float32) (quiet, signal [][2]int, ok bool) {
	nWin := len(x) / corpusWin
	if nWin < 1 {
		return nil, nil, false
	}
	starts := make([]int, nWin)
	for w := range starts {
		starts[w] = w * corpusWin
	}
	slices.SortStableFunc(starts, func(a, b int) int {
		return cmp.Compare(windowMS(x, a), windowMS(x, b))
	})
	nq := max(1, int(float64(nWin)*corpusQuietFrac))
	for i := range nq {
		quiet = append(quiet, [2]int{starts[i], starts[i] + corpusWin})
	}
	floor := spanRMSDB(x, quiet)
	if floor <= silenceFloorDB { // silent clip: at/below the -200 sentinel, nothing to reduce
		return nil, nil, false
	}
	threshold := floor + corpusSignalMarginDB
	for w := range nWin {
		s := w * corpusWin
		if windowLevelDB(x, s) > threshold {
			signal = append(signal, [2]int{s, s + corpusWin})
		}
	}
	if len(signal) == 0 {
		return nil, nil, false
	}
	return quiet, signal, true
}

// windowLevelDB is the window's level in the same dB scale spanRMSDB uses
// (10*log10 of the mean square). A fully silent window returns the -200 sentinel,
// which sits below any real floor+margin, so it is correctly excluded from the
// signal set.
func windowLevelDB(x []float32, s int) float64 {
	ms := windowMS(x, s)
	if ms == 0 {
		return -200
	}
	return 10 * math.Log10(ms)
}

func TestCorpusAgainstAfftdn(t *testing.T) {
	bin := ffmpegPath(t) // skips when ffmpeg is absent
	dir := corpusDir()
	clips, err := filepath.Glob(filepath.Join(dir, "*.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if len(clips) == 0 {
		t.Skipf("no *.wav in %s; supply a real-clip corpus there (or set DENOISER_CORPUS_DIR) to run the A/B", dir)
	}
	lags := afftdnLags(t) // afftdn's fixed FFT delay per preset, measured once

	measured := 0
	for _, clip := range clips {
		name := filepath.Base(clip)
		in := decodeToF32Mono(t, bin, clip, corpusSampleRate)
		// The aligned afftdn reference is zero-filled over its last corpusMaxLag
		// samples (shifted pulls the signal left by the delay), so every
		// measurement window must stay clear of that tail. Classify over the
		// leading region only; Denoise and afftdn still process the whole clip.
		if len(in) <= corpusMaxLag || (len(in)-corpusMaxLag)/corpusWin < corpusMinWindows {
			t.Logf("%s: %d samples is too short for a stable A/B clear of the alignment tail; skipping", name, len(in))
			continue
		}
		measure := in[:len(in)-corpusMaxLag]
		quiet, signal, ok := classifyWindows(measure)
		if !ok {
			t.Logf("%s: no distinct noise floor and signal (silent or featureless); skipping", name)
			continue
		}
		measured++
		inQuiet := spanRMSDB(in, quiet)
		inSignal := spanRMSDB(in, signal)
		for _, preset := range []Preset{Light, Medium, Heavy} {
			t.Run(fmt.Sprintf("%s/%v", name, preset), func(t *testing.T) {
				ours, err := Denoise(in, Config{SampleRate: corpusSampleRate, Preset: preset})
				if err != nil {
					t.Fatal(err)
				}
				if len(ours) != len(in) {
					t.Fatalf("denoised length %d, want %d (offline API must stay sample-aligned)", len(ours), len(in))
				}
				nrnf := afftdnPresets[preset]
				ref := shifted(runAfftdn(t, in, corpusSampleRate, nrnf[0], nrnf[1]), lags[preset])

				oursQuiet := spanRMSDB(ours, quiet)
				redOurs := inQuiet - oursQuiet
				redRef := inQuiet - spanRMSDB(ref, quiet)
				dropOurs := inSignal - spanRMSDB(ours, signal)
				dropRef := inSignal - spanRMSDB(ref, signal)
				t.Logf("noise reduction ours %.1f / afftdn %.1f dB; signal-level drop ours %.1f / afftdn %.1f dB (afftdn lag %d)",
					redOurs, redRef, dropOurs, dropRef, lags[preset])

				// Hard, ffmpeg-independent invariants (a real bug, never oracle
				// drift). The denoiser applies per-bin gain <= 1, so it cannot
				// materially raise the noise floor; floorRiseTolDB absorbs the small
				// rise WOLA overlap-add can leak into a quiet window at a burst edge.
				// It must not gate the quiet region to the silence sentinel, which
				// would make the reduction figure vacuous.
				if redOurs < -floorRiseTolDB {
					t.Errorf("noise floor rose by %.1f dB; a gain <= 1 denoiser cannot materially raise the floor", -redOurs)
				}
				if oursQuiet <= silenceFloorDB {
					t.Errorf("quiet region collapsed to %.0f dBFS (silent/sentinel); reduction is vacuous", oursQuiet)
				}

				// afftdn A/B, hard failures (this is a manual tuning tool; a pinned
				// ffmpeg is assumed per issue #6). Match or beat afftdn on reduction,
				// and do not damage the signal more than afftdn, both within abToleranceDB.
				if redOurs < redRef-abToleranceDB {
					t.Errorf("noise reduction %.1f dB is %.1f below afftdn's %.1f (1 dB bar)", redOurs, redRef-redOurs, redRef)
				}
				if dropOurs > dropRef+abToleranceDB {
					t.Errorf("signal-level drop %.1f dB exceeds afftdn's %.1f by %.1f (1 dB bar; over-attenuating the signal)", dropOurs, dropRef, dropOurs-dropRef)
				}
			})
		}
	}
	// Positive control: a non-empty corpus with ffmpeg present must actually
	// measure something. Without this, a corpus of only too-short or featureless
	// clips reports PASS while asserting nothing (each clip only t.Logf-skips).
	if measured == 0 {
		t.Errorf("corpus at %s has %d clip(s) but none were measurable (too short, or no distinct noise floor and signal); the A/B compared nothing", dir, len(clips))
	}
}

// afftdnLags measures afftdn's alignment delay per preset once, on an
// unambiguous synthetic chirp, and reuses it for every real clip. This rests on a
// property of STFT overlap-add filters, not on afftdn's internals: their group
// delay is set by the window and hop in samples, not by the signal, so a delay
// measured at corpusSampleRate transfers to any clip decoded at that rate. It is
// reasoned from how such filters work, not measured against afftdn's source;
// measuring per preset costs nothing and covers any preset-dependent latency.
// Cross-correlating a real clip's loudest region directly would risk cycle-
// slipping on a tonal call (a periodic waveform correlates at any whole period);
// the 2-6 kHz chirp of makeSynthClip peaks sharply instead.
func afftdnLags(t *testing.T) map[Preset]int {
	t.Helper()
	clip := makeSynthClip(corpusSampleRate, -40, -20, false, 11)
	lags := make(map[Preset]int, 3)
	for _, preset := range []Preset{Light, Medium, Heavy} {
		nrnf := afftdnPresets[preset]
		ref := runAfftdn(t, clip.mix, clip.sr, nrnf[0], nrnf[1])
		lags[preset] = bestLag(clip.mix, ref, clip.signalSpans[1][0], clip.signalSpans[1][1], corpusMaxLag)
	}
	return lags
}
