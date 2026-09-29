package audiotest

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Level is one comparison point in an afftdn A/B: a denoise function under test
// paired with the afftdn settings it is measured against.
type Level struct {
	// Name labels the subtest (for example the strength).
	Name string
	// NR and NF are the afftdn noise_reduction (dB) and noise_floor (dBFS)
	// options that form the baseline for this level.
	NR, NF int
	// Denoise runs the method under test over the whole clip. It must return a
	// buffer of the same length, sample-aligned with the input.
	Denoise func(in []float32) ([]float32, error)
}

// CorpusEnv names the environment variable that overrides the corpus directory
// for a corpus kept outside the repo.
const CorpusEnv = "DENOISER_CORPUS_DIR"

// CorpusDir returns the directory holding the user-supplied A/B corpus. It is
// gitignored (testdata/corpus) and absent in CI; CorpusEnv overrides the
// location. def is the default, relative to the calling package's directory.
func CorpusDir(def string) string {
	return dirFromEnv(CorpusEnv, def)
}

// dirFromEnv returns the directory named by env when set, else def.
func dirFromEnv(env, def string) string {
	if d := os.Getenv(env); d != "" {
		return d
	}
	return def
}

// AfftdnLag measures afftdn's alignment delay for one setting, once, on an
// unambiguous synthetic chirp, so it can be reused for every real clip. This
// rests on a property of STFT overlap-add filters, not on afftdn's internals:
// their group delay is set by the window and hop in samples, not by the signal,
// so a delay measured at corpusSampleRate transfers to any clip decoded at that
// rate. Cross-correlating a real clip's loudest region directly would risk
// cycle-slipping on a tonal call (a periodic waveform correlates at any whole
// period); the 2-6 kHz chirp of MakeSynthClip peaks sharply instead.
func AfftdnLag(t *testing.T, nr, nf int) int {
	t.Helper()
	clip := MakeSynthClip(corpusSampleRate, -40, -20, false, 11)
	ref := RunAfftdn(t, clip.Mix, clip.SR, nr, nf)
	return BestLag(clip.Mix, ref, clip.SignalSpans[1][0], clip.SignalSpans[1][1], corpusMaxLag)
}

// RunCorpusAB is the real-corpus A/B comparison of a denoise method against
// ffmpeg's afftdn (the baseline BirdNET-Go used). It runs every *.wav in dir
// through each Level's Denoise and through afftdn, and compares two
// reference-free metrics per clip per level:
//
//   - noise reduction: the level drop in the clip's quietest windows (its noise
//     floor). Higher is better; the method should match or beat afftdn.
//   - signal retention: the level drop in the loudest, high-SNR windows. Those
//     windows are signal-dominated, so their level barely moves when only noise
//     is removed; a large drop means the method is eating the signal. It should
//     not lose more than afftdn.
//
// Both are pooled RMS levels (SpanRMSDB), not spectral distances: real clips
// have no clean reference, so a distance-from-input metric would just re-measure
// noise removal and penalize the method for working. Spectral quality (musical
// noise) is a listening judgment this test cannot make.
//
// The corpus is user-supplied and gitignored, so this never runs in CI: it skips
// when the corpus or ffmpeg is absent. It is a manual tuning tool, so a miss
// against afftdn fails loudly. Authoritative numbers need a pinned ffmpeg (see
// issue #6). tolDB is how far the method may trail afftdn on either metric; a
// non-positive value selects the default of 1 dB.
func RunCorpusAB(t *testing.T, dir string, levels []Level, tolDB float64) {
	t.Helper()
	if tolDB <= 0 {
		tolDB = abToleranceDB
	}
	bin := FFmpegPath(t) // skips when ffmpeg is absent
	clips, err := filepath.Glob(filepath.Join(dir, "*.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if len(clips) == 0 {
		t.Skipf("no *.wav in %s; supply a real-clip corpus there (or set %s) to run the A/B", dir, CorpusEnv)
	}
	lags := make([]int, len(levels))
	for i, lv := range levels {
		lags[i] = AfftdnLag(t, lv.NR, lv.NF)
	}

	measured := 0
	for _, clip := range clips {
		name := filepath.Base(clip)
		in := DecodeToF32Mono(t, bin, clip, corpusSampleRate)
		// The aligned afftdn reference is zero-filled over its last corpusMaxLag
		// samples (Shifted pulls the signal left by the delay), so every
		// measurement window must stay clear of that tail. Classify over the
		// leading region only; the method and afftdn still process the whole clip.
		if len(in) <= corpusMaxLag || (len(in)-corpusMaxLag)/corpusWin < corpusMinWindows {
			t.Logf("%s: %d samples is too short for a stable A/B clear of the alignment tail; skipping", name, len(in))
			continue
		}
		quiet, signal, ok := ClassifyWindows(in[:len(in)-corpusMaxLag])
		if !ok {
			t.Logf("%s: no distinct noise floor and signal (silent or featureless); skipping", name)
			continue
		}
		measured++
		inQuiet := SpanRMSDB(in, quiet)
		inSignal := SpanRMSDB(in, signal)
		for i, lv := range levels {
			t.Run(fmt.Sprintf("%s/%s", name, lv.Name), func(t *testing.T) {
				ours, err := lv.Denoise(in)
				if err != nil {
					t.Fatal(err)
				}
				if len(ours) != len(in) {
					t.Fatalf("denoised length %d, want %d (offline API must stay sample-aligned)", len(ours), len(in))
				}
				AssertFinite(t, "denoised output", ours)
				ref := Shifted(RunAfftdn(t, in, corpusSampleRate, lv.NR, lv.NF), lags[i])
				AssertFinite(t, "afftdn reference", ref)

				oursQuiet := SpanRMSDB(ours, quiet)
				redOurs := inQuiet - oursQuiet
				redRef := inQuiet - SpanRMSDB(ref, quiet)
				dropOurs := inSignal - SpanRMSDB(ours, signal)
				dropRef := inSignal - SpanRMSDB(ref, signal)
				t.Logf("noise reduction ours %.1f / afftdn %.1f dB; signal-level drop ours %.1f / afftdn %.1f dB (afftdn lag %d)",
					redOurs, redRef, dropOurs, dropRef, lags[i])

				// Hard, ffmpeg-independent invariants (a real bug, never oracle
				// drift). A gain <= 1 method cannot materially raise the noise
				// floor; floorRiseTolDB absorbs the small rise WOLA overlap-add
				// can leak into a quiet window at a burst edge. It must not gate
				// the quiet region to the silence sentinel, which would make the
				// reduction figure vacuous.
				if redOurs < -floorRiseTolDB {
					t.Errorf("noise floor rose by %.1f dB; a gain <= 1 denoiser cannot materially raise the floor", -redOurs)
				}
				if oursQuiet <= silenceFloorDB {
					t.Errorf("quiet region collapsed to %.0f dBFS (silent/sentinel); reduction is vacuous", oursQuiet)
				}
				if redOurs < redRef-tolDB {
					t.Errorf("noise reduction %.1f dB is %.1f below afftdn's %.1f (%.1f dB bar)", redOurs, redRef-redOurs, redRef, tolDB)
				}
				if dropOurs > dropRef+tolDB {
					t.Errorf("signal-level drop %.1f dB exceeds afftdn's %.1f by %.1f (%.1f dB bar; over-attenuating the signal)", dropOurs, dropRef, dropOurs-dropRef, tolDB)
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
