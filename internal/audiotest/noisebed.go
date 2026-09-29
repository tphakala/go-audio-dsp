package audiotest

import (
	"fmt"
	"path/filepath"
	"slices"
	"testing"
)

// NoiseBedEnv names the environment variable that overrides the noise-bed
// directory.
const NoiseBedEnv = "AUDIOTEST_NOISEBED_DIR"

// NoiseBedDir returns the directory of recorded noise beds (*.wav), gitignored
// and local only. NoiseBedEnv overrides def, which is relative to the calling
// package's directory.
func NoiseBedDir(def string) string {
	return dirFromEnv(NoiseBedEnv, def)
}

// quietestSlice returns the n-sample slice of x with the lowest energy, scanned
// at one-second steps, or nil if x is shorter than n. A recorded clip may hold a
// call; the quietest stretch is the best available noise-only bed.
func quietestSlice(x []float32, sr, n int) []float32 {
	if len(x) < n {
		return nil
	}
	best, bestE := 0, -1.0
	for s := 0; s+n <= len(x); s += sr {
		var e float64
		for _, v := range x[s : s+n] {
			e += float64(v) * float64(v)
		}
		if bestE < 0 || e < bestE {
			best, bestE = s, e
		}
	}
	return x[best : best+n]
}

// minBedDB is the quietest noise bed, in dBFS RMS, the comparison accepts: the
// lowest noise_floor afftdn takes.
const minBedDB = -80.0

// loadNoiseBed decodes one recording and returns its quietest synthSeconds as a
// noise bed. When the recording cannot serve as a bed it returns a nil bed and
// the reason: shorter than the bed length, or below minBedDB.
func loadNoiseBed(t *testing.T, bin, path string, sr int) (bed []float32, skip string) {
	t.Helper()
	bed = quietestSlice(DecodeToF32Mono(t, bin, path, sr), sr, synthSeconds*sr)
	if bed == nil {
		return nil, fmt.Sprintf("shorter than %d s", synthSeconds)
	}
	bed = slices.Clone(bed)
	AssertFinite(t, filepath.Base(path), bed)
	if level, quiet := bedTooQuiet(bed); quiet {
		return nil, fmt.Sprintf("quietest %d s is %.0f dBFS, below the %.0f dBFS afftdn can act on", synthSeconds, level, minBedDB)
	}
	return bed, ""
}

// bedTooQuiet reports whether a noise bed is below minBedDB and returns its
// level in dBFS RMS. Below afftdn's lowest noise_floor the reference does almost
// nothing, so every bar would pass without measuring anything. An all-zero or
// empty bed is -200 dBFS and always too quiet.
func bedTooQuiet(bed []float32) (level float64, tooQuiet bool) {
	level = RMSDB(bed)
	return level, level < minBedDB
}

// Bars are the margins, in dB, a method may trail afftdn on the mixed-noise A/B.
type Bars struct {
	// ReductionDB: noise-span reduction may be at most this far below afftdn's.
	ReductionDB float64
	// LSDDB: signal-span log-spectral distance from the clean signal may exceed
	// afftdn's by at most this much.
	LSDDB float64
}

// RunNoiseBedAB compares a denoise method against afftdn on realistic noise with
// a known clean reference. For every *.wav in dir it takes the quietest 10 s of
// the recording as a noise bed, lays the four synthetic SynthClip bursts over it
// at snrDB above the bed's RMS level (MixBursts), and runs each Level over the
// mix. Unlike RunCorpusAB the signal is known, so the comparison uses distance
// metrics. It fails when the noise floor rises, when noise-span reduction or the
// signal-span log-spectral distance from the clean bursts misses the bars
// against afftdn, or when that distance is more than 1 dB worse than the
// untouched input's. Segmental SNR is logged only. It skips when no bed or
// ffmpeg is present. Beds shorter than 10 s, or whose quietest 10 s is below
// minBedDB, are skipped.
func RunNoiseBedAB(t *testing.T, dir string, snrDB float64, levels []Level, bars Bars) {
	t.Helper()
	bin := FFmpegPath(t)
	beds, err := filepath.Glob(filepath.Join(dir, "*.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if len(beds) == 0 {
		t.Skipf("no *.wav in %s; supply a recorded noise bed there (or set %s) to run the mixed A/B", dir, NoiseBedEnv)
	}
	const sr = corpusSampleRate
	lags := make([]int, len(levels))
	for i, lv := range levels {
		lags[i] = AfftdnLag(t, lv.NR, lv.NF)
	}
	measured := 0
	for _, path := range beds {
		name := filepath.Base(path)
		bed, skip := loadNoiseBed(t, bin, path, sr)
		if skip != "" {
			t.Logf("%s: %s; skipping", name, skip)
			continue
		}
		clip := MixBursts(sr, bed, RMSDB(bed)+snrDB)
		measured++
		segBefore := SegSNRDB(clip.Clean, clip.Mix, clip.SignalSpans, 960)
		for i, lv := range levels {
			t.Run(fmt.Sprintf("%s/%s", name, lv.Name), func(t *testing.T) {
				ours, err := lv.Denoise(clip.Mix)
				if err != nil {
					t.Fatal(err)
				}
				if len(ours) != len(clip.Mix) {
					t.Fatalf("denoised length %d, want %d", len(ours), len(clip.Mix))
				}
				AssertFinite(t, "denoised output", ours)
				ref := Shifted(RunAfftdn(t, clip.Mix, sr, lv.NR, lv.NF), lags[i])
				AssertFinite(t, "afftdn reference", ref)

				inNoise := SpanRMSDB(clip.Mix, clip.NoiseSpans)
				redOurs := inNoise - SpanRMSDB(ours, clip.NoiseSpans)
				redRef := inNoise - SpanRMSDB(ref, clip.NoiseSpans)
				lsdOurs := LSDDB(clip.Clean, ours, clip.SignalSpans, 1024, 256)
				lsdRef := LSDDB(clip.Clean, ref, clip.SignalSpans, 1024, 256)
				lsdIn := LSDDB(clip.Clean, clip.Mix, clip.SignalSpans, 1024, 256)
				segOurs := SegSNRDB(clip.Clean, ours, clip.SignalSpans, 960)
				segRef := SegSNRDB(clip.Clean, ref, clip.SignalSpans, 960)
				t.Logf("noise reduction ours %.1f / afftdn %.1f dB; signal LSD ours %.2f / afftdn %.2f / input %.2f dB; segSNR ours %.1f / afftdn %.1f / input %.1f dB (afftdn lag %d)",
					redOurs, redRef, lsdOurs, lsdRef, lsdIn, segOurs, segRef, segBefore, lags[i])

				if redOurs < -floorRiseTolDB {
					t.Errorf("noise floor rose by %.1f dB", -redOurs)
				}
				if redOurs < redRef-bars.ReductionDB {
					t.Errorf("noise reduction %.1f dB is %.1f below afftdn's %.1f (%.1f dB bar)", redOurs, redRef-redOurs, redRef, bars.ReductionDB)
				}
				if lsdOurs > lsdRef+bars.LSDDB {
					t.Errorf("signal LSD %.2f dB exceeds afftdn's %.2f by %.2f (%.1f dB bar)", lsdOurs, lsdRef, lsdOurs-lsdRef, bars.LSDDB)
				}
				if lsdOurs > lsdIn+1 {
					t.Errorf("signal LSD %.2f dB is worse than the untouched input's %.2f; the method is damaging the signal", lsdOurs, lsdIn)
				}
			})
		}
	}
	if measured == 0 {
		t.Errorf("noise beds in %s (%d file(s)) were all too short or too quiet; the A/B compared nothing", dir, len(beds))
	}
}
