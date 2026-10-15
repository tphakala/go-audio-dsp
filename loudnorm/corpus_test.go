package loudnorm

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tphakala/go-audio-dsp/gain"
)

// This file is the real-recording comparison of PlanBytes against ffmpeg's
// loudnorm filter. Synthetic tones agree trivially; the two implementations
// diverge only where R128 gating and the gain ceiling bite, which needs field
// recordings with a real noise floor. The corpus (loudnorm/testdata/corpus/*.wav,
// or LOUDNORM_CORPUS_DIR) is user-supplied and gitignored, so the comparison
// skips without it or without ffmpeg and never runs in CI. Run it with
// `task ab:loudnorm`.

const (
	// The BirdNET-Go clip policy, written out so the comparison does not depend
	// on any consumer: -23 LUFS, -2 dBTP, a 60 dB gain clamp, gate fallback on.
	corpusTargetLUFS = -23.0
	corpusCeilingTP  = -2.0
	corpusLRA        = 7.0
	corpusMaxGainDB  = 60.0

	corpusRate       = 48000
	corpusClipSecs   = 15 // the default detection clip length, far above the 400 ms gating block
	corpusClipsPerWV = 3

	// loudnormMaxOffsetDB bounds the true-peak anchored offset of ffmpeg's
	// fallback for clips R128 gates to nothing.
	loudnormMaxOffsetDB = 60.0

	// Tolerances for the checks the report makes. The meter agrees with ffmpeg's
	// to about 1e-5 LU, so the loudness tolerance is not about meter error: a
	// clip planned from a measurement just above the absolute gate gains blocks
	// that were under the gate before the boost, which lowers the integrated
	// loudness a little (0.54 LU on a 40 dB-attenuated owl clip, where ffmpeg
	// lands 0.5 LU low as well). 1 LU separates that known gating nonlinearity
	// from a real planning defect.
	corpusLoudnessTolLU = 1.0
	corpusPeakTolDB     = 0.5
)

var corpusPolicy = PlanOptions{
	Options:      Options{SampleRate: corpusRate, Channels: 1, TargetLUFS: corpusTargetLUFS, TruePeakDBTP: corpusCeilingTP},
	MaxGainDB:    corpusMaxGainDB,
	GateFallback: true,
}

type corpusCase struct {
	name string
	pcm  []byte
}

// ffloud is ffmpeg's loudnorm analysis of one file.
type ffloud struct {
	I, TP, LRA, thresh, offset string
}

func (f ffloud) float(v string) float64 {
	x, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return math.Inf(-1)
	}
	return x
}

// usable reports whether every field loudnorm's second pass consumes is finite.
func (f ffloud) usable() bool {
	for _, v := range []string{f.I, f.TP, f.LRA, f.thresh, f.offset} {
		x, err := strconv.ParseFloat(v, 64)
		if err != nil || math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// loudnormFilter builds the two-pass linear loudnorm chain the comparison treats
// as the reference: single pass when there is no analysis, a true-peak anchored
// offset when R128 gating leaves no integrated loudness, and linear two-pass
// otherwise.
func loudnormFilter(stats *ffloud) string {
	base := fmt.Sprintf("loudnorm=I=%.1f:TP=%.1f:LRA=%.1f", corpusTargetLUFS, corpusCeilingTP, corpusLRA)
	if stats == nil {
		return base
	}
	if !stats.usable() {
		tp := stats.float(stats.TP)
		if math.IsInf(tp, 0) || math.IsNaN(tp) {
			return base
		}
		offset := math.Max(-loudnormMaxOffsetDB, math.Min(loudnormMaxOffsetDB, corpusCeilingTP-tp))
		if math.Abs(offset) < 0.05 {
			return base
		}
		return fmt.Sprintf("%s:offset=%.1f", base, offset)
	}
	return fmt.Sprintf("%s:measured_I=%s:measured_LRA=%s:measured_TP=%s:measured_thresh=%s:linear=true:offset=%s",
		base, stats.I, stats.LRA, stats.TP, stats.thresh, stats.offset)
}

// TestLoudnormReferenceFilter pins the reference chain. It needs no ffmpeg and
// no corpus: a typo that dropped linear=true would put ffmpeg into dynamic mode
// and the report would show a fabricated difference.
func TestLoudnormReferenceFilter(t *testing.T) {
	const base = "loudnorm=I=-23.0:TP=-2.0:LRA=7.0"
	gated := func(tp string) *ffloud {
		return &ffloud{I: ffNegInf, TP: tp, LRA: "0.0", thresh: ffNegInf, offset: "0.0"}
	}
	cases := []struct {
		name  string
		stats *ffloud
		want  string
	}{
		{"nil stats", nil, base},
		{"two-pass", &ffloud{I: "-15.0", TP: "-1.0", LRA: "5.0", thresh: "-25.0", offset: "0.5"},
			base + ":measured_I=-15.0:measured_LRA=5.0:measured_TP=-1.0:measured_thresh=-25.0:linear=true:offset=0.5"},
		{"gated anchors to peak headroom", gated("-30.0"), base + ":offset=28.0"},
		{"gated silence", gated(ffNegInf), base},
		{"offset inside the dead band", gated("-2.02"), base},
		{"offset clamped", gated("-200.0"), base + ":offset=60.0"},
		{"unparseable field is not usable", &ffloud{I: "-15.0", TP: "-30.0", LRA: "5.0", thresh: "n/a", offset: "0.5"}, base + ":offset=28.0"},
	}
	for _, c := range cases {
		if got := loudnormFilter(c.stats); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func lookFFmpeg(t *testing.T) string {
	t.Helper()
	p, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not found in PATH; skipping the corpus comparison")
	}
	return p
}

func corpusDir() string {
	if d := os.Getenv("LOUDNORM_CORPUS_DIR"); d != "" {
		return d
	}
	return "testdata/corpus"
}

// ffNegInf is how ffmpeg prints a loudness the gate removed entirely.
const ffNegInf = "-inf"

var jsonBlock = regexp.MustCompile(`(?s)\{[^}]*"input_i"\s*:[^}]*\}`)

func analyze(ctx context.Context, t *testing.T, ffmpeg, path string) ffloud {
	t.Helper()
	filter := fmt.Sprintf("loudnorm=I=%.1f:LRA=%.1f:TP=%.1f:print_format=json", corpusTargetLUFS, corpusLRA, corpusCeilingTP)
	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, ffmpeg, "-hide_banner", "-i", path, "-af", filter, "-f", "null", "-") //nolint:gosec // G204: test-controlled binary and path
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("analyzing %s: %v\n%s", path, err, stderr.String())
	}
	m := jsonBlock.Find(stderr.Bytes())
	if m == nil {
		t.Fatalf("no loudnorm JSON for %s", path)
	}
	var raw struct {
		I      string `json:"input_i"`
		TP     string `json:"input_tp"`
		LRA    string `json:"input_lra"`
		Thresh string `json:"input_thresh"`
		Offset string `json:"target_offset"`
	}
	if err := json.Unmarshal(m, &raw); err != nil {
		t.Fatal(err)
	}
	return ffloud{I: raw.I, TP: raw.TP, LRA: raw.LRA, thresh: raw.Thresh, offset: raw.Offset}
}

func writeWAV(t *testing.T, path string, pcm []byte) string {
	t.Helper()
	var h bytes.Buffer
	h.WriteString("RIFF")
	_ = binary.Write(&h, binary.LittleEndian, uint32(36+len(pcm)))
	h.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(corpusRate), uint32(corpusRate * 2), uint16(2), uint16(16)} {
		_ = binary.Write(&h, binary.LittleEndian, v)
	}
	h.WriteString("data")
	_ = binary.Write(&h, binary.LittleEndian, uint32(len(pcm)))
	if err := os.WriteFile(path, append(h.Bytes(), pcm...), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// scaled copies int16 PCM with a linear factor, saturating like real hot audio.
func scaled(src []byte, factor float64) []byte {
	dst := make([]byte, len(src))
	for i := 0; i+1 < len(src); i += 2 {
		v := math.Round(float64(int16(binary.LittleEndian.Uint16(src[i:]))) * factor)
		binary.LittleEndian.PutUint16(dst[i:], uint16(int16(max(math.MinInt16, min(math.MaxInt16, v)))))
	}
	return dst
}

func dbFactor(db float64) float64 { return math.Pow(10, db/20) }

// loadCorpus decodes each WAV to 48 kHz mono 16-bit and expands detection-length
// clips into the loudness cases that separate the implementations.
func loadCorpus(t *testing.T, ffmpeg string) []corpusCase {
	t.Helper()
	dir := corpusDir()
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no corpus at %s (set LOUDNORM_CORPUS_DIR): %v", dir, err)
	}
	if err != nil {
		t.Fatalf("reading corpus %s: %v", dir, err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".wav") {
			names = append(names, e.Name())
		}
	}
	slices.Sort(names)
	if len(names) == 0 {
		t.Skipf("no .wav recordings in %s", dir)
	}
	clipBytes := corpusClipSecs * corpusRate * 2
	var cases []corpusCase
	for _, name := range names {
		out, err := exec.CommandContext(t.Context(), ffmpeg, "-hide_banner", "-v", "error", "-i", filepath.Join(dir, name), //nolint:gosec // G204: test-controlled
			"-t", strconv.Itoa(corpusClipSecs*corpusClipsPerWV), // only the first clips are used
			"-f", "s16le", "-acodec", "pcm_s16le", "-ar", strconv.Itoa(corpusRate), "-ac", "1", "-").Output()
		if err != nil || len(out) == 0 {
			t.Fatalf("decoding %s: %v", name, err)
		}
		base := strings.TrimSuffix(name, filepath.Ext(name))
		for i := range corpusClipsPerWV {
			if (i+1)*clipBytes > len(out) {
				break
			}
			clip := out[i*clipBytes : (i+1)*clipBytes]
			id := func(kind string) string { return fmt.Sprintf("%s#%d/%s", base, i, kind) }
			// A loud call over a quiet bed: high LRA, which pushes ffmpeg out of
			// linear mode.
			burst := scaled(clip, dbFactor(-30))
			start := (len(burst) / 3) &^ 1
			copy(burst[start:min(start+corpusRate*2, len(burst))], clip[start:])
			cases = append(cases,
				corpusCase{id("as-captured"), slices.Clone(clip)},
				corpusCase{id("quiet-20dB"), scaled(clip, dbFactor(-20))},
				corpusCase{id("quiet-40dB"), scaled(clip, dbFactor(-40))},
				corpusCase{id("hot+12dB"), scaled(clip, dbFactor(12))},
				corpusCase{id("transient-over-quiet-bed"), burst},
			)
		}
	}
	if len(cases) == 0 {
		t.Skipf("corpus recordings are shorter than %d s", corpusClipSecs)
	}
	return cases
}

// boundOf names what decided the planned gain, so a divergence from ffmpeg can
// be read as policy (clamp, gate lift) rather than as a defect.
func boundOf(p Plan) string {
	switch {
	case p.GateLifted && p.Clamped:
		return "gate-lift,clamp"
	case p.GateLifted:
		return "gate-lift"
	case p.Clamped:
		return "clamp"
	case p.PeakLimited:
		return "peak"
	case math.IsInf(p.Result.Input.IntegratedLUFS, -1):
		return "silent"
	}
	return "-"
}

func fmtDB(v float64) string {
	if math.IsInf(v, 0) {
		return "-Inf"
	}
	return fmt.Sprintf("%.2f", v)
}

// TestCorpusAgainstFFmpegLoudnorm plans each corpus case with PlanBytes, applies
// the gain with gain, and measures the result with ffmpeg alongside ffmpeg's own
// loudnorm output. It prints the table and fails only on a numeric defect: an
// unbound clip that misses the target, a gate-lifted clip lifted past the
// target, or a true peak above the ceiling.
func TestCorpusAgainstFFmpegLoudnorm(t *testing.T) {
	ffmpeg := lookFFmpeg(t)
	cases := loadCorpus(t, ffmpeg)
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Minute)
	defer cancel()

	t.Logf("targets: %.0f LUFS, %.0f dBTP, clamp %.0f dB, gate fallback on", corpusTargetLUFS, corpusCeilingTP, corpusMaxGainDB)
	header := fmt.Sprintf("%-34s | %14s | %8s | %8s | %-16s | %8s | %8s | %6s | %6s", "case", "input I/TP", "planned", "applied", "bound", "plan I", "ffmpeg I", "plan TP", "ff TP")
	t.Log(header)
	for i, c := range cases {
		stem := filepath.Join(dir, strconv.Itoa(i))
		in := writeWAV(t, stem+"-in.wav", c.pcm)
		input := analyze(ctx, t, ffmpeg, in)

		plan, err := PlanBytes(ctx, c.pcm, corpusPolicy)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		g, err := gain.New(plan.GainDB)
		if err != nil {
			t.Fatal(err)
		}
		out := slices.Clone(c.pcm)
		if err := g.ApplyBytes(out); err != nil {
			t.Fatal(err)
		}
		ours := analyze(ctx, t, ffmpeg, writeWAV(t, stem+"-plan.wav", out))

		ffOut := stem + "-ffmpeg.flac"
		var stderr bytes.Buffer
		cmd := exec.CommandContext(ctx, ffmpeg, "-hide_banner", "-loglevel", "error", "-i", in, "-af", loudnormFilter(&input), //nolint:gosec // G204: test-controlled
			"-ar", strconv.Itoa(corpusRate), "-c:a", "flac", "-y", ffOut)
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("%s: ffmpeg loudnorm: %v\n%s", c.name, err, stderr.String())
		}
		theirs := analyze(ctx, t, ffmpeg, ffOut)

		bound := boundOf(plan)
		t.Logf("%-34s | %6s/%6s | %8s | %8s | %-16s | %8s | %8s | %6s | %6s", c.name,
			fmtDB(input.float(input.I)), fmtDB(input.float(input.TP)), fmtDB(plan.PlannedGainDB), fmtDB(plan.GainDB), bound,
			fmtDB(ours.float(ours.I)), fmtDB(theirs.float(theirs.I)), fmtDB(ours.float(ours.TP)), fmtDB(theirs.float(theirs.TP)))

		gotI, gotTP := ours.float(ours.I), ours.float(ours.TP)
		if bound == "-" && math.Abs(gotI-corpusTargetLUFS) > corpusLoudnessTolLU {
			t.Errorf("%s: unbound clip measures %.2f LUFS, want %.0f +/-%.1f", c.name, gotI, corpusTargetLUFS, corpusLoudnessTolLU)
		}
		if strings.HasPrefix(bound, "gate-lift") && gotI > corpusTargetLUFS+corpusLoudnessTolLU {
			t.Errorf("%s: gate-lifted clip measures %.2f LUFS, past the %.0f target", c.name, gotI, corpusTargetLUFS)
		}
		if plan.GainDB != 0 && gotTP > corpusCeilingTP+corpusPeakTolDB && plan.GainDB > 0 {
			t.Errorf("%s: boosted clip peaks at %.2f dBTP, above the %.0f ceiling", c.name, gotTP, corpusCeilingTP)
		}
	}
}
