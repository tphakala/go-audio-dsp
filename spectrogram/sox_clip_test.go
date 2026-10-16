package spectrogram

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// This file holds the reference input and the list of sox renders that
// spectrogram/testdata/sox carries. The parity tests that compare Fit with those
// renders are in sox_test.go.

// hannName is the sox and benchmark label of the Hann window.
const hannName = "hann"

const (
	soxClipRate    = 24000
	soxClipSamples = 24000 // 1 s
	soxDir         = "testdata/sox"
)

// soxClip returns the reference input: 1 s at 24 kHz as int16 samples. It mixes
// fixed-seed noise, a linear chirp, two tones, a mid-clip digital silence and a
// click, so a level, a frequency or an edge bug shows up in the render. The clip
// end still holds signal, so the drain frames are not silent. The noise comes from
// an integer xorshift PRNG with a fixed seed and the tones from math.Sin, so the
// output does not depend on the run; a sample may differ by one LSB between
// targets (see TestSoxClipMatchesGenerator).
func soxClip() []int16 {
	const (
		noiseAmp   = 0.01
		chirpAmp   = 0.3
		chirpFrom  = 500.0
		chirpTo    = 11000.0
		toneAmp    = 0.1
		toneHz     = 3000.0
		edgeAmp    = 0.003 // near the -z 100 floor
		edgeHz     = 7777.0
		clickAmp   = 0.9
		clickAt    = soxClipSamples / 3
		silenceLo  = soxClipSamples * 45 / 100
		silenceHi  = soxClipSamples * 60 / 100
		durationS  = float64(soxClipSamples) / soxClipRate
		chirpSlope = (chirpTo - chirpFrom) / (2 * durationS)
	)
	x := uint64(0x9E3779B97F4A7C15)
	out := make([]int16, soxClipSamples)
	for n := range out {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		noise := float64(x>>11)/(1<<53)*2 - 1
		t := float64(n) / soxClipRate
		v := noiseAmp*noise +
			chirpAmp*math.Sin(2*math.Pi*(chirpFrom*t+chirpSlope*t*t)) +
			toneAmp*math.Sin(2*math.Pi*toneHz*t) +
			edgeAmp*math.Sin(2*math.Pi*edgeHz*t)
		if n >= silenceLo && n < silenceHi {
			v = 0
		}
		if n == clickAt {
			v += clickAmp
		}
		out[n] = int16(max(-32768, min(32767, math.Round(v*32768))))
	}
	return out
}

// wavBytes encodes mono 16-bit PCM in a canonical 44-byte header WAV.
func wavBytes(pcm []int16, rate int) []byte {
	var b bytes.Buffer
	dataLen := uint32(len(pcm) * 2)
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, 36+dataLen)
	b.WriteString("WAVEfmt ")
	_ = binary.Write(&b, binary.LittleEndian, struct {
		Size          uint32
		Format        uint16
		Channels      uint16
		Rate          uint32
		BytesPerSec   uint32
		BlockAlign    uint16
		BitsPerSample uint16
	}{16, 1, 1, uint32(rate), uint32(rate) * 2, 2, 16})
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, dataLen)
	_ = binary.Write(&b, binary.LittleEndian, pcm)
	return b.Bytes()
}

// readWAV decodes the canonical 44-byte header mono 16-bit WAV that wavBytes
// writes. It is a test helper, not a general reader.
func readWAV(tb testing.TB, path string) (pcm []int16, rate int) {
	tb.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatal(err)
	}
	if len(raw) < 44 || string(raw[0:4]) != "RIFF" || string(raw[8:16]) != "WAVEfmt " || string(raw[36:40]) != "data" {
		tb.Fatalf("%s: not a 44-byte header WAV", path)
	}
	if binary.LittleEndian.Uint16(raw[20:]) != 1 || binary.LittleEndian.Uint16(raw[22:]) != 1 || binary.LittleEndian.Uint16(raw[34:]) != 16 {
		tb.Fatalf("%s: not mono 16-bit PCM", path)
	}
	rate = int(binary.LittleEndian.Uint32(raw[24:]))
	n := int(binary.LittleEndian.Uint32(raw[40:])) / 2
	if len(raw) < 44+2*n {
		tb.Fatalf("%s: data chunk truncated", path)
	}
	pcm = make([]int16, n)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[44+2*i:]))
	}
	return pcm, rate
}

// soxRef is one sox render of the clip.
type soxRef struct {
	name       string // PNG file name without extension
	x, y       int    // sox -x and -y: column limit and bins (FrameSize = 2*(y-1))
	span       string // sox -d argument
	spanSample int64  // the same span in samples
	z          int    // sox -z, the dynamic range in dB
	dolph      bool   // -w dolph instead of -w hann
	trim       int    // samples kept by a leading "trim 0 Ns" (0: no trim)
	liveOnly   bool   // not committed; only TestSoxLive renders it
}

// frameSize is sox's dft size, 2*(y-1).
func (r soxRef) frameSize() int { return 2 * (r.y - 1) }

// args returns the sox command line after the program name, exactly as the
// testdata README lists it.
func (r soxRef) args() []string {
	a := []string{"clip.wav", "-n"}
	if r.trim > 0 {
		a = append(a, "trim", "0", fmt.Sprintf("%ds", r.trim))
	}
	w := hannName
	if r.dolph {
		w = "dolph"
	}
	return append(a, "spectrogram",
		"-x", fmt.Sprint(r.x), "-y", fmt.Sprint(r.y), "-d", r.span,
		"-z", fmt.Sprint(r.z), "-w", w, "-o", r.name+".png", "-r")
}

// command returns the README form of the command line.
func (r soxRef) command() string { return "sox " + strings.Join(r.args(), " ") }

// soxRefs lists every render. The committed ones carry the main parity
// assertions; the liveOnly ones run only when sox is installed.
func soxRefs() []soxRef {
	const sec15 = 15 * soxClipRate
	refs := make([]soxRef, 0, 24)
	geoms := []struct {
		tag  string
		x, y int
	}{{"sm", 258, 129}, {"md", 514, 257}, {"lg", 1026, 513}, {"xl", 2050, 1025}}
	for _, g := range geoms {
		refs = append(refs, soxRef{name: "hann-" + g.tag, x: g.x, y: g.y, span: "15", spanSample: sec15, z: 100})
	}
	for _, z := range []int{20, 80, 100, 120} {
		refs = append(refs, soxRef{name: fmt.Sprintf("dolph-sm-z%d", z), x: 258, y: 129, span: "15", spanSample: sec15, z: z, dolph: true})
	}
	for _, z := range []int{80, 100, 120} {
		refs = append(refs, soxRef{name: fmt.Sprintf("dolph-lg-z%d", z), x: 1026, y: 513, span: "15", spanSample: sec15, z: z, dolph: true})
	}
	refs = append(refs, soxRef{name: "hann-sm-trunc", x: 100, y: 129, span: "6000s", spanSample: 6000, z: 100})
	for _, l := range []int{175, 200, 1000} {
		refs = append(refs, soxRef{name: fmt.Sprintf("short%d-lg-hann", l), x: 1026, y: 513, span: "15", spanSample: sec15, z: 100, trim: l})
	}
	for _, l := range []int{175, 200, 1000} {
		refs = append(refs, soxRef{name: fmt.Sprintf("short%d-xl-dolph", l), x: 2050, y: 1025, span: "15", spanSample: sec15, z: 100, dolph: true, trim: l})
	}
	// Renders that are compared live only, to keep the committed data small.
	refs = append(refs,
		soxRef{name: "live-dolph-md-z100", x: 514, y: 257, span: "15", spanSample: sec15, z: 100, dolph: true, liveOnly: true},
		soxRef{name: "live-dolph-sm-z40", x: 258, y: 129, span: "15", spanSample: sec15, z: 40, dolph: true, liveOnly: true},
		soxRef{name: "live-dolph-sm-z60", x: 258, y: 129, span: "15", spanSample: sec15, z: 60, dolph: true, liveOnly: true},
		soxRef{name: "live-dolph-xl-z120", x: 2050, y: 1025, span: "15", spanSample: sec15, z: 120, dolph: true, liveOnly: true},
		soxRef{name: "live-dolph-lg-z20", x: 1026, y: 513, span: "15", spanSample: sec15, z: 20, dolph: true, liveOnly: true},
		soxRef{name: "live-hann-sm-cap", x: 10000, y: 129, span: "1", spanSample: soxClipRate, z: 100, liveOnly: true},
		soxRef{name: "live-hann-md-d2", x: 514, y: 257, span: "2", spanSample: 2 * soxClipRate, z: 100, liveOnly: true},
	)
	return refs
}

// runSoxErr runs sox in dir with args and returns its error with the output.
func runSoxErr(dir string, args []string) error {
	cmd := exec.Command("sox", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("sox %s: %w\n%s", strings.Join(args, " "), err, out)
	}
	return nil
}

// runSox runs sox in dir with args and fails the test on error. Callers run one
// sox at a time.
func runSox(tb testing.TB, dir string, args []string) {
	tb.Helper()
	if err := runSoxErr(dir, args); err != nil {
		tb.Fatal(err)
	}
}

func haveSox() bool {
	_, err := exec.LookPath("sox")
	return err == nil
}

// TestRegenerateSoxReferences rewrites testdata/sox/clip.wav and the committed
// PNGs. It runs only with SOX_REGEN=1 and sox on PATH.
func TestRegenerateSoxReferences(t *testing.T) {
	if os.Getenv("SOX_REGEN") != "1" {
		t.Skip("set SOX_REGEN=1 to regenerate the sox reference renders")
	}
	if !haveSox() {
		t.Skip("sox not on PATH")
	}
	if err := os.WriteFile(filepath.Join(soxDir, "clip.wav"), wavBytes(soxClip(), soxClipRate), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, r := range soxRefs() {
		if r.liveOnly {
			continue
		}
		runSox(t, soxDir, r.args())
	}
}

// TestSoxClipMatchesGenerator pins the committed clip to soxClip. The generator
// uses math.Sin, whose last bit may differ across targets (for example through
// fused multiply-add), so a sample may differ by one LSB; anything larger is a
// real change of the recipe.
func TestSoxClipMatchesGenerator(t *testing.T) {
	got, rate := readWAV(t, filepath.Join(soxDir, "clip.wav"))
	want := soxClip()
	if rate != soxClipRate || len(got) != len(want) {
		t.Fatalf("clip.wav has rate %d and %d samples, want %d and %d", rate, len(got), soxClipRate, len(want))
	}
	differ := 0
	for i := range want {
		d := int(got[i]) - int(want[i])
		if d < -1 || d > 1 {
			t.Fatalf("sample %d: committed %d, generated %d", i, got[i], want[i])
		}
		if d != 0 {
			differ++
		}
	}
	t.Logf("%d of %d samples differ by one LSB", differ, len(want))
}
