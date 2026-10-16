package spectrogram

import (
	"bytes"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// Parity thresholds against the sox palette indices. A palette index is the
// truncated 8-bit position of a dB value in the colour ramp, so a value that sits
// within float32 rounding of a step boundary lands one index off; larger
// differences would mean a real level or geometry error.
const (
	soxMinExact   = 0.999 // fraction of cells with the same index
	soxMaxIdxDiff = 1
)

// soxFloat returns the clip as float32 in [-1, 1), the scale sox reads 16-bit
// samples at.
func soxFloat(pcm []int16) []float32 {
	out := make([]float32, len(pcm))
	for i, v := range pcm {
		out[i] = float32(v) / 32768
	}
	return out
}

// soxFit builds the Fit the render r corresponds to.
func soxFit(tb testing.TB, r soxRef) *Fit {
	tb.Helper()
	shape := stft.WindowShape{Family: stft.ShapeHann}
	if r.dolph {
		shape = stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: DolphAttenuationDB(float64(r.z), 0, 0)}
	}
	f, err := NewFit(FitConfig{SampleRate: soxClipRate, FrameSize: r.frameSize(), Columns: r.x, Window: shape})
	if err != nil {
		tb.Fatalf("%s: NewFit: %v", r.name, err)
	}
	return f
}

// soxPaletteIndex is the palette index of the raw render for a value of x dB with
// dynamic range dr: the ramp starts at palette entry 4, values below -dr take its
// first step, values at or above 0 dB its last (step 250), and the range between
// maps linearly onto steps 1 to 249, truncated.
func soxPaletteIndex(x, dr float64) uint8 {
	switch {
	case x < -dr:
		return 4
	case x >= 0:
		return 4 + 250
	}
	return 4 + uint8(1+(1+x/dr)*249)
}

func readSoxPNG(tb testing.TB, path string) *image.Paletted {
	tb.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatal(err)
	}
	return decodeSoxPNG(tb, raw, path)
}

func decodeSoxPNG(tb testing.TB, raw []byte, name string) *image.Paletted {
	tb.Helper()
	img, err := png.Decode(bytes.NewReader(raw))
	if err != nil {
		tb.Fatalf("%s: %v", name, err)
	}
	p, ok := img.(*image.Paletted)
	if !ok {
		tb.Fatalf("%s: decoded as %T, want *image.Paletted", name, img)
	}
	return p
}

// parityResult summarizes a column-by-column comparison with a render.
type parityResult struct {
	cells, exact, maxDiff int
}

func (p parityResult) exactFrac() float64 { return float64(p.exact) / float64(p.cells) }

// compareToSox renders signal through f over span and compares every cell with
// the render. Row y of the image holds bin Bins-1-y.
func compareToSox(tb testing.TB, f *Fit, r soxRef, signal []float32, img *image.Paletted) parityResult {
	tb.Helper()
	m, err := f.Compute(signal, r.spanSample)
	if err != nil {
		tb.Fatalf("%s: Compute: %v", r.name, err)
	}
	b := img.Bounds()
	if m.Frames != b.Dx() || m.Bins != b.Dy() {
		tb.Fatalf("%s: Fit gave %d columns x %d bins, the render is %d x %d", r.name, m.Frames, m.Bins, b.Dx(), b.Dy())
	}
	var res parityResult
	for c := range m.Frames {
		for y := range m.Bins {
			want := img.Pix[y*img.Stride+c]
			got := soxPaletteIndex(float64(m.At(m.Bins-1-y, c)), float64(r.z))
			d := int(got) - int(want)
			if d < 0 {
				d = -d
			}
			res.cells++
			if d == 0 {
				res.exact++
			}
			res.maxDiff = max(res.maxDiff, d)
		}
	}
	return res
}

func (p parityResult) check(t *testing.T, name string) {
	t.Helper()
	t.Logf("%s: %d cells, %.4f%% exact, max index diff %d", name, p.cells, 100*p.exactFrac(), p.maxDiff)
	if p.exactFrac() < soxMinExact || p.maxDiff > soxMaxIdxDiff {
		t.Errorf("%s: %.4f%% exact (want >= %.1f%%), max index diff %d (want <= %d)", name, 100*p.exactFrac(), 100*soxMinExact, p.maxDiff, soxMaxIdxDiff)
	}
}

// soxSignal returns the clip, cut to the render's trim.
func soxSignal(tb testing.TB, r soxRef) []float32 {
	tb.Helper()
	pcm, rate := readWAV(tb, filepath.Join(soxDir, "clip.wav"))
	if rate != soxClipRate {
		tb.Fatalf("clip.wav rate %d, want %d", rate, soxClipRate)
	}
	if r.trim > 0 {
		pcm = pcm[:r.trim]
	}
	return soxFloat(pcm)
}

// runParity compares Fit with every committed render that match accepts.
func runParity(t *testing.T, match func(name string) bool) {
	t.Helper()
	n := 0
	for _, r := range soxRefs() {
		if r.liveOnly || !match(r.name) {
			continue
		}
		n++
		t.Run(r.name, func(t *testing.T) {
			img := readSoxPNG(t, filepath.Join(soxDir, r.name+".png"))
			compareToSox(t, soxFit(t, r), r, soxSignal(t, r), img).check(t, r.name)
		})
	}
	if n == 0 {
		t.Fatal("no committed render matches")
	}
}

func isOneOf(names ...string) func(string) bool {
	return func(s string) bool { return slices.Contains(names, s) }
}

func hasPrefix(prefix string) func(string) bool {
	return func(s string) bool { return strings.HasPrefix(s, prefix) }
}

// TestFitSoxParityHann compares Fit with the sox Hann renders at the four
// BirdNET-Go geometries (frame sizes 256 to 2048).
func TestFitSoxParityHann(t *testing.T) {
	runParity(t, isOneOf("hann-sm", "hann-md", "hann-lg", "hann-xl"))
}

// TestFitSoxParityDolph compares Fit with the sox Dolph-Chebyshev renders at
// 20 to 120 dB of dynamic range.
func TestFitSoxParityDolph(t *testing.T) {
	runParity(t, hasPrefix("dolph-"))
}

// TestFitSoxEdges covers the clip edges: a span shorter than the clip (columns
// truncated at the limit), clips shorter than a frame (every frame holds lead
// zeros, drain zeros or both), and, in the Hann and Dolph tests above, a span
// longer than the clip.
func TestFitSoxEdges(t *testing.T) {
	runParity(t, func(n string) bool { return n == "hann-sm-trunc" || strings.HasPrefix(n, "short") })
}

// TestFitSoxEdgesZeroColumns pins the clips that give no column. sox writes no
// image for these (TestSoxLive checks that where sox is installed).
func TestFitSoxEdgesZeroColumns(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	const span = 15 * soxClipRate
	lg := soxRef{name: "lg", x: 1026, y: 513, spanSample: span, z: 100}
	cases := []struct {
		ref soxRef
		l   int
	}{
		{lg, 1},
		{lg, 100},
		{lg, 174},
		{soxRef{name: "sm", x: 258, y: 129, spanSample: span, z: 100, dolph: true}, 1},
		{soxRef{name: "xl", x: 2050, y: 1025, spanSample: span, z: 100, dolph: true}, 1},
	}
	for _, c := range cases {
		f := soxFit(t, c.ref)
		if err := f.Reset(c.ref.spanSample); err != nil {
			t.Fatal(err)
		}
		calls := 0
		f.Feed(clip[:c.l], func([]float32, int) { calls++ })
		f.Finish(func([]float32, int) { calls++ })
		if calls != 0 || f.NumColumns(int64(c.l)) != 0 {
			t.Errorf("%s L=%d: %d callbacks, NumColumns %d, want none", c.ref.name, c.l, calls, f.NumColumns(int64(c.l)))
		}
		m, err := f.Compute(clip[:c.l], c.ref.spanSample)
		if err != nil || m.Frames != 0 || len(m.Data) != 0 {
			t.Errorf("%s L=%d: Compute = (%d columns, %d values, %v), want an empty matrix", c.ref.name, c.l, m.Frames, len(m.Data), err)
		}
	}
}

// TestSoxReadmeListsCommands keeps the README command lines equal to the ones the
// regeneration and live tests run.
func TestSoxReadmeListsCommands(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(soxDir, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	readme := string(raw)
	for _, r := range soxRefs() {
		if !strings.Contains(readme, r.command()+"\n") {
			t.Errorf("README.md does not list %q on a line of its own", r.command())
		}
	}
}

// soxTempDir returns a fresh directory holding a copy of the reference clip, for
// sox to write its renders into.
func soxTempDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	clip, err := os.ReadFile(filepath.Join(soxDir, "clip.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "clip.wav"), clip, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestSoxLive re-renders every reference with the installed sox and checks that
// the committed images are what this sox build writes, then compares Fit with the
// renders that are not committed. It skips without sox.
func TestSoxLive(t *testing.T) {
	if !haveSox() {
		t.Skip("sox not on PATH")
	}
	dir := soxTempDir(t)
	for _, r := range soxRefs() {
		runSox(t, dir, r.args())
		live := readSoxPNG(t, filepath.Join(dir, r.name+".png"))
		if !r.liveOnly {
			committed := readSoxPNG(t, filepath.Join(soxDir, r.name+".png"))
			if live.Bounds() != committed.Bounds() || !bytes.Equal(live.Pix, committed.Pix) {
				t.Errorf("%s: this sox build renders different indices than the committed image", r.name)
			}
			continue
		}
		t.Run(r.name, func(t *testing.T) {
			compareToSox(t, soxFit(t, r), r, soxSignal(t, r), live).check(t, r.name)
		})
	}
}

// TestSoxLiveEmpty checks the clips that give zero columns in
// TestFitSoxEdgesZeroColumns: sox produces no image for them.
func TestSoxLiveEmpty(t *testing.T) {
	if !haveSox() {
		t.Skip("sox not on PATH")
	}
	dir := soxTempDir(t)
	empty := []soxRef{
		{name: "e-lg-1", x: 1026, y: 513, span: "15", z: 100, trim: 1},
		{name: "e-lg-174", x: 1026, y: 513, span: "15", z: 100, trim: 174},
		{name: "e-sm-dolph-1", x: 258, y: 129, span: "15", z: 100, dolph: true, trim: 1},
		{name: "e-xl-dolph-1", x: 2050, y: 1025, span: "15", z: 100, dolph: true, trim: 1},
	}
	for _, r := range empty {
		if err := runSoxErr(dir, r.args()); err == nil {
			if _, statErr := os.Stat(filepath.Join(dir, r.name+".png")); statErr == nil {
				t.Errorf("%s: sox wrote an image, want none", r.name)
			}
		}
	}
}

// TestSoxLiveGain checks FitConfig.GainDB against sox -Z: the render's top of
// scale moves by the same amount as the gain.
func TestSoxLiveGain(t *testing.T) {
	if !haveSox() {
		t.Skip("sox not on PATH")
	}
	dir := soxTempDir(t)
	const gain = 10
	r := soxRef{name: "gain", x: 514, y: 257, spanSample: 15 * soxClipRate, z: 100}
	args := []string{"clip.wav", "-n", "spectrogram", "-x", "514", "-y", "257", "-d", "15", "-z", "100", "-Z", "-10", "-w", hannName, "-o", "gain.png", "-r"}
	runSox(t, dir, args)
	img := readSoxPNG(t, filepath.Join(dir, "gain.png"))
	f, err := NewFit(FitConfig{SampleRate: soxClipRate, FrameSize: r.frameSize(), Columns: r.x, Window: stft.WindowShape{Family: stft.ShapeHann}, GainDB: gain})
	if err != nil {
		t.Fatal(err)
	}
	pcm, _ := readWAV(t, filepath.Join(soxDir, "clip.wav"))
	compareToSox(t, f, r, soxFloat(pcm), img).check(t, "gain")
}
