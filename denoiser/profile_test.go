package denoiser

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
)

// whiteNoise returns n samples of Gaussian white noise with the given RMS.
func whiteNoise(n int, rms float64, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * rms)
	}
	return x
}

func TestNoiseProfileFromWhiteNoise(t *testing.T) {
	const sr, sigma = 48000, 0.01
	d, _ := New(Config{SampleRate: sr})
	noise := whiteNoise(4*sr, sigma, 5)
	p, err := d.NoiseProfileFromSamples(noise)
	if err != nil {
		t.Fatal(err)
	}
	if p.Info().Source != ProfileSamples {
		t.Errorf("source = %v, want ProfileSamples", p.Info().Source)
	}
	// Expected per-bin power of Hann-windowed white noise: sigma^2 * sum(w^2)
	// = sigma^2 * 3n/8.
	want := sigma * sigma * 3 * 1024 / 8
	spec := p.Spectrum()
	if len(spec) != 513 {
		t.Fatalf("spectrum has %d bins, want 513", len(spec))
	}
	for k, v := range spec {
		if db := 10 * math.Log10(float64(v)/want); math.Abs(db) > 1 {
			t.Errorf("bin %d: %.2f dB from expected", k, db)
		}
	}
	// Spectrum returns a copy.
	spec[0] = -1
	if p.Spectrum()[0] == -1 {
		t.Error("Spectrum aliases internal storage")
	}
}

func TestNoiseProfileErrorsAndSet(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	if _, err := d.NoiseProfileFromSamples(make([]float32, 1023)); !errors.Is(err, ErrProfileTooShort) {
		t.Fatalf("short sample err = %v", err)
	}
	if d.NoiseProfile() != nil {
		t.Fatal("fresh Denoiser must report no profile")
	}
	p, err := d.NoiseProfileFromSamples(whiteNoise(4096, 0.01, 1))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetNoiseProfile(p); err != nil || d.NoiseProfile() != p {
		t.Fatalf("SetNoiseProfile: %v, got %p want %p", err, d.NoiseProfile(), p)
	}
	other, _ := New(Config{SampleRate: 48000, FrameSize: 512})
	if err := other.SetNoiseProfile(p); !errors.Is(err, ErrProfileMismatch) {
		t.Fatalf("mismatch err = %v", err)
	}
	if err := d.SetNoiseProfile(nil); err != nil || d.NoiseProfile() != nil {
		t.Fatalf("clearing the profile: %v", err)
	}
	// Spectrum -> NewNoiseProfile round trip preserves the values and infers
	// the frame size; bad lengths are rejected.
	q, err := NewNoiseProfile(p.Spectrum())
	if err != nil || q.Info().Source != ProfileExternal || q.frameSize != 1024 {
		t.Fatalf("NewNoiseProfile: %v, source %v, frameSize %d", err, q.Info().Source, q.frameSize)
	}
	if err := d.SetNoiseProfile(q); err != nil {
		t.Fatalf("SetNoiseProfile(rebuilt): %v", err)
	}
	for _, bad := range [][]float32{nil, make([]float32, 1), make([]float32, 100), make([]float32, 32)} {
		if _, err := NewNoiseProfile(bad); !errors.Is(err, ErrProfileMismatch) {
			t.Errorf("NewNoiseProfile(len %d) err = %v, want ErrProfileMismatch", len(bad), err)
		}
	}
}

func TestFixedProfileReducesWhiteNoise(t *testing.T) {
	const sr, sigma = 48000, 0.01
	d, _ := New(Config{SampleRate: sr, Strength: Medium})
	p, _ := d.NoiseProfileFromSamples(whiteNoise(sr, sigma, 9))
	if err := d.SetNoiseProfile(p); err != nil {
		t.Fatal(err)
	}
	x := whiteNoise(3*sr, sigma, 10)
	y := runStream(t, d, x, []int{4096})
	in, out := rmsDB(x[sr:]), rmsDB(y[sr:])
	if red := in - out; red < 9 {
		t.Errorf("Medium reduced stationary white noise by %.1f dB, want >= 9", red)
	}
}

// TestLearnNoiseMatchesTwoStep checks the NoiseLearner capability: the one-call
// LearnNoise sets up the same noise model as the manual NoiseProfileFromSamples
// + SetNoiseProfile pair it collapses, so both denoisers, fed the same clip,
// produce identical output. Gutting LearnNoise (dropping the SetNoiseProfile
// call, or swapping the excerpt) leaves the learned denoiser on the adaptive
// path and diverges the output. It also exercises capability discovery through
// a dsp.Processor value.
func TestLearnNoiseMatchesTwoStep(t *testing.T) {
	const sr, sigma = 48000, 0.02
	noise := whiteNoise(sr, sigma, 30)

	ref, _ := New(Config{SampleRate: sr, Strength: Medium})
	p, err := ref.NoiseProfileFromSamples(noise)
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.SetNoiseProfile(p); err != nil {
		t.Fatal(err)
	}

	learned, _ := New(Config{SampleRate: sr, Strength: Medium})
	// Discover the capability the way a consumer does: from a dsp.Processor.
	var proc dsp.Processor = learned
	nl, ok := proc.(NoiseLearner)
	if !ok {
		t.Fatal("*Denoiser must satisfy NoiseLearner")
	}
	if err := nl.LearnNoise(noise); err != nil {
		t.Fatalf("LearnNoise: %v", err)
	}
	if learned.NoiseProfile() == nil {
		t.Fatal("LearnNoise did not activate a profile")
	}

	x := whiteNoise(3*sr, sigma, 31)
	want := runStream(t, ref, x, []int{4096})
	got := runStream(t, learned, x, []int{4096})
	if len(got) != len(want) {
		t.Fatalf("LearnNoise output length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LearnNoise output diverges from two-step at sample %d: %g vs %g", i, got[i], want[i])
		}
	}
}

// TestLearnNoiseRejectsShortExcerpt checks LearnNoise propagates the
// too-short-excerpt error rather than swallowing it, and leaves no profile set.
func TestLearnNoiseRejectsShortExcerpt(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	if err := d.LearnNoise(make([]float32, d.FrameSize()-1)); !errors.Is(err, ErrProfileTooShort) {
		t.Fatalf("LearnNoise(short) err = %v, want ErrProfileTooShort", err)
	}
	if d.NoiseProfile() != nil {
		t.Fatal("a rejected LearnNoise must not activate a profile")
	}
}

// rmsDB returns 20*log10 of the RMS of x (-200 for silence).
func rmsDB(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	if s == 0 || len(x) == 0 {
		return -200
	}
	return 10 * math.Log10(s/float64(len(x)))
}

// toneAt fills x[from:to] with a sine of the given amplitude and frequency.
func toneAt(x []float32, sr, from, to int, freq, amp float64) {
	for i := from; i < to && i < len(x); i++ {
		x[i] += float32(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(sr)))
	}
}

func TestEstimateNoiseProfileFindsPlantedQuietWindow(t *testing.T) {
	const sr = 48000
	d, _ := New(Config{SampleRate: sr})
	// 10 s of -20 dBFS tone everywhere except a 1 s hole [4,5) s holding only
	// -40 dBFS noise; the noise is present throughout.
	x := whiteNoise(10*sr, 0.01, 21)
	toneAt(x, sr, 0, 4*sr, 1000, 0.1*math.Sqrt2)
	toneAt(x, sr, 5*sr, 10*sr, 1000, 0.1*math.Sqrt2)
	p, err := d.EstimateNoiseProfile(x)
	if err != nil {
		t.Fatal(err)
	}
	info := p.Info()
	if info.Source != ProfileAuto {
		t.Errorf("source = %v, want ProfileAuto", info.Source)
	}
	if info.StartSec < 4.0-0.01 || info.StartSec > 4.5+0.01 {
		t.Errorf("window starts at %.3f s, want within [4.0, 4.5]", info.StartSec)
	}
	if math.Abs(info.DurationSec-0.5) > 0.01 {
		t.Errorf("window duration %.3f s, want 0.5", info.DurationSec)
	}
	// The profile is the noise, not the tone: the 1 kHz bin (bin 21 at
	// 46.875 Hz/bin) must not stand out.
	spec := p.Spectrum()
	med := spec[10]
	if spec[21] > 3*med {
		t.Errorf("1 kHz bin %g vs neighbour %g: tone leaked into the profile", spec[21], med)
	}
}

func TestEstimateNoiseProfileAcceptsMostlyNoiseClip(t *testing.T) {
	const sr = 48000
	d, _ := New(Config{SampleRate: sr})
	x := whiteNoise(15*sr, 0.01, 22)
	toneAt(x, sr, 6*sr, 9*sr, 2000, 0.1) // one 3 s burst at -23 dBFS
	p, err := d.EstimateNoiseProfile(x)
	if err != nil {
		t.Fatal(err)
	}
	if s := p.Info().StartSec; s >= 5.5 && s < 9 {
		t.Errorf("window at %.2f s overlaps the burst", s)
	}
}

func TestEstimateNoiseProfileRejectsFlatClips(t *testing.T) {
	const sr = 48000
	d, _ := New(Config{SampleRate: sr})
	tone := make([]float32, 10*sr)
	toneAt(tone, sr, 0, len(tone), 1000, 0.1)
	if _, err := d.EstimateNoiseProfile(tone); !errors.Is(err, ErrNoQuietRegion) {
		t.Errorf("continuous tone: err = %v, want ErrNoQuietRegion", err)
	}
	if _, err := d.EstimateNoiseProfile(whiteNoise(10*sr, 0.01, 23)); !errors.Is(err, ErrNoQuietRegion) {
		t.Errorf("featureless noise: err = %v, want ErrNoQuietRegion", err)
	}
	if _, err := d.EstimateNoiseProfile(make([]float32, 1000)); !errors.Is(err, ErrProfileTooShort) {
		t.Errorf("short input: err = %v, want ErrProfileTooShort", err)
	}
}

func TestEstimateNoiseProfileHandlesNonFinite(t *testing.T) {
	const sr = 48000
	d, _ := New(Config{SampleRate: sr})
	// A NaN sample poisons the running energy sum; the scan must fall back to
	// ErrNoQuietRegion (caller then uses the adaptive tracker) rather than
	// slicing at a negative index and panicking.
	x := whiteNoise(4*sr, 0.01, 71)
	x[2*sr] = float32(math.NaN())
	if _, err := d.EstimateNoiseProfile(x); !errors.Is(err, ErrNoQuietRegion) {
		t.Errorf("NaN clip: err = %v, want ErrNoQuietRegion (no panic)", err)
	}
	y := whiteNoise(4*sr, 0.01, 72)
	y[3*sr] = float32(math.Inf(1))
	if _, err := d.EstimateNoiseProfile(y); !errors.Is(err, ErrNoQuietRegion) {
		t.Errorf("Inf clip: err = %v, want ErrNoQuietRegion (no panic)", err)
	}
}

func TestProfileConstructorsFloorNonFinite(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	// NoiseProfileFromSamples must floor non-finite bins rather than emit a NaN
	// or Inf profile that would poison the stream (Go's max(NaN, x) is NaN).
	z := whiteNoise(4096, 0.01, 73)
	z[100] = float32(math.NaN())
	z[200] = float32(math.Inf(1))
	p, err := d.NoiseProfileFromSamples(z)
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range p.Spectrum() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("NoiseProfileFromSamples bin %d = %g, want finite", k, v)
		}
	}
	// NewNoiseProfile floors NaN, +Inf and non-positive spectrum values.
	spec := make([]float32, 513)
	for i := range spec {
		spec[i] = 1e-6
	}
	spec[5] = float32(math.Inf(1))
	spec[6] = float32(math.NaN())
	spec[7] = -1
	q, err := NewNoiseProfile(spec)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []int{5, 6, 7} {
		if v := q.Spectrum()[k]; v != epsPower {
			t.Errorf("NewNoiseProfile bin %d = %g, want epsPower", k, v)
		}
	}
}

func TestEstimateNoiseProfileSkipsDigitalSilence(t *testing.T) {
	const sr = 48000
	d, _ := New(Config{SampleRate: sr})
	x := make([]float32, 8*sr) // 1 s of zeros, then noise with a loud middle
	copy(x[sr:], whiteNoise(7*sr, 0.01, 24))
	toneAt(x, sr, 3*sr, 5*sr, 3000, 0.2)
	p, err := d.EstimateNoiseProfile(x)
	if err != nil {
		t.Fatal(err)
	}
	if p.Info().StartSec < 1.0-0.006 {
		t.Errorf("window at %.3f s lies in the digital silence", p.Info().StartSec)
	}
	// All-silent input has no usable window: its measured stretch is silence.
	if _, err := d.EstimateNoiseProfile(make([]float32, 4*sr)); !errors.Is(err, ErrNoQuietRegion) {
		t.Fatalf("all-silent clip: err = %v, want ErrNoQuietRegion", err)
	}
}

// mostlySilentClip is 8 s of -30 dBFS white noise with a 1 s run of digital
// silence at 5 s and a 20 ms zero gap every 0.4 s, so every 0.5 s window holds
// silence and the gaps and the long run leave nothing that is silence-free.
func mostlySilentClip(sr int) []float32 {
	x := whiteNoise(8*sr, 0.0316, 61)
	clear(x[5*sr : 6*sr])
	for at := 0; at+sr/50 <= len(x); at += 2 * sr / 5 {
		clear(x[at : at+sr/50])
	}
	return x
}

// wideFrameClip is 12 loud blocks (-30 dBFS), quiet quiet blocks (-50 dBFS),
// one zero block and 20 loud blocks, for a frame of 4 blocks that is longer
// than the 0.5 s window: the window is silence-free but widening it to one
// frame reaches the zero block.
func wideFrameClip(hop, quiet int) []float32 {
	loud := whiteNoise(32*hop, 0.0316, 21)
	q := whiteNoise(quiet*hop, 0.00316, 22)
	x := make([]float32, 0, (33+quiet)*hop)
	x = append(x, loud[:12*hop]...)
	x = append(x, q...)
	x = append(x, make([]float32, hop)...)
	return append(x, loud[12*hop:]...)
}

// silentRangeCase is a clip whose quiet window, as measured, holds a block of
// digital silence.
type silentRangeCase struct {
	name string
	cfg  Config
	x    []float32
}

func silentRangeCases(t *testing.T) []silentRangeCase {
	t.Helper()
	cs := make([]silentRangeCase, 0, 11)
	for _, sr := range []int{16000, 48000, 192000} {
		cs = append(cs, silentRangeCase{fmt.Sprintf("mostly-silent/%d", sr), Config{SampleRate: sr}, mostlySilentClip(sr)})
	}
	for _, sr := range []int{16000, 48000, 192000} {
		cfg := Config{SampleRate: sr}
		hop := mustDenoiser(t, cfg).hop
		x := whiteNoise(4*sr, 0.0316, 62)
		noiseSpan(x, 5*sr/2, 7*sr/2, 0.00562, 63)
		for k := 0; int(float64(k)*0.3*float64(sr)) < len(x)-hop; k++ {
			at := int(float64(k)*0.3*float64(sr)) / hop * hop
			clear(x[at : at+hop])
		}
		cs = append(cs, silentRangeCase{fmt.Sprintf("one-silent-block-per-window/%d", sr), cfg, x})
	}
	short := whiteNoise(48000*2/5, 0.0316, 64)
	clear(short[30*256 : 31*256])
	cs = append(cs, silentRangeCase{"short-clip-silent-block", Config{SampleRate: 48000}, short})
	for _, w := range []struct{ sr, frame, hop, quiet int }{
		{16000, 16384, 4096, 2}, {48000, 32768, 8192, 3}, {192000, 131072, 32768, 3},
	} {
		cs = append(cs, silentRangeCase{
			fmt.Sprintf("wide-frame-mid/%d", w.sr),
			Config{SampleRate: w.sr, FrameSize: w.frame, HopSize: w.hop},
			wideFrameClip(w.hop, w.quiet),
		})
	}
	const hop = 8192
	end := slices.Concat(whiteNoise(32*hop, 0.0316, 65), make([]float32, hop), whiteNoise(3*hop+4096, 0.00316, 66))
	cs = append(cs, silentRangeCase{"wide-frame-clip-end-off-grid", Config{SampleRate: 48000, FrameSize: 32768, HopSize: hop}, end})
	return cs
}

func TestEstimateNoiseProfileRejectsSilentMeasuredRange(t *testing.T) {
	for _, c := range silentRangeCases(t) {
		t.Run(c.name, func(t *testing.T) {
			d := mustDenoiser(t, c.cfg)
			if _, err := d.EstimateNoiseProfile(c.x); !errors.Is(err, ErrNoQuietRegion) {
				t.Fatalf("EstimateNoiseProfile: err = %v, want ErrNoQuietRegion", err)
			}
			for name, next := range map[string]func() int{
				"chunk-hop-1": fixed(d.hop - 1),
				"chunk-4801":  fixed(4801),
				"whole-clip":  fixed(len(c.x)),
			} {
				e := d.NewNoiseProfileEstimator()
				feedFloat(e, c.x, next)
				if _, err := e.Profile(); !errors.Is(err, ErrNoQuietRegion) {
					t.Errorf("%s: Profile err = %v, want ErrNoQuietRegion", name, err)
				}
			}
		})
	}
}

// TestEstimateNoiseProfileRangeOnHopGrid pins that a window widened to one
// frame at the end of the clip is cut at the last whole hop block rather than
// at the last sample: the 5000 samples after the last block are never measured.
func TestEstimateNoiseProfileRangeOnHopGrid(t *testing.T) {
	c := findCase(t, "widen-48k-end-left")
	d := mustDenoiser(t, c.cfg)
	const wantStart, wantLen = 16 * 16384, 65536
	check := func(t *testing.T, p *NoiseProfile, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		start := int(math.Round(p.Info().StartSec * 48000))
		n := int(math.Round(p.Info().DurationSec * 48000))
		if start != wantStart || n != wantLen {
			t.Fatalf("measured [%d, +%d), want [%d, +%d)", start, n, wantStart, wantLen)
		}
	}
	t.Run("whole-clip", func(t *testing.T) {
		p, err := d.EstimateNoiseProfile(c.x)
		check(t, p, err)
	})
	t.Run("streaming", func(t *testing.T) {
		e := d.NewNoiseProfileEstimator()
		feedFloat(e, c.x, fixed(4801))
		p, err := e.Profile()
		check(t, p, err)
	})
}
