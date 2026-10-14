package gate

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// profileFor measures x's quiet-window profile the way Denoise's pre-scan does.
func profileFor(t *testing.T, sr, frame, hop int, x []float32) (*denoiser.NoiseProfile, error) {
	t.Helper()
	d, err := denoiser.New(denoiser.Config{SampleRate: sr, FrameSize: frame, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	return d.EstimateNoiseProfile(x)
}

// songClip is noise throughout (-40 dBFS white) with a continuous on-bin 3 kHz
// tone (-20 dBFS) from leadSec to the end: song that fills bins 63 to 65 of a
// 1024-point frame for every frame after the lead-in, the case where a per-bin
// rolling median becomes the song itself.
func songClip(sr int, durSec, leadSec float64) []float32 {
	n := int(durSec * float64(sr))
	lead := int(leadSec * float64(sr))
	noise := whiteNoise(n, dbToLin(-40), 71)
	tn := tone(n, sr, 3000, dbToLin(-20))
	for i := lead; i < n; i++ {
		noise[i] += tn[i]
	}
	return noise
}

// TestDenoisePrescanKeepsDenseSong pins that blind Denoise does not take
// continuous song for noise. The song occupies its bins for far more than half of
// the 0.75 s median window, so without the pre-scan cap the per-bin median becomes
// the song level and the gate eats the song (5 to 20 dB on main). Rows with
// FreqSmoothBins=0 isolate the cap from the SNR smoothing.
func TestDenoisePrescanKeepsDenseSong(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	clip := songClip(sr, 4, 1)

	// Precondition: the pre-scan finds its quiet window inside the noise-only
	// lead-in, otherwise the test would not exercise the cap.
	prof, err := profileFor(t, sr, frame, hop, clip)
	if err != nil {
		t.Fatalf("EstimateNoiseProfile: %v", err)
	}
	if end := prof.Info().StartSec + prof.Info().DurationSec; end > 1.0 {
		t.Fatalf("quiet window ends at %.3f s, want inside the 1 s lead-in", end)
	}

	songSpan := [][2]int{{2 * sr, 3*sr + 3*sr/4}}
	leadSpan := [][2]int{{sr / 4, 3 * sr / 4}}
	for _, strength := range []denoiser.Strength{denoiser.Light, denoiser.Medium, denoiser.Heavy} {
		for _, noFreqSmooth := range []bool{false, true} {
			p := ParamsFor(strength)
			name := strength.String() + "/tuned"
			if noFreqSmooth {
				p.FreqSmoothBins = 0
				name = strength.String() + "/nofreqsmooth"
			}
			t.Run(name, func(t *testing.T) {
				out, err := Denoise(clip, Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p})
				if err != nil {
					t.Fatal(err)
				}
				assertFinite(t, "out", out)
				drop := spanRMSDB(clip, songSpan) - spanRMSDB(out, songSpan)
				reduction := spanRMSDB(clip, leadSpan) - spanRMSDB(out, leadSpan)
				t.Logf("song drop %.2f dB, lead-in reduction %.1f dB", drop, reduction)
				if drop > 1.0 {
					t.Errorf("continuous song lost %.2f dB (>1 dB); the blind floor rode up on the song", drop)
				}
				// The lead-in is gated exactly as on the plain blind path (the cap sits
				// above the median there); this only shows the cap did not disable
				// gating. Rows without frequency smoothing reduce a little less.
				if want := float64(p.MaxAttenuationDB) - 6; reduction < want {
					t.Errorf("lead-in noise reduction %.1f dB < %.1f; the cap disabled gating", reduction, want)
				}
			})
		}
	}
}

// TestDenoiseFallsBackToBlind pins both pre-scan fallbacks: a clip with no
// distinctly quiet window (stationary noise plus a steady tone) and a clip
// shorter than one frame. Denoise must then equal the plain streaming blind
// gate bit for bit; erroring, or applying a cap, would diverge.
func TestDenoiseFallsBackToBlind(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}

	stationary := whiteNoise(4*sr, dbToLin(-30), 5)
	tn := tone(len(stationary), sr, 3000, dbToLin(-20))
	for i := range stationary {
		stationary[i] += tn[i]
	}
	cases := []struct {
		name string
		x    []float32
		want error
	}{
		{"no quiet region", stationary, denoiser.ErrNoQuietRegion},
		{"shorter than a frame", whiteNoise(frame-1, dbToLin(-30), 6), denoiser.ErrProfileTooShort},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := profileFor(t, sr, frame, hop, tc.x); !errors.Is(err, tc.want) {
				t.Fatalf("precondition: EstimateNoiseProfile error = %v, want %v", err, tc.want)
			}
			got, err := Denoise(tc.x, cfg)
			if err != nil {
				t.Fatal(err)
			}
			g, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertBitExact(t, "Denoise vs plain blind stream", got, streamAll(t, g, tc.x))
		})
	}
}

// TestDenoiseMatchesManualPrescanSequence pins that Denoise is exactly New +
// setBlindCap(profile) + ProcessInto + FlushInto, with the profile measured at the
// gate's resolved frame geometry (explicit 512/128 here, not the 1024/256 default).
// A pre-scan at the wrong geometry fails setBlindCap's length check or diverges;
// dropping the cap call leaves the plain blind path.
func TestDenoiseMatchesManualPrescanSequence(t *testing.T) {
	const sr, frame, hop = 48000, 512, 128
	clip := songClip(sr, 4, 1)
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}

	auto, err := Denoise(clip, cfg)
	if err != nil {
		t.Fatal(err)
	}
	prof, err := profileFor(t, sr, frame, hop, clip)
	if err != nil {
		t.Fatal(err)
	}
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.setBlindCap(prof.Spectrum()); err != nil {
		t.Fatal(err)
	}
	assertBitExact(t, "Denoise vs manual pre-scan sequence", auto, streamAll(t, g, clip))
	plain, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(auto, streamAll(t, plain, clip)) {
		t.Fatal("precondition: the cap changed nothing, so this test cannot tell Denoise from the plain blind path")
	}
}

// TestBlindCapLevelIsPlus3dBPower pins the cap level: a flat profile spectrum is
// scaled by the power ratio for +3 dB, 10^(3/10) (about 1.9953), not the
// amplitude factor 10^(3/20) and not a different margin.
func TestBlindCapLevelIsPlus3dBPower(t *testing.T) {
	g, err := New(Config{SampleRate: 48000, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	const level = 0.5
	if err := g.setBlindCap(slices.Repeat([]float32{level}, g.bins)); err != nil {
		t.Fatal(err)
	}
	want := float32(level * math.Pow(10, 0.3))
	for k, v := range g.tracker.upper {
		if d := math.Abs(float64(v-want)) / float64(want); d > 1e-6 {
			t.Fatalf("bin %d: cap %g, want profile power x 10^(3/10) = %g", k, v, want)
		}
	}
}

// TestBlindCapTransitions walks the capped blind floor through its state changes.
// Even bins get a cap far below the noise median and odd bins one far above it:
// (a) once warm, the active floor is the tracker's and equals the cap on even bins
// and the median (below the cap) on odd bins; (b) after Flush the warm-up still
// passes audio through (floor at epsPower) with the cap kept; (c) the bound holds
// again once warm; (d) a learned floor wins over the cap and is not changed by
// further input; (e) SetNoiseFloor(nil) drops the cap; (f) a wrong-length cap is
// rejected. A second case caps with +Inf (a spectrum value near MaxFloat32 scaled
// up): the floor falls back to the median there and the output stays finite.
func TestBlindCapTransitions(t *testing.T) {
	const sr, frame, hop = 48000, 1024, 256
	p := ParamsFor(denoiser.Heavy)
	p.FloorWindowSec = 0.1
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p}
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	capIn := make([]float32, g.bins)
	for k := range capIn {
		if k%2 == 0 {
			capIn[k] = 1e-6
		} else {
			capIn[k] = 1e30
		}
	}
	if err := g.setBlindCap(capIn); err != nil {
		t.Fatal(err)
	}
	noise := whiteNoise(sr, dbToLin(-30), 9)

	checkBound := func(stage string) {
		t.Helper()
		if &g.noise[0] != &g.tracker.noise[0] {
			t.Fatalf("%s: active floor is not the tracker's", stage)
		}
		for k, v := range g.noise {
			u := g.tracker.upper[k]
			if k%2 == 0 && v != u {
				t.Fatalf("%s: bin %d floor %g, want the cap %g (the median of -30 dBFS noise is far above it)", stage, k, v, u)
			}
			if k%2 == 1 && !(v < u && v > epsPower) {
				t.Fatalf("%s: bin %d floor %g, want the median, below the cap %g and above epsPower", stage, k, v, u)
			}
		}
	}

	if _, err := g.Process(noise); err != nil {
		t.Fatal(err)
	}
	checkBound("warm")

	if _, err := g.Flush(); err != nil { // Flush resets the stream and the tracker
		t.Fatal(err)
	}
	if _, err := g.Process(noise[:2*hop]); err != nil { // fewer frames than the warm-up
		t.Fatal(err)
	}
	for k, v := range g.noise {
		if v != epsPower {
			t.Fatalf("warm-up after Reset: bin %d floor %g, want epsPower (pass-through)", k, v)
		}
	}
	if g.tracker.upper == nil {
		t.Fatal("Reset dropped the cap")
	}
	if _, err := g.Process(noise); err != nil {
		t.Fatal(err)
	}
	checkBound("warm again")

	learned := slices.Repeat([]float32{1e-3}, g.bins)
	if err := g.SetNoiseFloor(learned); err != nil {
		t.Fatal(err)
	}
	if _, err := g.Process(noise); err != nil {
		t.Fatal(err)
	}
	if &g.noise[0] != &g.noiseBuf[0] || !slices.Equal(g.noiseBuf, learned) {
		t.Fatal("a learned floor did not win over the cap, or further input changed it")
	}

	if err := g.SetNoiseFloor(nil); err != nil {
		t.Fatal(err)
	}
	if g.tracker.upper != nil || &g.noise[0] != &g.tracker.noise[0] {
		t.Fatal("SetNoiseFloor(nil) did not return to the plain blind tracker")
	}

	if err := g.setBlindCap(capIn[:g.bins-1]); !errors.Is(err, ErrFloorMismatch) {
		t.Fatalf("wrong-length cap: error %v, want ErrFloorMismatch", err)
	}

	t.Run("infinite cap", func(t *testing.T) {
		g, err := New(cfg)
		if err != nil {
			t.Fatal(err)
		}
		huge := slices.Repeat([]float32{math.MaxFloat32}, g.bins)
		if err := g.setBlindCap(huge); err != nil {
			t.Fatal(err)
		}
		out := streamAll(t, g, noise)
		assertFinite(t, "out", out)
		if !math.IsInf(float64(g.tracker.upper[0]), 1) {
			t.Fatalf("precondition: cap %g did not overflow to +Inf", g.tracker.upper[0])
		}
	})
}
