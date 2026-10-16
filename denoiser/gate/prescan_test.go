package gate

import (
	"errors"
	"fmt"
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
	// Rows with frame 0 use the auto geometry for the rate (frame 256 at 16 kHz,
	// 4096 at 192 kHz), which keeps the 3 kHz tone on bin.
	for _, row := range []struct{ sr, frame, hop int }{
		{48000, 1024, 256},
		{16000, 0, 0},
		{192000, 0, 0},
	} {
		t.Run(fmt.Sprintf("%d", row.sr), func(t *testing.T) {
			testPrescanKeepsDenseSong(t, row.sr, row.frame, row.hop)
		})
	}
}

func testPrescanKeepsDenseSong(t *testing.T, sr, frame, hop int) {
	t.Helper()
	clip := songClip(sr, 4, 1)

	// Precondition: the pre-scan finds its quiet window inside the noise-only
	// lead-in, otherwise the test would not exercise the cap.
	scanFrame, scanHop := frame, hop
	if frame == 0 {
		scanFrame = autoGeometry(t, sr)
		scanHop = scanFrame / 4
	}
	prof, err := profileFor(t, sr, scanFrame, scanHop, clip)
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
// gate's resolved sample rate and frame geometry. The rows use HopSize frame/8
// (not the frame/4 default) at three rates, so a pre-scan that dropped HopSize or
// assumed 48 kHz measures a different profile and diverges, as does one at the
// wrong frame (setBlindCap's length check); dropping the cap call leaves the
// plain blind path.
func TestDenoiseMatchesManualPrescanSequence(t *testing.T) {
	for _, row := range []struct{ sr, frame, hop int }{
		{48000, 512, 64},
		{16000, 256, 32},
		{192000, 4096, 512},
	} {
		t.Run(fmt.Sprintf("%d", row.sr), func(t *testing.T) {
			clip := songClip(row.sr, 4, 1)
			cfg := Config{SampleRate: row.sr, FrameSize: row.frame, HopSize: row.hop, Strength: denoiser.Medium}

			auto, err := Denoise(clip, cfg)
			if err != nil {
				t.Fatal(err)
			}
			prof, err := profileFor(t, row.sr, row.frame, row.hop, clip)
			if err != nil {
				t.Fatal(err)
			}
			plain, capped := plainAndCapped(t, cfg, clip, prof)
			assertBitExact(t, "Denoise vs manual pre-scan sequence", auto, capped)
			if slices.Equal(auto, plain) {
				t.Fatal("precondition: the cap changed nothing, so this test cannot tell Denoise from the plain blind path")
			}
		})
	}
}

// TestDenoiseShortClipTakesQuietestWindow pins the pre-scan on clips too short
// to be judged for dynamic range. The 0.9 s rows have fewer than two windows of
// non-silent audio, so the acceptance test is skipped; the 0.4 s rows are
// shorter than one window, so the whole clip is the window. Both take the
// quietest window as is and cap with it. The clip is -40 dBFS noise with a
// -20 dBFS 3 kHz tone from toneSec on, and FloorWindowSec is shortened so the
// blind median engages within the clip.
func TestDenoiseShortClipTakesQuietestWindow(t *testing.T) {
	for _, sr := range []int{16000, 48000, 192000} {
		for _, row := range []struct{ durSec, toneSec, floorSec float64 }{
			{0.9, 0.55, 0.2},
			{0.4, 0.24, 0.1},
		} {
			t.Run(fmt.Sprintf("%d/%.1fs", sr, row.durSec), func(t *testing.T) {
				x := songClip(sr, row.durSec, row.toneSec)
				frame := autoGeometry(t, sr)
				p := ParamsFor(denoiser.Medium)
				p.FloorWindowSec = float32(row.floorSec)
				cfg := Config{SampleRate: sr, Params: &p}

				prof, err := profileFor(t, sr, frame, frame/4, x)
				if err != nil {
					t.Fatalf("precondition: EstimateNoiseProfile: %v", err)
				}
				plain, capped := plainAndCapped(t, cfg, x, prof)
				if slices.Equal(plain, capped) {
					t.Fatal("precondition: the cap changed nothing, so this test cannot tell a cap from the plain blind path")
				}
				got, err := Denoise(x, cfg)
				if err != nil {
					t.Fatal(err)
				}
				assertBitExact(t, "Denoise vs manual capped stream", got, capped)
			})
		}
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

// autoGeometry returns the frame size Config{SampleRate: sr} resolves to.
func autoGeometry(t *testing.T, sr int) int {
	t.Helper()
	g, err := New(Config{SampleRate: sr})
	if err != nil {
		t.Fatal(err)
	}
	return g.FrameSize()
}

// plainAndCapped returns the plain blind stream of x at cfg and the stream
// after capping the blind floor with the spectrum of p.
func plainAndCapped(t *testing.T, cfg Config, x []float32, p *denoiser.NoiseProfile) (plain, capped []float32) {
	t.Helper()
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	plain = streamAll(t, g, x)
	g, err = New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.setBlindCap(p.Spectrum()); err != nil {
		t.Fatal(err)
	}
	return plain, streamAll(t, g, x)
}

// TestDenoiseSkipsCapOnSilentWindow pins that Denoise keeps the plain blind
// median when no quiet window of the clip is free of digital silence.
//
// The late-silence rows are nine 0.4 s noise bursts separated by 0.05 s of
// zeros, then 1 s of zeros: no 0.5 s run is silence-free, and the all-zero tail
// is the minimum-energy window, whose epsPower profile would stop all gating (a
// cap at the floor leaves the bursts that the plain median reduces by several
// dB). The partly-silent rows are noise with one zero block of exactly HopSize
// samples every 0.3 s and a quieter second, at HopSize frame/8: every 0.5 s
// window holds a silent block. At HopSize frame/4 the same dropouts are half a
// block, no block is silent, and the estimator returns a usable profile, so a
// pre-scan that dropped HopSize would cap where Denoise must not.
func TestDenoiseSkipsCapOnSilentWindow(t *testing.T) {
	for _, sr := range []int{16000, 48000, 192000} {
		frame := autoGeometry(t, sr)
		t.Run(fmt.Sprintf("late-silence/%d", sr), func(t *testing.T) {
			burst := whiteNoise(sr*4/10, dbToLin(-30), 3)
			gap := sr * 5 / 100
			x := make([]float32, 0, 9*(len(burst)+gap)+sr)
			for range 9 {
				x = append(x, burst...)
				x = append(x, make([]float32, gap)...)
			}
			x = append(x, make([]float32, sr)...)
			cfg := Config{SampleRate: sr, Strength: denoiser.Medium}
			if _, err := profileFor(t, sr, frame, frame/4, x); !errors.Is(err, denoiser.ErrNoQuietRegion) {
				t.Fatalf("precondition: EstimateNoiseProfile error = %v, want ErrNoQuietRegion", err)
			}
			g, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			floor, err := denoiser.NewNoiseProfile(slices.Repeat([]float32{epsPower}, g.bins))
			if err != nil {
				t.Fatal(err)
			}
			plain, capped := plainAndCapped(t, cfg, x, floor)
			if slices.Equal(plain, capped) {
				t.Fatal("precondition: an epsPower cap changed nothing, so this test cannot tell a skipped cap from an applied one")
			}
			got, err := Denoise(x, cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertBitExact(t, "Denoise vs plain blind stream", got, plain)
		})
		t.Run(fmt.Sprintf("partly-silent/%d", sr), func(t *testing.T) {
			hop := frame / 8
			x := whiteNoise(4*sr, dbToLin(-30), 4)
			copy(x[5*sr/2:7*sr/2], whiteNoise(sr, dbToLin(-45), 5))
			for k := 0; ; k++ {
				at := int(float64(k)*0.3*float64(sr)/float64(hop)) * hop
				if at+hop > len(x) {
					break
				}
				clear(x[at : at+hop])
			}
			cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}
			if _, err := profileFor(t, sr, frame, hop, x); !errors.Is(err, denoiser.ErrNoQuietRegion) {
				t.Fatalf("precondition: EstimateNoiseProfile at hop %d error = %v, want ErrNoQuietRegion", hop, err)
			}
			coarse, err := profileFor(t, sr, frame, frame/4, x)
			if err != nil {
				t.Fatalf("precondition: EstimateNoiseProfile at hop %d: %v", frame/4, err)
			}
			plain, capped := plainAndCapped(t, cfg, x, coarse)
			if slices.Equal(plain, capped) {
				t.Fatal("precondition: the hop frame/4 cap changed nothing, so this test cannot tell a skipped cap from an applied one")
			}
			got, err := Denoise(x, cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertBitExact(t, "Denoise vs plain blind stream", got, plain)
		})
	}
}

// wideFrameClip is 12 blocks of -30 dBFS noise, quiet blocks at -50 dBFS, one
// block of digital silence and 20 more loud blocks, for a FrameSize of four
// blocks. The 0.5 s window of the quiet blocks is silence-free but shorter
// than a frame, so the estimator measures a whole frame, which reaches the
// silent block.
func wideFrameClip(hop, quiet int) []float32 {
	loud := whiteNoise(32*hop, dbToLin(-30), 21)
	x := make([]float32, 0, (33+quiet)*hop)
	x = append(x, loud[:12*hop]...)
	x = append(x, whiteNoise(quiet*hop, dbToLin(-50), 22)...)
	x = append(x, make([]float32, hop)...)
	return append(x, loud[12*hop:]...)
}

// TestDenoiseSkipsCapWhenWideFrameReachesSilence pins the FrameSize case of the
// silence rule. When FrameSize is longer than the 0.5 s window (for example
// 32768 at 48 kHz, where the window is three hop blocks of 8192) the estimator
// measures a whole frame from the window start. The mid-clip rows put the
// quietest blocks right before one block of digital silence: the window is
// silence-free, but the measured frame reaches the silent block, so Denoise
// must keep the plain blind median. The clip-end row has one silent block, then
// three and a half quiet blocks: the quiet window is the last three whole
// blocks and the frame measured back from the last whole block reaches the
// silent block, where a frame measured back from the last sample would start
// off the hop grid. A silence check limited to the 0.5 s window would cap.
func TestDenoiseSkipsCapWhenWideFrameReachesSilence(t *testing.T) {
	const hop48 = 8192
	tests := []struct {
		name            string
		sr, frame, hop  int
		x               []float32
		wouldBeMeasured func(x []float32, hop, frame int) []float32
	}{
		{"48000", 48000, 4 * hop48, hop48, wideFrameClip(hop48, 3), midClip},
		{"16000", 16000, 16384, 4096, wideFrameClip(4096, 2), midClip},
		{"192000", 192000, 131072, 32768, wideFrameClip(32768, 3), midClip},
		{
			"clip-end-off-grid", 48000, 4 * hop48, hop48,
			slices.Concat(whiteNoise(32*hop48, dbToLin(-30), 21), make([]float32, hop48), whiteNoise(3*hop48+4096, dbToLin(-50), 22)),
			func(x []float32, _, frame int) []float32 { return x[len(x)-frame:] },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{SampleRate: tc.sr, FrameSize: tc.frame, HopSize: tc.hop, Strength: denoiser.Medium}
			// The profile an estimator that accepted the clip would have measured.
			d, err := denoiser.New(denoiser.Config{SampleRate: tc.sr, FrameSize: tc.frame, HopSize: tc.hop})
			if err != nil {
				t.Fatal(err)
			}
			would, err := d.NoiseProfileFromSamples(tc.wouldBeMeasured(tc.x, tc.hop, tc.frame))
			if err != nil {
				t.Fatal(err)
			}
			plain, capped := plainAndCapped(t, cfg, tc.x, would)
			if slices.Equal(plain, capped) {
				t.Fatal("precondition: the cap changed nothing, so this test cannot tell a skipped cap from an applied one")
			}
			if _, err := profileFor(t, tc.sr, tc.frame, tc.hop, tc.x); !errors.Is(err, denoiser.ErrNoQuietRegion) {
				t.Fatalf("EstimateNoiseProfile error = %v, want ErrNoQuietRegion", err)
			}
			got, err := Denoise(tc.x, cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertBitExact(t, "Denoise vs plain blind stream", got, plain)
		})
	}
}

// midClip is the frame starting at block 12 of a wideFrameClip, the quiet
// window widened to one frame.
func midClip(x []float32, hop, frame int) []float32 { return x[12*hop : 12*hop+frame] }
