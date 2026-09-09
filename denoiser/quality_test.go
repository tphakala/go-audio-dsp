package denoiser

import (
	"fmt"
	"testing"
)

// minReductionMarginDB is how far below MaxAttenuationDB the measured
// noise-region reduction may fall on the synthetic harness. MaxAttenuationDB is
// a gain floor, not a target: even in noise-only regions the MMSE-LSA gain sits
// a few dB above the floor, and pink noise (more low-frequency energy) reduces a
// little shallower than white. 4 dB tracks that physical gap while still failing
// any real under-reduction regression. Do not loosen it to paper over a tuning
// regression. A tighter within-1-dB-of-afftdn bar is the eventual goal, but the
// afftdn oracle (ffmpeg_test.go) currently records and skips on a miss rather
// than enforcing, pending a pinned ffmpeg and a real bird-clip corpus, so this
// synthetic bar is the live reduction guard.
const minReductionMarginDB = 4

func TestPresetsOnSyntheticClips(t *testing.T) {
	// A few fixed seeds so the bars are not validated on a single noise
	// realization (guards against a cherry-picked seed).
	for _, seed := range []uint64{7, 12, 20} {
		for _, pink := range []bool{false, true} {
			clip := makeSynthClip(48000, -40, -20, pink, seed)
			// reduction on the same clip per preset, for the monotonicity check.
			reductionByPreset := make(map[Preset]float64, 3)
			for _, preset := range []Preset{Light, Medium, Heavy} {
				t.Run(fmt.Sprintf("%v/pink=%v/seed=%d", preset, pink, seed), func(t *testing.T) {
					p := preset.Params()
					out, err := Denoise(clip.mix, Config{SampleRate: clip.sr, Preset: preset})
					if err != nil {
						t.Fatal(err)
					}
					if len(out) != len(clip.mix) {
						t.Fatalf("%d samples, want %d", len(out), len(clip.mix))
					}
					outNoise := spanRMSDB(out, clip.noiseSpans)
					red := spanRMSDB(clip.mix, clip.noiseSpans) - outNoise
					reductionByPreset[preset] = red
					want := float64(p.MaxAttenuationDB) - minReductionMarginDB
					before := segSNRDB(clip.clean, clip.mix, clip.signalSpans, 960) // 20 ms segments at 48 kHz
					after := segSNRDB(clip.clean, out, clip.signalSpans, 960)
					// Align on the second burst (a 2-6 kHz chirp): its cross
					// correlation peaks sharply and unambiguously. The first burst is
					// a pure 4 kHz tone (period 12 samples at 48 kHz), whose periodic
					// correlation would invite cycle slipping.
					lag := bestLag(clip.mix, out, clip.signalSpans[1][0], clip.signalSpans[1][1], 64)
					t.Logf("reduction %.1f dB (want >= %.1f, floor %g) segSNR %.1f -> %.1f dB lag %d", red, want, p.MaxAttenuationDB, before, after, lag)

					// Signal preservation and alignment hold for every preset.
					if after < before {
						t.Errorf("segmental SNR degraded %.1f -> %.1f dB", before, after)
					}
					if lag != 0 {
						t.Errorf("output misaligned by %d samples", lag)
					}
					// The reduction bar is meaningless if the noise region were
					// zeroed to the silence sentinel (a bug would then pass with a
					// huge "reduction"). Real output stays well above it (noise
					// level minus at most MaxAttenuationDB).
					if outNoise <= -190 {
						t.Errorf("noise region collapsed to %.0f dBFS (silent/sentinel); reduction figure is vacuous", outNoise)
					}
					// Noise reduction bar: every preset clears its floor.
					if red < want {
						t.Errorf("noise reduced by %.1f dB, want >= %.1f", red, want)
					}
				})
			}
			// Presets must order by strength on the same clip: Heavy reduces more
			// than Medium, which reduces more than Light. The measured gaps are
			// large (~5-6 dB), so this does not flake on any noise realization; it
			// catches a preset table wired out of order or a knob change that
			// inverts the intended ordering. The per-preset subtests above run
			// before this one (subtests are sequential), so the map is populated.
			t.Run(fmt.Sprintf("monotonic/pink=%v/seed=%d", pink, seed), func(t *testing.T) {
				for _, p := range []Preset{Light, Medium, Heavy} {
					if _, ok := reductionByPreset[p]; !ok {
						t.Fatalf("no reduction recorded for %v (a preset subtest failed before recording it)", p)
					}
				}
				l, m, h := reductionByPreset[Light], reductionByPreset[Medium], reductionByPreset[Heavy]
				if !(l < m && m < h) {
					t.Errorf("reduction not monotonic in preset strength: Light %.1f, Medium %.1f, Heavy %.1f dB", l, m, h)
				}
			})
		}
	}
}

// TestFreqSmoothingParamEngages exercises the frequency-smoothing gain path
// (gainState.compute -> smoothGain), which no preset enables by default; it is
// reachable only through a custom Params.FreqSmoothBins > 1. This guards the
// integration wiring end to end, complementing the direct smoothGain unit test.
func TestFreqSmoothingParamEngages(t *testing.T) {
	clip := makeSynthClip(48000, -40, -20, false, 7)
	off := Medium.Params()
	off.FreqSmoothBins = 0
	on := off
	on.FreqSmoothBins = 5
	outOff, err := Denoise(clip.mix, Config{SampleRate: clip.sr, Params: &off})
	if err != nil {
		t.Fatal(err)
	}
	outOn, err := Denoise(clip.mix, Config{SampleRate: clip.sr, Params: &on})
	if err != nil {
		t.Fatal(err)
	}
	if len(outOn) != len(clip.mix) {
		t.Fatalf("%d samples, want %d", len(outOn), len(clip.mix))
	}
	// Frequency smoothing must actually change the output; identical output would
	// mean the compute -> smoothGain wiring is dead.
	var diff float64
	for i := range outOn {
		d := float64(outOn[i] - outOff[i])
		diff += d * d
	}
	if diff == 0 {
		t.Error("FreqSmoothBins > 1 gave output identical to FreqSmoothBins = 0; smoothing path not engaged")
	}
}
