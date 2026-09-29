package gate

import (
	"fmt"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/internal/audiotest"
)

// The A/B tests measure the gate against ffmpeg's afftdn, the same baseline the
// flagship denoiser is held to (denoiser/ffmpeg_test.go, corpus_test.go). They
// skip when ffmpeg is absent; the corpus and noise-bed tests also need local
// recordings, so they never run in CI.

// abBars are regression guards, not quality claims. Measured on the synthetic
// clips and a recorded noise bed the gate lands within about 2 dB of afftdn at
// light and medium and well ahead at heavy; 3 dB of slack absorbs unpinned host
// ffmpeg drift while still catching a real regression.
var abBars = audiotest.Bars{ReductionDB: 3, LSDDB: 3}

var abStrengths = []denoiser.Strength{denoiser.Light, denoiser.Medium, denoiser.Heavy}

// abLevels binds each strength to the afftdn preset it is measured against.
func abLevels() []audiotest.Level {
	levels := make([]audiotest.Level, 0, len(abStrengths))
	for _, s := range abStrengths {
		nrnf := audiotest.AfftdnPreset[s.String()]
		levels = append(levels, audiotest.Level{
			Name: s.String(), NR: nrnf[0], NF: nrnf[1],
			Denoise: func(in []float32) ([]float32, error) {
				return Denoise(in, Config{SampleRate: 48000, Strength: s})
			},
		})
	}
	return levels
}

// TestGateAgainstAfftdnSynthetic measures the gate on the shared synthetic clips
// (white and pink noise, known bursts) both blind and with an oracle noise
// profile, and logs the reduction and signal LSD next to afftdn's. The hard
// invariants are that the floor never rises and the signal is not destroyed;
// the afftdn comparison is held to the looser abBars guard.
func TestGateAgainstAfftdnSynthetic(t *testing.T) {
	audiotest.FFmpegPath(t)
	for _, pink := range []bool{false, true} {
		clip := audiotest.MakeSynthClip(48000, -40, -20, pink, 11)
		for _, strength := range abStrengths {
			for _, mode := range []string{"blind", "learned"} {
				t.Run(fmt.Sprintf("%v/%s/pink=%v", strength, mode, pink), func(t *testing.T) {
					cfg := Config{SampleRate: clip.SR, Strength: strength}
					var ours []float32
					var err error
					if mode == "blind" {
						ours, err = Denoise(clip.Mix, cfg)
					} else {
						ours, err = DenoiseWithNoise(clip.Mix, clip.Noise, cfg)
					}
					if err != nil {
						t.Fatal(err)
					}
					audiotest.AssertFinite(t, "gate output", ours)
					nrnf := audiotest.AfftdnPreset[strength.String()]
					ref := audiotest.RunAfftdn(t, clip.Mix, clip.SR, nrnf[0], nrnf[1])
					lag := audiotest.BestLag(clip.Mix, ref, clip.SignalSpans[1][0], clip.SignalSpans[1][1], 4096)
					ref = audiotest.Shifted(ref, lag)

					inNoise := audiotest.SpanRMSDB(clip.Mix, clip.NoiseSpans)
					redOurs := inNoise - audiotest.SpanRMSDB(ours, clip.NoiseSpans)
					redRef := inNoise - audiotest.SpanRMSDB(ref, clip.NoiseSpans)
					lsdOurs := audiotest.LSDDB(clip.Clean, ours, clip.SignalSpans, 1024, 256)
					lsdRef := audiotest.LSDDB(clip.Clean, ref, clip.SignalSpans, 1024, 256)
					lsdIn := audiotest.LSDDB(clip.Clean, clip.Mix, clip.SignalSpans, 1024, 256)
					t.Logf("reduction ours %.1f / afftdn %.1f dB; signal LSD ours %.2f / afftdn %.2f / input %.2f dB (afftdn lag %d)",
						redOurs, redRef, lsdOurs, lsdRef, lsdIn, lag)

					if redOurs < -0.5 {
						t.Errorf("noise floor rose by %.1f dB", -redOurs)
					}
					if redOurs < redRef-abBars.ReductionDB {
						t.Errorf("noise reduction %.1f dB is %.1f below afftdn's %.1f (%.1f dB bar)", redOurs, redRef-redOurs, redRef, abBars.ReductionDB)
					}
					if lsdOurs > lsdRef+abBars.LSDDB {
						t.Errorf("signal LSD %.2f dB exceeds afftdn's %.2f by %.2f (%.1f dB bar)", lsdOurs, lsdRef, lsdOurs-lsdRef, abBars.LSDDB)
					}
					if lsdOurs > lsdIn+1 {
						t.Errorf("signal LSD %.2f dB is worse than the untouched input's %.2f; the gate is damaging the signal", lsdOurs, lsdIn)
					}
				})
			}
		}
	}
}

// TestCorpusAgainstAfftdn is the real-recording A/B behind the Taskfile `ab`
// target: it shares audiotest.RunCorpusAB with the flagship denoiser. The corpus
// is the same gitignored denoiser/testdata/corpus/*.wav.
func TestCorpusAgainstAfftdn(t *testing.T) {
	audiotest.RunCorpusAB(t, audiotest.CorpusDir("../testdata/corpus"), abLevels(), 0)
}

// TestNoiseBedAgainstAfftdn mixes synthetic bursts of known position and level
// into a recorded noise bed (denoiser/testdata/noisebed/*.wav, gitignored), so
// realistic noise can be scored against a clean reference.
func TestNoiseBedAgainstAfftdn(t *testing.T) {
	audiotest.RunNoiseBedAB(t, audiotest.NoiseBedDir("../testdata/noisebed"), 12, abLevels(),
		abBars)
}
