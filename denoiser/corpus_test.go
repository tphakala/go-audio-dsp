package denoiser

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/internal/audiotest"
)

// TestCorpusAgainstAfftdn is the real-corpus A/B comparison behind the Taskfile
// `ab` target. The comparison itself lives in audiotest.RunCorpusAB, shared with
// denoiser/gate; this only binds the flagship denoiser's three strengths to the
// afftdn settings they are measured against.
//
// The corpus (denoiser/testdata/corpus/*.wav) is user-supplied and gitignored, so
// this test never runs in CI: it skips when the corpus or ffmpeg is absent.
func TestCorpusAgainstAfftdn(t *testing.T) {
	levels := make([]audiotest.Level, 0, 3)
	for _, s := range []Strength{Light, Medium, Heavy} {
		nrnf := audiotest.AfftdnPreset[s.String()]
		levels = append(levels, audiotest.Level{
			Name: s.String(), NR: nrnf[0], NF: nrnf[1],
			Denoise: func(in []float32) ([]float32, error) {
				return Denoise(in, Config{SampleRate: 48000, Strength: s})
			},
		})
	}
	audiotest.RunCorpusAB(t, audiotest.CorpusDir("testdata/corpus"), levels, 0)
}
