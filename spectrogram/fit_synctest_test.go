package spectrogram

import (
	"testing"
	"testing/synctest"

	"github.com/tphakala/go-audio-dsp/stft"
)

// A Fit built inside a testing/synctest bubble shares its edge windows through
// the process-wide stft cache, so building the same geometry again outside the
// bubble, or in another one, must not touch anything bound to the first bubble.
// stft pins the cache itself; this is the consumer-level regression test.
func TestNewFitAcrossSynctestBubbles(t *testing.T) {
	cfg := FitConfig{SampleRate: 24000, FrameSize: 512, Columns: 514,
		Window: stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: 100.5}}
	newFit := func(t *testing.T) {
		t.Helper()
		if _, err := NewFit(cfg); err != nil {
			t.Fatal(err)
		}
	}
	synctest.Test(t, newFit)
	newFit(t)
	synctest.Test(t, newFit)
}
