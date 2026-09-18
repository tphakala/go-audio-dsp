package gate

import (
	"errors"
	"testing"

	"github.com/tphakala/go-audio-dsp/denoiser"
)

// TestConfigResolveErrors pins the Config validation: a bad sample rate, a
// non-power-of-two or too-small frame, and a hop that is not a smaller divisor of
// the frame all wrap ErrInvalidConfig.
func TestConfigResolveErrors(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"zero sample rate", Config{SampleRate: 0}},
		{"non-power-of-two frame", Config{SampleRate: 48000, FrameSize: 1000}},
		{"frame too small", Config{SampleRate: 48000, FrameSize: 32}},
		{"hop not a divisor", Config{SampleRate: 48000, FrameSize: 1024, HopSize: 300}},
		{"hop equals frame", Config{SampleRate: 48000, FrameSize: 1024, HopSize: 1024}},
	}
	for _, tc := range cases {
		if _, err := New(tc.cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("%s: New err %v, want ErrInvalidConfig", tc.name, err)
		}
	}
}

// TestNewRejectsInvalidCustomParams pins that a Config.Params override is run
// through validate() by resolve(): an invalid custom Params fails New with
// ErrInvalidConfig (covering the resolve -> validate wiring on the override path,
// which the direct validate() tests do not exercise).
func TestNewRejectsInvalidCustomParams(t *testing.T) {
	bad := ParamsFor(denoiser.Medium)
	bad.TransitionDB = 0 // invalid: must be finite and > 0
	if _, err := New(Config{SampleRate: 48000, Params: &bad}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("New with invalid custom Params: err %v, want ErrInvalidConfig", err)
	}
}

// TestDefaultHopAndStrength pins the resolved defaults: hop is FrameSize/4 and the
// zero-value Strength is Medium's floor.
func TestDefaultHopAndStrength(t *testing.T) {
	g, err := New(Config{SampleRate: 48000, FrameSize: 1024})
	if err != nil {
		t.Fatal(err)
	}
	if g.HopSize() != 256 {
		t.Errorf("default HopSize = %d, want 256 (FrameSize/4)", g.HopSize())
	}
	if g.params != ParamsFor(denoiser.Medium) {
		t.Errorf("zero-value Strength did not resolve to Medium params")
	}
}
