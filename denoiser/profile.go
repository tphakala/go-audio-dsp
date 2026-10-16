package denoiser

import (
	"fmt"
	"math"
	"slices"

	"github.com/tphakala/go-audio-dsp/internal/dspshared"
	"github.com/tphakala/simd/f32"
)

// ProfileSource tells how a NoiseProfile was obtained.
type ProfileSource int

const (
	// ProfileSamples: measured from caller-supplied noise-only samples.
	ProfileSamples ProfileSource = iota
	// ProfileAuto: measured from the quietest window EstimateNoiseProfile or
	// NoiseProfileEstimator.Profile found.
	ProfileAuto
	// ProfileExternal: rebuilt from a saved spectrum with NewNoiseProfile.
	ProfileExternal
)

// String returns the source name.
func (s ProfileSource) String() string {
	switch s {
	case ProfileSamples:
		return "samples"
	case ProfileAuto:
		return "auto"
	case ProfileExternal:
		return "external"
	}
	return fmt.Sprintf("ProfileSource(%d)", int(s))
}

// ProfileInfo describes where a NoiseProfile came from, for UI feedback such
// as "noise profile taken from 0.00-0.50 s".
type ProfileInfo struct {
	Source ProfileSource
	// StartSec and DurationSec locate the window used by EstimateNoiseProfile
	// or NoiseProfileEstimator.Profile within the clip it scanned; both are 0 for ProfileSamples.
	StartSec, DurationSec float64
}

// NoiseProfile is a measured per-bin noise power spectrum. It is immutable
// once built and may be reused across streams and Denoisers with the same
// FrameSize.
type NoiseProfile struct {
	power     []float32
	frameSize int
	info      ProfileInfo
}

// Info reports how the profile was obtained.
func (p *NoiseProfile) Info() ProfileInfo { return p.info }

// Spectrum returns a copy of the per-bin noise power (FrameSize/2+1 values,
// in the same scale as the denoiser's internal frame power: |RFFT(hann*x)|^2).
// Feed it back to NewNoiseProfile to reuse a profile across sessions.
func (p *NoiseProfile) Spectrum() []float32 { return slices.Clone(p.power) }

// NewNoiseProfile rebuilds a profile from a spectrum previously returned by
// Spectrum (or produced elsewhere in the same scale). len(power) must be
// FrameSize/2+1 for a power-of-two FrameSize >= 64; otherwise
// ErrProfileMismatch is returned. Values are copied and floored at epsPower.
func NewNoiseProfile(power []float32) (*NoiseProfile, error) {
	n := (len(power) - 1) * 2
	if len(power) < 2 || n < dspshared.MinFrameSize || n&(n-1) != 0 {
		return nil, ErrProfileMismatch
	}
	p := &NoiseProfile{power: make([]float32, len(power)), frameSize: n, info: ProfileInfo{Source: ProfileExternal}}
	dspshared.CopyFloor(p.power, power)
	return p, nil
}

// meanPower averages the frame power spectrum over every full frame of x (hop
// HopSize, no padding) into dst (len bins), floored at epsPower, and returns
// the number of frames averaged (0 when x is shorter than one frame). It
// delegates the averaging to stft.Plan.MeanPowerInto (the generic, unfloored
// statistic) and then applies the noise-profile epsPower floor, which is this
// method's policy, not the transform's. It allocates scratch; it is a
// setup-time helper, not part of the stream path.
func (d *Denoiser) meanPower(dst, x []float32) int {
	frames := d.plan.MeanPowerInto(dst, x)
	if frames == 0 {
		return 0
	}
	dspshared.CopyFloor(dst, dst)
	return frames
}

// NoiseProfileFromSamples measures a noise profile from noise-only audio (for
// example a region the user selected on a spectrogram; the caller slices it
// out). At least FrameSize samples are required (ErrProfileTooShort); half a
// second or more gives a stable estimate.
func (d *Denoiser) NoiseProfileFromSamples(noise []float32) (*NoiseProfile, error) {
	if len(noise) < d.n {
		return nil, ErrProfileTooShort
	}
	p := &NoiseProfile{power: make([]float32, d.bins), frameSize: d.n, info: ProfileInfo{Source: ProfileSamples}}
	d.meanPower(p.power, noise)
	return p, nil
}

// SetNoiseProfile fixes the noise estimate for the stream to p; it takes
// effect from the next frame and survives Flush and Reset. Passing nil
// returns to adaptive tracking. A profile built for another FrameSize is
// rejected with ErrProfileMismatch.
func (d *Denoiser) SetNoiseProfile(p *NoiseProfile) error {
	if p == nil {
		d.profile = nil
		d.initNoiseSource()
		return nil
	}
	if p.frameSize != d.n {
		return ErrProfileMismatch
	}
	d.profile = p
	copy(d.noiseBuf, p.power)
	d.noise = d.noiseBuf
	return nil
}

// LearnNoise measures a noise profile from a noise-only excerpt and makes it the
// active noise model, the two-step NoiseProfileFromSamples then SetNoiseProfile
// path in one call. It satisfies the NoiseLearner capability interface, so a
// consumer holding a dsp.Processor can learn noise without depending on the
// concrete type. samples must be at least FrameSize long.
func (d *Denoiser) LearnNoise(samples []float32) error {
	p, err := d.NoiseProfileFromSamples(samples)
	if err != nil {
		return err
	}
	return d.SetNoiseProfile(p)
}

// NoiseProfile returns the profile set with SetNoiseProfile, or nil when the
// estimate is adaptive.
func (d *Denoiser) NoiseProfile() *NoiseProfile { return d.profile }

// Tier 1 pre-scan constants (package-level in v1; see the design spec).
const (
	autoWindowSec   = 0.5                    // length of the quiet window averaged into the profile
	silencePower    = dspshared.SilencePower // block mean-square below this (-90 dBFS RMS) is digital silence
	quietPercentile = 0.9                    // the window is compared against this block-energy percentile
	quietMarginDB   = 6.0                    // and must sit at least this far below it
)

// EstimateNoiseProfile scans a clip for its quietest autoWindowSec window and
// measures the noise profile there (Tier 1, zero-click). Blocks of digital
// silence (zero padding) never make up the window. The window is accepted only
// if the clip has dynamic range: its mean energy must be at least
// quietMarginDB below the quietPercentile of block energies, otherwise
// ErrNoQuietRegion is returned (continuous song, or featureless noise, where
// the adaptive tracker does as well). Clips shorter than about two windows skip
// that test. The stretch measured for the window (widened to one frame when
// FrameSize is longer than the window, and ending at the last whole hop block)
// must hold no block of digital silence (below -90 dBFS RMS), otherwise
// ErrNoQuietRegion is returned: a silent block would pull the profile below
// the real noise. Info() on the result reports the measured stretch. Returns
// ErrProfileTooShort for inputs shorter than one frame. For a clip that
// arrives in chunks, or is held as int16, use NewNoiseProfileEstimator, which
// returns the same result in bounded memory.
func (d *Denoiser) EstimateNoiseProfile(x []float32) (*NoiseProfile, error) {
	if len(x) < d.n {
		return nil, ErrProfileTooShort
	}
	// With the whole clip in hand there is nothing to buffer: take the block
	// energies and window selection from the estimator's shared code and read
	// the window straight from x.
	nb := len(x) / d.hop
	e := d.newNoiseProfileEstimator(nb)
	for b := range nb {
		e.addBlock(f32.SumOfSquares(x[b*d.hop:(b+1)*d.hop]) / float32(d.hop))
	}
	e.pos = len(x)
	best, w, err := e.selectWindow()
	if err != nil {
		return nil, err
	}
	start, end, err := e.measuredRange(best, w)
	if err != nil {
		return nil, err
	}
	return e.buildProfile(x[start:end], start, end)
}

// windowTracker finds the minimum-energy window of w blocks one block at a
// time. The statements in push are the sliding-sum scan of quietestWindow, so
// a stream of pushes and a whole-slice scan select identical windows.
type windowTracker struct {
	w       int
	sum     float64
	silent  int
	best    int // start block of the best window so far, -1 if none
	bestSum float64
}

func newWindowTracker(w int) windowTracker {
	return windowTracker{w: w, best: -1, bestSum: math.Inf(1)}
}

// reset returns the tracker to its initial state, keeping w.
func (t *windowTracker) reset() {
	*t = newWindowTracker(t.w)
}

// push accounts for the last element of energy (block len(energy)-1, with the
// whole history before it) and reports whether it became the new best window.
// Ties keep the earliest window; a NaN or Inf block poisons the running sum
// for every later window.
func (t *windowTracker) push(energy []float32) (improved bool) {
	b := len(energy) - 1
	e := energy[b]
	t.sum += float64(e)
	if e < silencePower {
		t.silent++
	}
	if b >= t.w {
		t.sum -= float64(energy[b-t.w])
		if energy[b-t.w] < silencePower {
			t.silent--
		}
	}
	if b >= t.w-1 && t.silent == 0 && t.sum < t.bestSum {
		t.bestSum, t.best = t.sum, b-t.w+1
		return true
	}
	return false
}

// quietestWindow returns the start block and energy sum of the minimum-energy
// window of w blocks that contains no digital-silence block, or -1 if there is
// none. Ties go to the earliest window.
func quietestWindow(energy []float32, w int) (start int, energySum float64) {
	t := newWindowTracker(w)
	for b := range energy {
		t.push(energy[:b+1])
	}
	return t.best, t.bestSum
}
