package denoiser

import (
	"fmt"
	"math"
	"slices"

	"github.com/tphakala/go-audio-dsp/stft"
	"github.com/tphakala/simd/f32"
)

// ProfileSource tells how a NoiseProfile was obtained.
type ProfileSource int

const (
	// ProfileSamples: measured from caller-supplied noise-only samples.
	ProfileSamples ProfileSource = iota
	// ProfileAuto: measured from the quietest window EstimateNoiseProfile found.
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
	// within the clip it scanned; both are 0 for ProfileSamples.
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
	if len(power) < 2 || n < minFrameSize || n&(n-1) != 0 {
		return nil, ErrProfileMismatch
	}
	p := &NoiseProfile{power: make([]float32, len(power)), frameSize: n, info: ProfileInfo{Source: ProfileExternal}}
	for k, v := range power {
		if !(v > epsPower) || !(v < math.MaxFloat32) { // NaN, <= floor, or +Inf
			v = epsPower
		}
		p.power[k] = v
	}
	return p, nil
}

// profileBatchFrames bounds the scratch meanPower uses per STFT batch.
const profileBatchFrames = 64

// meanPower averages the frame power spectrum over every full frame of x (hop
// HopSize, no padding) into dst (len bins), floored at epsPower, and returns
// the number of frames averaged (0 when x is shorter than one frame). It
// allocates scratch; it is a setup-time helper, not part of the stream path.
func (d *Denoiser) meanPower(dst, x []float32) int {
	frames := d.plan.NumFrames(len(x), stft.NoPad)
	if frames == 0 {
		return 0
	}
	acc := make([]float64, d.bins)
	flat := make([]float32, profileBatchFrames*d.bins)
	for f0 := 0; f0 < frames; f0 += profileBatchFrames {
		nf := min(profileBatchFrames, frames-f0)
		start := f0 * d.hop
		end := min(len(x), start+(nf-1)*d.hop+d.n)
		got := d.plan.PowerInto(flat[:nf*d.bins], x[start:end], stft.NoPad)
		for f := range got {
			row := flat[f*d.bins : (f+1)*d.bins]
			for k, v := range row {
				acc[k] += float64(v)
			}
		}
	}
	for k := range dst {
		v := float32(acc[k] / float64(frames))
		if !(v > epsPower) || !(v < math.MaxFloat32) { // NaN, <= floor, or +Inf
			v = epsPower
		}
		dst[k] = v
	}
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
	autoWindowSec   = 0.5  // length of the quiet window averaged into the profile
	silencePower    = 1e-9 // block mean-square below this (-90 dBFS RMS) is digital silence
	quietPercentile = 0.9  // the window is compared against this block-energy percentile
	quietMarginDB   = 6.0  // and must sit at least this far below it
)

// EstimateNoiseProfile scans a clip for its quietest autoWindowSec window and
// measures the noise profile there (Tier 1, zero-click). Blocks of digital
// silence (zero padding) are skipped when anything else exists. The window is
// accepted only if the clip has dynamic range: its mean energy must be at
// least quietMarginDB below the quietPercentile of block energies, otherwise
// ErrNoQuietRegion is returned (continuous song, or featureless noise, where
// the adaptive tracker does as well). Clips shorter than about two windows
// accept the quietest window unconditionally. Info() on the result reports
// the window position. Returns ErrProfileTooShort for inputs shorter than one
// frame.
func (d *Denoiser) EstimateNoiseProfile(x []float32) (*NoiseProfile, error) {
	if len(x) < d.n {
		return nil, ErrProfileTooShort
	}
	hop := d.hop
	nb := len(x) / hop
	energy := make([]float32, nb) // mean square per hop block
	for b := range energy {
		energy[b] = f32.SumOfSquares(x[b*hop:(b+1)*hop]) / float32(hop)
	}
	w := max(1, int(math.Round(autoWindowSec*float64(d.sampleRate)/float64(hop))))
	w = min(w, nb)

	best, bestSum := quietestWindow(energy, w, true)
	if best < 0 {
		best, bestSum = quietestWindow(energy, w, false) // everything touches silence
	}
	if best < 0 {
		// No finite window: a NaN or Inf sample poisons the running energy sum,
		// so no window is ever selected. Fall back to adaptive tracking (which
		// guards non-finite bins itself) rather than slicing at a negative index.
		return nil, ErrNoQuietRegion
	}

	// Acceptance: enough unmasked blocks to compare against, and the window
	// must be quietMarginDB below the quietPercentile block energy.
	unmasked := make([]float32, 0, nb)
	for _, e := range energy {
		if e >= silencePower {
			unmasked = append(unmasked, e)
		}
	}
	if len(unmasked) >= 2*w {
		slices.Sort(unmasked)
		ref := unmasked[int(quietPercentile*float64(len(unmasked)-1))]
		winMean := bestSum / float64(w)
		if winMean > float64(ref)*math.Pow(10, -quietMarginDB/10) {
			return nil, ErrNoQuietRegion
		}
	}

	start := best * hop
	end := min(len(x), start+w*hop)
	if end-start < d.n { // window shorter than a frame (tiny clips): widen it
		end = min(len(x), start+d.n)
		start = max(0, end-d.n)
	}
	p, err := d.NoiseProfileFromSamples(x[start:end])
	if err != nil {
		return nil, err
	}
	p.info = ProfileInfo{
		Source:      ProfileAuto,
		StartSec:    float64(start) / float64(d.sampleRate),
		DurationSec: float64(end-start) / float64(d.sampleRate),
	}
	return p, nil
}

// quietestWindow returns the start block and energy sum of the minimum-energy
// window of w blocks; with maskSilence it considers only windows containing
// no digital-silence block, returning -1 if there is none. Ties go to the
// earliest window.
func quietestWindow(energy []float32, w int, maskSilence bool) (start int, energySum float64) {
	best, bestSum := -1, math.Inf(1)
	var sum float64
	silent := 0
	for b, e := range energy {
		sum += float64(e)
		if e < silencePower {
			silent++
		}
		if b >= w {
			sum -= float64(energy[b-w])
			if energy[b-w] < silencePower {
				silent--
			}
		}
		if b >= w-1 && (!maskSilence || silent == 0) && sum < bestSum {
			bestSum, best = sum, b-w+1
		}
	}
	return best, bestSum
}
