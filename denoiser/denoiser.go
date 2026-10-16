package denoiser

import (
	"github.com/tphakala/go-audio-dsp/internal/dspshared"
	"github.com/tphakala/go-audio-dsp/stft"
)

// Denoiser is a streaming single-channel spectral denoiser. Create one with
// New, then call Process (or ProcessInto) with consecutive chunks and Flush at
// the end of the stream. It holds scratch and stream state and is not safe for
// concurrent use.
type Denoiser struct {
	cfg        Config
	params     Params
	sampleRate int
	n, hop     int // FrameSize, HopSize
	ovl        int // n/hop: frames overlapping any sample
	bins       int // n/2+1

	plan *stft.Plan // whole-clip transform, for noise-profile measurement
	st   dspshared.Stream

	gains    *gainState
	noise    []float32 // active noise power: noiseBuf (fixed profile) or tracker.noise
	noiseBuf []float32
	profile  *NoiseProfile
	tracker  *mcra

	gain []float32 // per-bin real gain, reused each frame
}

// New returns a Denoiser for cfg. Until SetNoiseProfile is called the noise
// estimate is adaptive: a minima-controlled tracker that needs about
// Params.TrackWindowSec of audio to converge.
func New(cfg Config) (*Denoiser, error) {
	rc, p, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	// One shared analysis Config drives two independent stft objects: a whole-clip
	// Plan for noise-profile measurement and a streaming Analyzer for the
	// frame-by-frame denoise path. Each owns its own simd plan, a deliberate
	// setup-time cost that keeps their transform scratch independent (the two are
	// never used concurrently). Both resolve the same periodic Hann window, so the
	// profile power and the stream power are measured on the identical window. The
	// two constructors already wrap any failure in ErrInvalidConfig, and the
	// denoiser's own Config.resolve validates a tighter range, so they never fail here.
	stftCfg := dspshared.STFTConfig(rc.FrameSize, rc.HopSize)
	plan, err := stft.New(stftCfg)
	if err != nil {
		return nil, err
	}
	an, err := stft.NewAnalyzer(stftCfg)
	if err != nil {
		return nil, err
	}
	d := &Denoiser{
		cfg:        rc,
		params:     p,
		sampleRate: rc.SampleRate,
		n:          rc.FrameSize,
		hop:        rc.HopSize,
		ovl:        rc.FrameSize / rc.HopSize,
		bins:       plan.NumBins(),
		plan:       plan,
	}
	if err := d.st.Init(an, 0, d.frame, d.resetState); err != nil {
		return nil, err
	}
	d.gains = newGainState(d.bins, p)
	d.noiseBuf = make([]float32, d.bins)
	d.gain = make([]float32, d.bins)
	d.initNoiseSource()
	d.Reset()
	return d, nil
}

// initNoiseSource points the noise slot at the adaptive tracker (the default
// until SetNoiseProfile is called with a profile).
func (d *Denoiser) initNoiseSource() {
	if d.tracker == nil {
		d.tracker = newMCRA(d.bins, dspshared.TrackWindowFrames(d.params.TrackWindowSec, d.sampleRate, d.hop))
	}
	d.tracker.reset()
	d.noise = d.tracker.noise
}

// Latency returns how many samples the output lags the input in steady state:
// FrameSize - HopSize. Output is emitted in whole hop blocks, so between block
// boundaries the lag is up to HopSize-1 samples more; Flush drains the rest.
func (d *Denoiser) Latency() int { return d.st.Latency() }

// Reset clears the stream state (analysis buffer, overlap-add accumulator,
// decision-directed and tracker state, sample counters) so the next Process
// starts a new stream. The configuration and any set noise profile are kept.
func (d *Denoiser) Reset() { d.st.Reset() }

// resetState clears the decision-directed gain and tracker state; the stream
// driver calls it from Reset.
func (d *Denoiser) resetState() {
	d.gains.reset()
	if d.tracker != nil {
		d.tracker.reset()
	}
}

// Process denoises the next chunk of the stream and returns the output samples
// that became final, in a freshly allocated slice (possibly empty). Output is
// aligned sample-for-sample with input and lags by Latency(); see Flush.
func (d *Denoiser) Process(in []float32) ([]float32, error) { return d.st.Process(in) }

// ProcessInto is Process without allocation: it writes the output samples that
// become final into out and returns their count. The count is known before
// any work is done; if len(out) is smaller, ErrBufferTooSmall is returned and
// nothing is consumed. len(out) >= MaxOutputLen(len(in)) always suffices.
func (d *Denoiser) ProcessInto(in, out []float32) (int, error) { return d.st.ProcessInto(in, out) }

// Flush ends the stream: it completes the frames overlapping the last input
// samples (with a frozen noise estimate), returns the remaining output so the
// total output length equals the total input length, and resets the stream
// state so the Denoiser can start a new stream.
func (d *Denoiser) Flush() ([]float32, error) { return d.st.Flush() }

// FlushInto is Flush without allocation. It returns ErrBufferTooSmall (and
// does nothing) if len(out) is smaller than the remaining output;
// len(out) >= Latency()+HopSize() (which is FrameSize()) always suffices.
func (d *Denoiser) FlushInto(out []float32) (int, error) { return d.st.FlushInto(out) }

// Denoise denoises a whole clip held in memory: it measures the noise profile
// from the clip's quietest window (EstimateNoiseProfile), falling back to
// adaptive tracking when it finds no usable quiet window (none distinctly
// quiet, or the quietest holds digital silence), then runs the stream to
// completion. The result has len(x) samples aligned with x.
func Denoise(x []float32, cfg Config) ([]float32, error) {
	d, err := New(cfg)
	if err != nil {
		return nil, err
	}
	if p, err := d.EstimateNoiseProfile(x); err == nil {
		_ = d.SetNoiseProfile(p) // same Denoiser, same FrameSize: cannot mismatch
	}
	return d.st.ProcessAll(x)
}

// DenoiseWithNoise denoises a whole clip using a profile measured from the
// given noise-only samples (for example a user-selected region of the same
// clip). The result has len(x) samples aligned with x.
func DenoiseWithNoise(x, noise []float32, cfg Config) ([]float32, error) {
	d, err := New(cfg)
	if err != nil {
		return nil, err
	}
	p, err := d.NoiseProfileFromSamples(noise)
	if err != nil {
		return nil, err
	}
	if err := d.SetNoiseProfile(p); err != nil {
		return nil, err
	}
	return d.st.ProcessAll(x)
}

// DenoiseWithProfile denoises a whole clip with an existing profile (for
// example one measured once and applied to every clip from the same station).
// A nil profile means adaptive tracking. The result has len(x) samples.
func DenoiseWithProfile(x []float32, p *NoiseProfile, cfg Config) ([]float32, error) {
	d, err := New(cfg)
	if err != nil {
		return nil, err
	}
	if err := d.SetNoiseProfile(p); err != nil {
		return nil, err
	}
	return d.st.ProcessAll(x)
}
