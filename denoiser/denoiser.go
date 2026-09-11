package denoiser

import (
	"github.com/tphakala/go-audio-dsp/stft"
	"github.com/tphakala/simd/f32"
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

	plan     *stft.Plan     // whole-clip transform, for noise-profile measurement
	an       *stft.Analyzer // streaming analysis: framing, window, RFFT, |X|^2
	window   []float32      // periodic Hann, analysis and synthesis (the analyzer's)
	invNorm  []float32      // len hop: 1 / WOLA normalization per position in a block
	prezeros []float32      // len n-hop: the leading-zero preroll fed on Reset

	gains    *gainState
	noise    []float32 // active noise power: noiseBuf (fixed profile) or tracker.noise
	noiseBuf []float32
	profile  *NoiseProfile
	tracker  *mcra

	// stream state
	gain              []float32 // per-bin real gain, reused each frame
	synth             []float32 // inverse frame, len n
	ola               []float32 // overlap-add accumulator, len n
	zeros             []float32 // len hop, fed by Flush
	frames            int64     // frames processed in this stream
	totalIn, totalOut int64
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
	stftCfg := stft.Config{FrameSize: rc.FrameSize, HopSize: rc.HopSize, Window: stft.Hann}
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
		an:         an,
	}
	d.window = an.Window()
	norm := stft.WOLANorm(d.window, d.window, d.hop)
	d.invNorm = make([]float32, d.hop)
	f32.Reciprocal(d.invNorm, norm) // full-precision division, not approximate rcp
	d.prezeros = make([]float32, d.n-d.hop)
	d.gains = newGainState(d.bins, p)
	d.noiseBuf = make([]float32, d.bins)
	d.gain = make([]float32, d.bins)
	d.synth = make([]float32, d.n)
	d.ola = make([]float32, d.n)
	d.zeros = make([]float32, d.hop)
	d.initNoiseSource()
	d.Reset()
	return d, nil
}

// initNoiseSource points the noise slot at the adaptive tracker (the default
// until SetNoiseProfile is called with a profile).
func (d *Denoiser) initNoiseSource() {
	if d.tracker == nil {
		d.tracker = newMCRA(d.bins, trackWindowFrames(d.params.TrackWindowSec, d.sampleRate, d.hop))
	}
	d.tracker.reset()
	d.noise = d.tracker.noise
}

// Latency returns how many samples the output lags the input in steady state:
// FrameSize - HopSize. Output is emitted in whole hop blocks, so between block
// boundaries the lag is up to HopSize-1 samples more; Flush drains the rest.
func (d *Denoiser) Latency() int { return d.n - d.hop }

// Reset clears the stream state (analysis buffer, overlap-add accumulator,
// decision-directed and tracker state, sample counters) so the next Process
// starts a new stream. The configuration and any set noise profile are kept.
func (d *Denoiser) Reset() {
	// The stream is modelled as n-hop leading zeros ++ input. Prime the analyzer
	// with that preroll so the first output sample aligns with the first input
	// sample; n-hop < n, so no frame completes and noEmit is never called.
	d.an.Reset()
	d.an.Feed(d.prezeros, noEmit)
	clear(d.ola)
	d.frames = 0
	d.totalIn, d.totalOut = 0, 0
	d.gains.reset()
	if d.tracker != nil {
		d.tracker.reset()
	}
}

// Process denoises the next chunk of the stream and returns the output samples
// that became final, in a freshly allocated slice (possibly empty). Output is
// aligned sample-for-sample with input and lags by Latency(); see Flush.
func (d *Denoiser) Process(in []float32) ([]float32, error) {
	out := make([]float32, d.pendingOutput(len(in)))
	n, err := d.ProcessInto(in, out)
	return out[:n], err
}

// ProcessInto is Process without allocation: it writes the output samples that
// become final into out and returns their count. The count is known before
// any work is done; if len(out) is smaller, ErrBufferTooSmall is returned and
// nothing is consumed. len(out) >= MaxOutputLen(len(in)) always suffices.
func (d *Denoiser) ProcessInto(in, out []float32) (int, error) {
	need := d.pendingOutput(len(in))
	if len(out) < need {
		return 0, ErrBufferTooSmall
	}
	n := d.feed(in, out[:need], false)
	d.totalIn += int64(len(in))
	d.totalOut += int64(n)
	return n, nil
}

// Flush ends the stream: it completes the frames overlapping the last input
// samples (with a frozen noise estimate), returns the remaining output so the
// total output length equals the total input length, and resets the stream
// state so the Denoiser can start a new stream.
func (d *Denoiser) Flush() ([]float32, error) {
	out := make([]float32, int(d.totalIn-d.totalOut))
	n, err := d.FlushInto(out)
	return out[:n], err
}

// FlushInto is Flush without allocation. It returns ErrBufferTooSmall (and
// does nothing) if len(out) is smaller than the remaining output;
// len(out) >= FrameSize() always suffices.
func (d *Denoiser) FlushInto(out []float32) (int, error) {
	need := int(d.totalIn - d.totalOut)
	if len(out) < need {
		return 0, ErrBufferTooSmall
	}
	written := 0
	for written < need {
		written += d.feed(d.zeros, out[written:need], true)
	}
	d.Reset()
	return written, nil
}

// Denoise denoises a whole clip held in memory: it measures the noise profile
// from the clip's quietest window (EstimateNoiseProfile), falling back to
// adaptive tracking when no distinct quiet region exists, then runs the stream
// to completion. The result has len(x) samples aligned with x.
func Denoise(x []float32, cfg Config) ([]float32, error) {
	d, err := New(cfg)
	if err != nil {
		return nil, err
	}
	if p, err := d.EstimateNoiseProfile(x); err == nil {
		_ = d.SetNoiseProfile(p) // same Denoiser, same FrameSize: cannot mismatch
	}
	return d.denoiseAll(x)
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
	return d.denoiseAll(x)
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
	return d.denoiseAll(x)
}

// denoiseAll runs x through the stream and flushes into one exact-length
// result.
func (d *Denoiser) denoiseAll(x []float32) ([]float32, error) {
	out := make([]float32, len(x))
	n, err := d.ProcessInto(x, out)
	if err != nil {
		return nil, err
	}
	m, err := d.FlushInto(out[n:])
	if err != nil {
		return nil, err
	}
	return out[:n+m], nil
}
