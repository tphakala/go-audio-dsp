package gate

import (
	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/stft"
	"github.com/tphakala/simd/f32"
)

// epsPower floors every noise power estimate so SNR ratios stay finite. It
// matches the flagship denoiser's floor; inputs are normalized float32 audio, so
// real bin powers sit far above it.
const epsPower = 1e-12

// Gate is a streaming single-channel spectral soft-gate denoiser. Create one with
// New, then call Process (or ProcessInto) with consecutive chunks and Flush at the
// end of the stream. It holds scratch and stream state and is not safe for
// concurrent use. See the package doc for the method overview.
type Gate struct {
	cfg        Config
	params     Params
	sampleRate int
	n, hop     int // FrameSize, HopSize
	ovl        int // n/hop: frames overlapping any sample
	bins       int // n/2+1
	lookahead  int // L = TimeSmoothFrames/2: frames of time-smoothing lookahead
	freqR      int // R = FreqSmoothBins/2: frequency-smoothing half-width
	warm       int // ovl-1+L: frames whose output block is discarded (leading zeros + lookahead priming)

	plan     *stft.Plan     // whole-clip transform, for the learned floor (LearnNoise only)
	an       *stft.Analyzer // streaming analysis: framing, window, RFFT, |X|^2
	window   []float32      // periodic Hann, analysis and synthesis (the analyzer's)
	invNorm  []float32      // len hop: 1 / WOLA normalization per position in a block
	prezeros []float32      // len n-hop: the leading-zero preroll fed on Reset

	gFloor        float32 // residual gain floor, FactorFromDB(-MaxAttenuationDB)
	slope, offset float32 // affine map from log10 power ratio to the sigmoid knee argument

	noise    []float32 // active noise power: noiseBuf (learned) or tracker.noise (blind)
	noiseBuf []float32
	learned  bool
	tracker  *floorTracker

	// per-frame scratch (allocated in New, never in the hot path)
	ratio    []float32   // power ratio -> log ratio -> knee argument, in place
	mask     []float32   // raw sigmoid mask -> per-frame gain in [gFloor, 1]
	acc      []float32   // time-smoothed gain accumulator
	tmp      []float32   // frequency-smoothing accumulator
	invCount []float32   // 1 / (bins in the frequency box at k), for edge-aware averaging
	gain     []float32   // final per-bin gain for the output frame
	synth    []float32   // inverse frame, len n
	ola      []float32   // overlap-add accumulator, len n
	zeros    []float32   // len hop, fed by Flush
	maskRing []float32   // (2L+1)*bins: the last 2L+1 frames' masks, for time smoothing
	specRing []complex64 // (L+1)*bins: the last L+1 frames' spectra, awaiting their delayed output

	frames            int64 // frames processed in this stream
	totalIn, totalOut int64
}

// New returns a Gate for cfg. Until LearnNoise or SetNoiseFloor is called the
// noise floor is estimated blind: a per-bin rolling median that needs about
// Params.FloorWindowSec/4 of audio before it engages (audio passes through
// unchanged until then).
func New(cfg Config) (*Gate, error) {
	rc, p, err := cfg.resolve()
	if err != nil {
		return nil, err
	}
	// One shared analysis Config drives two independent stft objects: a whole-clip
	// Plan for the learned floor (LearnNoise) and a streaming Analyzer for the
	// frame-by-frame gate path. Each owns its own simd plan, a deliberate setup-time
	// cost that keeps their transform scratch independent (the two are never used
	// concurrently). Both resolve the same periodic Hann window, so a learned floor
	// and the stream power are measured on the identical window. The two constructors
	// wrap any failure in ErrInvalidConfig and Config.resolve validates a tighter
	// range, so they never fail here.
	stftCfg := stft.Config{FrameSize: rc.FrameSize, HopSize: rc.HopSize, Window: stft.Hann}
	plan, err := stft.New(stftCfg)
	if err != nil {
		return nil, err
	}
	an, err := stft.NewAnalyzer(stftCfg)
	if err != nil {
		return nil, err
	}
	bins := plan.NumBins()
	l := p.TimeSmoothFrames / 2
	r := min(p.FreqSmoothBins/2, bins-1) // a box wider than the spectrum just averages all bins
	ovl := rc.FrameSize / rc.HopSize
	g := &Gate{
		cfg:        rc,
		params:     p,
		sampleRate: rc.SampleRate,
		n:          rc.FrameSize,
		hop:        rc.HopSize,
		ovl:        ovl,
		bins:       bins,
		lookahead:  l,
		freqR:      r,
		warm:       ovl - 1 + l,
		plan:       plan,
		an:         an,
	}
	g.window = an.Window()
	norm := stft.WOLANorm(g.window, g.window, g.hop)
	g.invNorm = make([]float32, g.hop)
	f32.Reciprocal(g.invNorm, norm) // full-precision division, not approximate rcp
	g.prezeros = make([]float32, g.n-g.hop)
	g.gFloor = float32(dsp.FactorFromDB(-float64(p.MaxAttenuationDB)))
	// knee argument = 4*(snrDB - ThresholdDB)/TransitionDB, with
	// snrDB = 10*log10(power/noise). After Log10Floored gives log10(power/noise),
	// one affine map slope*x+offset produces the argument.
	g.slope = 40 / p.TransitionDB
	g.offset = -4 * p.ThresholdDB / p.TransitionDB
	g.noiseBuf = make([]float32, bins)
	g.ratio = make([]float32, bins)
	g.mask = make([]float32, bins)
	g.acc = make([]float32, bins)
	g.tmp = make([]float32, bins)
	g.gain = make([]float32, bins)
	g.synth = make([]float32, g.n)
	g.ola = make([]float32, g.n)
	g.zeros = make([]float32, g.hop)
	g.maskRing = make([]float32, (2*l+1)*bins)
	g.specRing = make([]complex64, (l+1)*bins)
	g.invCount = make([]float32, bins)
	for k := range bins {
		lo, hi := max(0, k-r), min(bins-1, k+r)
		g.invCount[k] = 1 / float32(hi-lo+1)
	}
	g.tracker = newFloorTracker(bins, trackWindowFrames(p.FloorWindowSec, rc.SampleRate, rc.HopSize), blindEvery)
	g.initNoiseSource()
	g.Reset()
	return g, nil
}

// initNoiseSource points the noise slot at the blind tracker (the default until
// LearnNoise or SetNoiseFloor supplies a fixed floor).
func (g *Gate) initNoiseSource() {
	g.learned = false
	g.tracker.reset()
	g.noise = g.tracker.noise
}

// Latency returns how many samples the output lags the input in steady state:
// (FrameSize - HopSize) + L*HopSize, where L = TimeSmoothFrames/2 is the
// time-smoothing lookahead. Output is emitted in whole hop blocks, so between
// block boundaries the lag is up to HopSize-1 samples more; Flush drains the rest.
func (g *Gate) Latency() int { return (g.n - g.hop) + g.lookahead*g.hop }

// Reset clears the stream state (analysis buffer, overlap-add accumulator, the
// smoothing rings, the blind tracker, sample counters) so the next Process starts
// a new stream. The configuration and any learned floor are kept.
func (g *Gate) Reset() {
	// The stream is modelled as n-hop leading zeros ++ input. Prime the analyzer
	// with that preroll so the first output sample aligns with the first input
	// sample; n-hop < n, so no frame completes and noEmit is never called.
	g.an.Reset()
	g.an.Feed(g.prezeros, noEmit)
	clear(g.ola)
	// Prime the mask ring to unity gain, not zero: the pre-stream frames a
	// start-edge output frame averages over represent "no gating yet"
	// (pass-through), so a unity-floor stream reconstructs exactly and real gating
	// only eases in over the first L frames rather than over-attenuating them.
	for i := range g.maskRing {
		g.maskRing[i] = 1
	}
	clear(g.specRing)
	g.frames = 0
	g.totalIn, g.totalOut = 0, 0
	if g.tracker != nil {
		g.tracker.reset()
	}
}

// Process gates the next chunk of the stream and returns the output samples that
// became final, in a freshly allocated slice (possibly empty). Output is aligned
// sample-for-sample with input and lags by Latency(); see Flush.
func (g *Gate) Process(in []float32) ([]float32, error) {
	out := make([]float32, g.pendingOutput(len(in)))
	n, err := g.ProcessInto(in, out)
	return out[:n], err
}

// ProcessInto is Process without allocation: it writes the output samples that
// become final into out and returns their count. The count is known before any
// work is done; if len(out) is smaller, ErrBufferTooSmall is returned and nothing
// is consumed. len(out) >= MaxOutputLen(len(in)) always suffices.
func (g *Gate) ProcessInto(in, out []float32) (int, error) {
	need := g.pendingOutput(len(in))
	if len(out) < need {
		return 0, ErrBufferTooSmall
	}
	n := g.feed(in, out[:need], false)
	g.totalIn += int64(len(in))
	g.totalOut += int64(n)
	return n, nil
}

// Flush ends the stream: it completes the frames overlapping the last input
// samples (with a frozen noise floor), returns the remaining output so the total
// output length equals the total input length, and resets the stream state so the
// Gate can start a new stream.
func (g *Gate) Flush() ([]float32, error) {
	out := make([]float32, int(g.totalIn-g.totalOut))
	n, err := g.FlushInto(out)
	return out[:n], err
}

// FlushInto is Flush without allocation. It returns ErrBufferTooSmall (and does
// nothing) if len(out) is smaller than the remaining output;
// len(out) >= Latency()+HopSize() always suffices.
func (g *Gate) FlushInto(out []float32) (int, error) {
	need := int(g.totalIn - g.totalOut)
	if len(out) < need {
		return 0, ErrBufferTooSmall
	}
	written := 0
	for written < need {
		written += g.feed(g.zeros, out[written:need], true)
	}
	g.Reset()
	return written, nil
}

// Denoise gates a whole clip held in memory with the blind noise-floor estimator,
// streamed to completion. For a clip with no noise-only lead-in, prefer
// DenoiseWithNoise or DenoiseWithFloor: unlike the flagship's Denoise, the gate
// does not run a quietest-window pre-scan (its blind path is a rolling median
// designed for gapless continuous noise). The result has len(x) samples aligned
// with x.
func Denoise(x []float32, cfg Config) ([]float32, error) {
	g, err := New(cfg)
	if err != nil {
		return nil, err
	}
	return g.denoiseAll(x)
}

// DenoiseWithNoise gates a whole clip using a floor learned from the given
// noise-only samples (for example a user-selected region of the same clip). The
// result has len(x) samples aligned with x.
func DenoiseWithNoise(x, noise []float32, cfg Config) ([]float32, error) {
	g, err := New(cfg)
	if err != nil {
		return nil, err
	}
	if err := g.LearnNoise(noise); err != nil {
		return nil, err
	}
	return g.denoiseAll(x)
}

// DenoiseWithFloor gates a whole clip with an explicit per-bin noise power floor
// (FrameSize/2+1 values in the same scale as NoiseFloor returns, for example one
// measured once and applied to every clip from the same station). The result has
// len(x) samples aligned with x.
func DenoiseWithFloor(x, floor []float32, cfg Config) ([]float32, error) {
	g, err := New(cfg)
	if err != nil {
		return nil, err
	}
	if err := g.SetNoiseFloor(floor); err != nil {
		return nil, err
	}
	return g.denoiseAll(x)
}

// denoiseAll runs x through the stream and flushes into one exact-length result.
func (g *Gate) denoiseAll(x []float32) ([]float32, error) {
	out := make([]float32, len(x))
	n, err := g.ProcessInto(x, out)
	if err != nil {
		return nil, err
	}
	m, err := g.FlushInto(out[n:])
	if err != nil {
		return nil, err
	}
	return out[:n+m], nil
}
