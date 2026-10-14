package denoiser

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/pcm"
	"github.com/tphakala/simd/f32"
)

// referenceEstimate is a verbatim copy of EstimateNoiseProfile as of 7f76d8c
// (before the streaming estimator), the parity oracle for the estimator and
// the EstimateNoiseProfile wrapper. Do not edit it to follow the production code.
func referenceEstimate(d *Denoiser, x []float32) (*NoiseProfile, error) {
	if len(x) < d.n {
		return nil, ErrProfileTooShort
	}
	hop := d.hop
	nb := len(x) / hop
	energy := make([]float32, nb)
	for b := range energy {
		energy[b] = f32.SumOfSquares(x[b*hop:(b+1)*hop]) / float32(hop)
	}
	w := max(1, int(math.Round(autoWindowSec*float64(d.sampleRate)/float64(hop))))
	w = min(w, nb)

	best, bestSum := referenceQuietestWindow(energy, w, true)
	if best < 0 {
		best, bestSum = referenceQuietestWindow(energy, w, false)
	}
	if best < 0 {
		return nil, ErrNoQuietRegion
	}

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
	if end-start < d.n {
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

func referenceQuietestWindow(energy []float32, w int, maskSilence bool) (start int, energySum float64) {
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

// sameProfile fails t unless got/gotErr equal want/wantErr: the same error
// sentinel, a bit-identical spectrum and an identical Info.
func sameProfile(t *testing.T, got *NoiseProfile, gotErr error, want *NoiseProfile, wantErr error) {
	t.Helper()
	if (gotErr == nil) != (wantErr == nil) || (wantErr != nil && !errors.Is(gotErr, wantErr)) {
		t.Fatalf("err = %v, want %v", gotErr, wantErr)
	}
	if wantErr != nil {
		return
	}
	if got.Info() != want.Info() {
		t.Fatalf("info = %+v, want %+v", got.Info(), want.Info())
	}
	gs, ws := got.Spectrum(), want.Spectrum()
	if len(gs) != len(ws) {
		t.Fatalf("spectrum has %d bins, want %d", len(gs), len(ws))
	}
	for k := range ws {
		if gs[k] != ws[k] {
			t.Fatalf("spectrum bin %d = %g, want %g (bit-exact)", k, gs[k], ws[k])
		}
	}
}

// profileCase is one named clip and the Config its Denoiser uses.
type profileCase struct {
	name string
	cfg  Config
	x    []float32
}

func mustDenoiser(t *testing.T, cfg Config) *Denoiser {
	t.Helper()
	d, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// noiseSpan overwrites x[from:to] with white noise of the given RMS.
func noiseSpan(x []float32, from, to int, rms float64, seed uint64) {
	copy(x[from:to], whiteNoise(to-from, rms, seed))
}

// quietBlockClip is nblocks hop blocks of loud noise plus tail extra samples,
// with block quiet replaced by very quiet noise.
func quietBlockClip(hop, nblocks, tail, quiet int, seed uint64) []float32 {
	x := whiteNoise(nblocks*hop+tail, 0.05, seed)
	noiseSpan(x, quiet*hop, (quiet+1)*hop, 0.001, seed+100)
	return x
}

const (
	caseUnmaskedFallback = "unmasked-fallback"
	caseEarlyQuiet       = "early-quiet-then-loud"
	caseQuietAtEnd       = "quiet-at-end"
	caseWidenMidLong     = "widen-48k-mid-long"
)

func profileCases() []profileCase {
	const sr = 48000
	def := Config{SampleRate: sr}
	var cs []profileCase
	add := func(name string, cfg Config, x []float32) {
		cs = append(cs, profileCase{name: name, cfg: cfg, x: x})
	}

	planted := whiteNoise(10*sr, 0.01, 21)
	toneAt(planted, sr, 0, 4*sr, 1000, 0.1*math.Sqrt2)
	toneAt(planted, sr, 5*sr, 10*sr, 1000, 0.1*math.Sqrt2)
	add("planted-quiet-window", def, planted)

	burst := whiteNoise(15*sr, 0.01, 22)
	toneAt(burst, sr, 6*sr, 9*sr, 2000, 0.1)
	add("noise-with-burst", def, burst)

	tone := make([]float32, 10*sr)
	toneAt(tone, sr, 0, len(tone), 1000, 0.1)
	add("continuous-tone", def, tone)
	add("featureless-noise", def, whiteNoise(10*sr, 0.01, 23))
	add("too-short", def, whiteNoise(1000, 0.01, 24))

	nan := whiteNoise(4*sr, 0.01, 71)
	nan[2*sr] = float32(math.NaN())
	add("nan-at-2s", def, nan)
	inf := whiteNoise(4*sr, 0.01, 72)
	inf[3*sr] = float32(math.Inf(1))
	add("inf-at-3s", def, inf)
	nanFirst := whiteNoise(4*sr, 0.01, 73)
	nanFirst[10] = float32(math.NaN())
	add("nan-in-first-block", def, nanFirst)
	nanLate := whiteNoise(10*sr, 0.05, 74)
	noiseSpan(nanLate, sr, 2*sr, 0.002, 75)
	nanLate[8*sr] = float32(math.NaN())
	add("nan-late-after-quiet", def, nanLate)

	lead := make([]float32, 8*sr)
	copy(lead[sr:], whiteNoise(7*sr, 0.01, 76))
	toneAt(lead, sr, 3*sr, 5*sr, 3000, 0.2)
	add("leading-silence", def, lead)
	add("all-silent", def, make([]float32, 4*sr))

	// Every window touches digital silence, yet the clip has dynamic range:
	// quiet noise first, loud tone after, one zeroed hop block every 0.25 s.
	dyn := whiteNoise(8*sr, 0.01, 77)
	toneAt(dyn, sr, sr, 8*sr, 1000, 0.1*math.Sqrt2)
	for at := 0; at+256 <= len(dyn); at += 12000 {
		clear(dyn[at/256*256 : at/256*256+256])
	}
	add(caseUnmaskedFallback, def, dyn)
	flatSilent := whiteNoise(8*sr, 0.01, 78)
	for at := 0; at+256 <= len(flatSilent); at += 12000 {
		clear(flatSilent[at/256*256 : at/256*256+256])
	}
	add("silence-holes-featureless", def, flatSilent)

	w0 := 94 // round(0.5*48000/256)
	for _, n := range []int{1024, 1024 + 128, 14400, w0*256 - 1, w0 * 256, w0*256 + 255} {
		x := whiteNoise(n, 0.02, uint64(100+n))
		noiseSpan(x, n/3, n/3+min(2000, n/3), 0.002, uint64(200+n))
		add(fmt.Sprintf("short-%d", n), def, x)
	}

	early := whiteNoise(22*sr, 0.05, 79)
	noiseSpan(early, 0, sr, 0.003, 80)
	add(caseEarlyQuiet, def, early)

	end := whiteNoise(11*sr, 0.05, 81)
	noiseSpan(end, 10*sr, 11*sr, 0.002, 82)
	add(caseQuietAtEnd, def, end)

	fade := whiteNoise(10*sr, 1, 83)
	for i := range fade {
		fade[i] *= float32(0.1 * (1 - 0.99*float64(i)/float64(len(fade))))
	}
	add("monotonic-fade-out", def, fade)

	big := Config{SampleRate: sr, FrameSize: 65536} // hop 16384, w0 1, m 49152
	add("widen-48k-mid-short", big, quietBlockClip(16384, 12, 3000, 4, 84))
	add(caseWidenMidLong, big, quietBlockClip(16384, 29, 100, 5, 85))
	add("widen-48k-end-left", big, quietBlockClip(16384, 20, 5000, 19, 86))
	add("widen-48k-start", big, quietBlockClip(16384, 20, 5000, 0, 87))
	small := Config{SampleRate: 8000, FrameSize: 8192} // hop 2048, w0 2, m 4096
	add("widen-8k-mid", small, quietBlockClip(2048, 70, 500, 30, 88))
	add("widen-8k-end-left", small, quietBlockClip(2048, 40, 1500, 39, 89))
	add("widen-8k-start", small, quietBlockClip(2048, 40, 1500, 0, 90))

	add("hop-512", Config{SampleRate: sr, FrameSize: 1024, HopSize: 512}, planted)
	add("44k1-default", Config{SampleRate: 44100}, whiteNoiseWithHole(44100, 12, 4, 5, 91))
	return cs
}

// whiteNoiseWithHole is seconds of loud noise with a quiet hole [from, to) s.
func whiteNoiseWithHole(sr, seconds, from, to int, seed uint64) []float32 {
	x := whiteNoise(seconds*sr, 0.05, seed)
	noiseSpan(x, from*sr, to*sr, 0.002, seed+1)
	return x
}

func findCase(t *testing.T, name string) profileCase {
	t.Helper()
	for _, c := range profileCases() {
		if c.name == name {
			return c
		}
	}
	t.Fatalf("no profile case %q", name)
	return profileCase{}
}

// feedFloat streams x into e in chunks of the sizes next returns.
func feedFloat(e *NoiseProfileEstimator, x []float32, next func() int) {
	for len(x) > 0 {
		k := min(max(1, next()), len(x))
		e.AddFloat32(x[:k])
		x = x[k:]
	}
}

func fixed(k int) func() int { return func() int { return k } }

// randomSizes returns a seeded schedule of chunk sizes in [1, maxSize].
func randomSizes(seed uint64, maxSize int) func() int {
	rng := rand.New(rand.NewPCG(seed, 7))
	return func() int { return 1 + rng.IntN(maxSize) }
}

func chunkSchedules(hop int) map[string]func() func() int {
	s := map[string]func() func() int{
		"random": func() func() int { return randomSizes(5, 3*hop) },
	}
	for _, k := range []int{1, 2, 3, 7, 13, 97, hop - 1, hop, hop + 1, 2*hop + 1, 4801, 65537, math.MaxInt32} {
		s[fmt.Sprintf("chunk-%d", k)] = func() func() int { return fixed(k) }
	}
	return s
}

func TestEstimateNoiseProfileMatchesReference(t *testing.T) {
	for _, c := range profileCases() {
		t.Run(c.name, func(t *testing.T) {
			d := mustDenoiser(t, c.cfg)
			want, wantErr := referenceEstimate(d, c.x)
			got, gotErr := d.EstimateNoiseProfile(c.x)
			sameProfile(t, got, gotErr, want, wantErr)
		})
	}
}

func TestNoiseProfileEstimatorMatchesReferenceAcrossChunkSizes(t *testing.T) {
	for _, c := range profileCases() {
		t.Run(c.name, func(t *testing.T) {
			d := mustDenoiser(t, c.cfg)
			want, wantErr := referenceEstimate(d, c.x)
			for name, mk := range chunkSchedules(d.hop) {
				if name == "chunk-1" && len(c.x) > 2*c.cfg.SampleRate {
					continue // keep -race runtime reasonable
				}
				e := d.NewNoiseProfileEstimator()
				feedFloat(e, c.x, mk())
				got, gotErr := e.Profile()
				if t.Failed() {
					return
				}
				t.Run(name, func(t *testing.T) { sameProfile(t, got, gotErr, want, wantErr) })
			}
		})
	}
}

// randomClip builds a clip of random segments (noise at random levels, tone,
// digital silence, an occasional NaN) and picks a config.
func randomClip(rng *rand.Rand) ([]float32, Config) {
	cfgs := []Config{
		{SampleRate: 48000},
		{SampleRate: 44100},
		{SampleRate: 16000, FrameSize: 512, HopSize: 128},
		{SampleRate: 8000, FrameSize: 8192},
		{SampleRate: 48000, FrameSize: 65536},
	}
	cfg := cfgs[rng.IntN(len(cfgs))]
	total := 200 + rng.IntN(6*cfg.SampleRate)
	x := make([]float32, total)
	for at := 0; at < total; {
		n := min(total-at, 1+rng.IntN(cfg.SampleRate))
		switch rng.IntN(4) {
		case 0:
			noiseSpan(x, at, at+n, math.Pow(10, -1-3*rng.Float64()), rng.Uint64())
		case 1:
			toneAt(x, cfg.SampleRate, at, at+n, 200+rng.Float64()*3000, 0.05+0.2*rng.Float64())
		case 2: // digital silence: already zero
		default:
			noiseSpan(x, at, at+n, 0.001, rng.Uint64())
		}
		at += n
	}
	if rng.IntN(8) == 0 {
		x[rng.IntN(total)] = float32(math.NaN())
	}
	return x, cfg
}

func TestNoiseProfileEstimatorRandomClipsAndChunkings(t *testing.T) {
	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, 99))
		x, cfg := randomClip(rng)
		d := mustDenoiser(t, cfg)
		want, wantErr := referenceEstimate(d, x)
		e := d.NewNoiseProfileEstimator()
		feedFloat(e, x, randomSizes(seed, 3*d.hop))
		got, gotErr := e.Profile()
		if t.Failed() {
			return
		}
		t.Run(fmt.Sprintf("seed-%d", seed), func(t *testing.T) { sameProfile(t, got, gotErr, want, wantErr) })
	}
}

func TestNoiseProfileEstimatorEvictionPathCovered(t *testing.T) {
	tests := []struct {
		name         string
		wantResident bool
		useUnmasked  bool
	}{
		{caseEarlyQuiet, false, false},
		{caseUnmaskedFallback, false, true},
		{caseWidenMidLong, false, false},
		{caseQuietAtEnd, true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := findCase(t, tt.name)
			d := mustDenoiser(t, c.cfg)
			e := d.NewNoiseProfileEstimator()
			feedFloat(e, c.x, fixed(4800))
			got, gotErr := e.Profile()
			want, wantErr := referenceEstimate(d, c.x)
			sameProfile(t, got, gotErr, want, wantErr)
			if gotErr != nil {
				t.Fatalf("case must produce a profile, got %v", gotErr)
			}
			idx := 0
			if tt.useUnmasked {
				idx = 1
				if e.masked.best >= 0 {
					t.Fatal("masked tracker found a window; case does not exercise the unmasked fallback")
				}
			}
			cand := e.cand[idx]
			if !cand.active {
				t.Fatalf("candidate %d is not active", idx)
			}
			if cand.resident != tt.wantResident {
				t.Fatalf("candidate resident = %v, want %v", cand.resident, tt.wantResident)
			}
		})
	}
}

// int16Clip is a random int16 clip with a planted quiet window.
func int16Clip(n int, seed uint64) []int16 {
	rng := rand.New(rand.NewPCG(seed, 3))
	x := make([]int16, n)
	for i := range x {
		amp := 4000.0
		if i > n/3 && i < n/3+30000 {
			amp = 40
		}
		x[i] = int16(rng.NormFloat64() * amp)
	}
	return x
}

func TestNoiseProfileEstimatorInt16MatchesConvertFirst(t *testing.T) {
	d := mustDenoiser(t, Config{SampleRate: 48000})
	pcmClip := int16Clip(10*48000, 5)
	whole := make([]float32, len(pcmClip))
	pcm.Int16ToFloat32(whole, pcmClip)
	want, wantErr := referenceEstimate(d, whole)
	if wantErr != nil {
		t.Fatal(wantErr)
	}
	for name, mk := range chunkSchedules(d.hop) {
		t.Run(name, func(t *testing.T) {
			e := d.NewNoiseProfileEstimator()
			next := mk()
			for rest := pcmClip; len(rest) > 0; {
				k := min(max(1, next()), len(rest))
				e.AddInt16(rest[:k])
				rest = rest[k:]
			}
			got, gotErr := e.Profile()
			sameProfile(t, got, gotErr, want, wantErr)
		})
	}
	t.Run("mixed", func(t *testing.T) {
		e := d.NewNoiseProfileEstimator()
		next := randomSizes(11, 3*d.hop)
		for i, rest := 0, pcmClip; len(rest) > 0; i++ {
			k := min(max(1, next()), len(rest))
			if i%2 == 0 {
				e.AddInt16(rest[:k])
			} else {
				f := make([]float32, k)
				pcm.Int16ToFloat32(f, rest[:k])
				e.AddFloat32(f)
			}
			rest = rest[k:]
		}
		got, gotErr := e.Profile()
		sameProfile(t, got, gotErr, want, wantErr)
	})
}

func TestNoiseProfileEstimatorProfileIsRepeatable(t *testing.T) {
	const sr = 48000
	d := mustDenoiser(t, Config{SampleRate: sr})
	// a: short and loud-ish with a quiet stretch; b: quieter, so the best
	// window moves into b. Splits also cross the nb < w0 to nb >= w0 boundary.
	a := whiteNoise(sr/2, 0.05, 31)
	b := whiteNoise(6*sr, 0.05, 32)
	noiseSpan(b, 2*sr, 3*sr, 0.001, 33)
	for _, split := range []int{1024, 5000, 24064 - 1, 24064, 30000, sr / 2} {
		full := slices.Concat(a, b)
		first, rest := full[:split], full[split:]
		e := d.NewNoiseProfileEstimator()
		feedFloat(e, first, fixed(997))
		p1, err1 := e.Profile()
		p1again, err1again := e.Profile()
		w1, w1Err := referenceEstimate(d, first)
		sameProfile(t, p1, err1, w1, w1Err)
		sameProfile(t, p1again, err1again, w1, w1Err)
		feedFloat(e, rest, fixed(997))
		p2, err2 := e.Profile()
		w2, w2Err := referenceEstimate(d, full)
		sameProfile(t, p2, err2, w2, w2Err)
		if t.Failed() {
			t.Fatalf("split %d failed", split)
		}
	}
}

func TestNoiseProfileEstimatorResetMatchesFresh(t *testing.T) {
	d := mustDenoiser(t, Config{SampleRate: 48000})
	clipA := findCase(t, caseEarlyQuiet).x // best window early, snapshotted
	clipB := findCase(t, "noise-with-burst").x[:8*48000]
	e := d.NewNoiseProfileEstimator()
	feedFloat(e, clipA, fixed(4800))
	if _, err := e.Profile(); err != nil {
		t.Fatal(err)
	}
	e.Reset()
	if e.Len() != 0 {
		t.Fatalf("Len after Reset = %d, want 0", e.Len())
	}
	if _, err := e.Profile(); !errors.Is(err, ErrProfileTooShort) {
		t.Fatalf("Profile after Reset: err = %v, want ErrProfileTooShort", err)
	}
	feedFloat(e, clipB, fixed(3001))
	got, gotErr := e.Profile()
	fresh := d.NewNoiseProfileEstimator()
	feedFloat(fresh, clipB, fixed(3001))
	f, fErr := fresh.Profile()
	sameProfile(t, got, gotErr, f, fErr)
	want, wantErr := referenceEstimate(d, clipB)
	sameProfile(t, got, gotErr, want, wantErr)
}

func TestNoiseProfileEstimatorDoesNotRetainInput(t *testing.T) {
	c := findCase(t, caseEarlyQuiet)
	d := mustDenoiser(t, c.cfg)
	want, wantErr := referenceEstimate(d, c.x)
	e := d.NewNoiseProfileEstimator()
	buf := make([]float32, 4800)
	for rest := c.x; len(rest) > 0; {
		k := copy(buf, rest)
		e.AddFloat32(buf[:k])
		for i := range buf {
			buf[i] = float32(math.NaN())
		}
		rest = rest[k:]
	}
	got, gotErr := e.Profile()
	sameProfile(t, got, gotErr, want, wantErr)

	ibuf := make([]int16, 4800)
	pcmClip := int16Clip(6*48000, 9)
	whole := make([]float32, len(pcmClip))
	pcm.Int16ToFloat32(whole, pcmClip)
	want, wantErr = referenceEstimate(d, whole)
	e.Reset()
	for rest := pcmClip; len(rest) > 0; {
		k := copy(ibuf, rest)
		e.AddInt16(ibuf[:k])
		clear(ibuf)
		rest = rest[k:]
	}
	got, gotErr = e.Profile()
	sameProfile(t, got, gotErr, want, wantErr)
}

func TestNoiseProfileEstimatorEmptyChunks(t *testing.T) {
	c := findCase(t, "planted-quiet-window")
	d := mustDenoiser(t, c.cfg)
	want, wantErr := referenceEstimate(d, c.x)
	e := d.NewNoiseProfileEstimator()
	e.AddFloat32(nil)
	e.AddInt16(nil)
	for rest := c.x; len(rest) > 0; {
		k := min(5000, len(rest))
		e.AddFloat32(rest[:0])
		e.AddInt16([]int16{})
		e.AddFloat32(rest[:k])
		e.AddFloat32(nil)
		rest = rest[k:]
	}
	if e.Len() != len(c.x) {
		t.Fatalf("Len = %d, want %d", e.Len(), len(c.x))
	}
	got, gotErr := e.Profile()
	sameProfile(t, got, gotErr, want, wantErr)
}

func TestNoiseProfileEstimatorProfileUsableBySetNoiseProfile(t *testing.T) {
	c := findCase(t, "planted-quiet-window")
	d := mustDenoiser(t, c.cfg)
	e := d.NewNoiseProfileEstimator()
	e.AddFloat32(c.x)
	p, err := e.Profile()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetNoiseProfile(p); err != nil {
		t.Fatalf("SetNoiseProfile on the same Denoiser: %v", err)
	}
	other := mustDenoiser(t, Config{SampleRate: 48000, FrameSize: 2048})
	if err := other.SetNoiseProfile(p); !errors.Is(err, ErrProfileMismatch) {
		t.Fatalf("SetNoiseProfile on another FrameSize: err = %v, want ErrProfileMismatch", err)
	}
}
