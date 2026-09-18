package gate

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	dsp "github.com/tphakala/go-audio-dsp"
	"github.com/tphakala/go-audio-dsp/denoiser"
	"github.com/tphakala/go-audio-dsp/stft"
)

// --- test helpers ---

func dbToLin(db float64) float64 { return math.Pow(10, db/20) }

func freqToBin(f float64, sr, n int) int { return int(math.Round(f * float64(n) / float64(sr))) }

// whiteNoise returns n Gaussian samples scaled to the target linear RMS.
func whiteNoise(n int, rms float64, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	if cur := math.Sqrt(s / float64(max(n, 1))); cur > 0 {
		g := float32(rms / cur)
		for i := range x {
			x[i] *= g
		}
	}
	return x
}

// tone returns n samples of a sine at freqHz with the given linear amplitude.
func tone(n, sr int, freqHz, amp float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(amp * math.Sin(2*math.Pi*freqHz*float64(i)/float64(sr)))
	}
	return x
}

// spanRMSDB pools RMS level (power dB) over the given sample spans, returning the
// -200 sentinel for a silent span.
func spanRMSDB(x []float32, spans [][2]int) float64 {
	var s float64
	var cnt int
	for _, sp := range spans {
		for _, v := range x[sp[0]:sp[1]] {
			s += float64(v) * float64(v)
		}
		cnt += sp[1] - sp[0]
	}
	if cnt == 0 || s == 0 {
		return -200
	}
	return 10 * math.Log10(s/float64(cnt))
}

// assertFinite fails if x holds a NaN or Inf. Every dB/RMS metric here propagates
// a non-finite sample (NaN), and an ordered comparison against NaN is false, so a
// regression that emits non-finite audio would pass a bar vacuously; assert
// finiteness before measuring (the go-audio-dsp #8 escaped-defect pattern).
func assertFinite(t *testing.T, name string, x []float32) {
	t.Helper()
	for i, v := range x {
		if f := float64(v); math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("%s: non-finite sample at index %d (%v)", name, i, v)
		}
	}
}

// streamAll runs x through g and flushes, returning one exact-length result.
func streamAll(t *testing.T, g *Gate, x []float32) []float32 {
	t.Helper()
	out := make([]float32, len(x))
	n, err := g.ProcessInto(x, out)
	if err != nil {
		t.Fatalf("ProcessInto: %v", err)
	}
	m, err := g.FlushInto(out[n:])
	if err != nil {
		t.Fatalf("FlushInto: %v", err)
	}
	return out[:n+m]
}

// runStream feeds x through a fresh Gate in the given repeating chunk sizes and
// returns the concatenated Process + Flush output. A non-nil learn excerpt selects
// the learned path; nil leaves the blind tracker active.
func runStream(t *testing.T, cfg Config, learn, x []float32, chunks []int) []float32 {
	t.Helper()
	g, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if learn != nil {
		if err := g.LearnNoise(learn); err != nil {
			t.Fatalf("LearnNoise: %v", err)
		}
	}
	out := make([]float32, 0, len(x)+g.FrameSize())
	buf := make([]float32, len(x)+g.HopSize())
	i, ci := 0, 0
	for i < len(x) {
		c := min(chunks[ci%len(chunks)], len(x)-i)
		ci++
		n, err := g.ProcessInto(x[i:i+c], buf[:g.MaxOutputLen(c)])
		if err != nil {
			t.Fatalf("ProcessInto: %v", err)
		}
		out = append(out, buf[:n]...)
		i += c
	}
	fb := make([]float32, g.Latency()+g.HopSize())
	m, err := g.FlushInto(fb)
	if err != nil {
		t.Fatalf("FlushInto: %v", err)
	}
	return append(out, fb[:m]...)
}

// --- framing tests ---

// TestUnityFloorIsIdentity pins that MaxAttenuationDB == 0 makes the gate an exact
// pass-through (gFloor == 1 forces every bin gain to unity). Smoothing is left on
// so the identity flows through the smoothing passes too. Deleting the residual-
// floor affine (mask.go step 6) or the invNorm divide in finishBlock breaks it.
func TestUnityFloorIsIdentity(t *testing.T) {
	base := ParamsFor(denoiser.Medium)
	base.MaxAttenuationDB = 0
	// {256,128} is the 2x-overlap (ovl=2) minimum, the edge of the warm=ovl-1+L
	// accounting; {64,1} is heavy overlap; {1024,256} the default.
	for _, cc := range []struct{ frame, hop int }{{1024, 256}, {256, 64}, {256, 128}, {64, 1}} {
		for _, n := range []int{0, 1, cc.frame - 1, cc.frame, cc.frame + 1, 3*cc.frame + 7, 20000} {
			p := base
			cfg := Config{SampleRate: 48000, FrameSize: cc.frame, HopSize: cc.hop, Params: &p}
			x := whiteNoise(n, 0.1, uint64(n)+1)
			out, err := Denoise(x, cfg)
			if err != nil {
				t.Fatalf("frame=%d hop=%d n=%d: %v", cc.frame, cc.hop, n, err)
			}
			if len(out) != len(x) {
				t.Fatalf("frame=%d hop=%d n=%d: got %d samples, want %d", cc.frame, cc.hop, n, len(out), len(x))
			}
			var maxErr float64
			for i := range x {
				// A non-finite output makes math.Max(_, NaN)=NaN and maxErr>1e-4
				// false (vacuous pass); fail explicitly instead.
				if v := out[i]; math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("frame=%d hop=%d n=%d: non-finite output at %d (%v)", cc.frame, cc.hop, n, i, v)
				}
				maxErr = math.Max(maxErr, math.Abs(float64(out[i]-x[i])))
			}
			if maxErr > 1e-4 {
				t.Errorf("frame=%d hop=%d n=%d: identity max error %g exceeds 1e-4", cc.frame, cc.hop, n, maxErr)
			}
		}
	}
}

// TestLatencyIncludesLookahead pins Latency() == (FrameSize-HopSize) + L*HopSize.
// Deleting the L*hop lookahead term turns it red.
func TestLatencyIncludesLookahead(t *testing.T) {
	const frame, hop = 1024, 256
	for _, tc := range []struct{ tsf, wantL int }{{1, 0}, {3, 1}, {5, 2}} {
		p := ParamsFor(denoiser.Medium)
		p.TimeSmoothFrames = tc.tsf
		g, err := New(Config{SampleRate: 48000, FrameSize: frame, HopSize: hop, Params: &p})
		if err != nil {
			t.Fatal(err)
		}
		if want, got := (frame-hop)+tc.wantL*hop, g.Latency(); got != want {
			t.Errorf("TimeSmoothFrames=%d: Latency()=%d, want %d", tc.tsf, got, want)
		}
	}
}

// TestFlushDrainsExactly pins that Process + Flush emit exactly len(in) samples for
// a range of lengths and lookaheads, and that a buffer of Latency()+HopSize()
// always suffices for FlushInto. Deleting the warm = ovl-1+L accounting drops or
// duplicates samples so the totals stop matching.
func TestFlushDrainsExactly(t *testing.T) {
	const frame, hop = 256, 64
	for _, tsf := range []int{1, 3, 5} { // L = 0, 1, 2
		for _, n := range []int{0, 1, frame - 1, frame, frame + 1, 3*frame + 7, 20000} {
			p := ParamsFor(denoiser.Medium)
			p.TimeSmoothFrames = tsf
			g, err := New(Config{SampleRate: 48000, FrameSize: frame, HopSize: hop, Params: &p})
			if err != nil {
				t.Fatal(err)
			}
			x := whiteNoise(n, 0.1, uint64(n*7+tsf))
			out, err := g.Process(x)
			if err != nil {
				t.Fatalf("tsf=%d n=%d process: %v", tsf, n, err)
			}
			fbuf := make([]float32, g.Latency()+g.HopSize())
			m, err := g.FlushInto(fbuf)
			if err != nil {
				t.Fatalf("tsf=%d n=%d flush (buf %d): %v", tsf, n, len(fbuf), err)
			}
			if got := len(out) + m; got != n {
				t.Errorf("tsf=%d n=%d: Process+Flush emitted %d, want %d", tsf, n, got, n)
			}
		}
	}
}

// TestStreamChunkInvarianceBitExact pins that the output does not depend on how the
// input was split across ProcessInto calls, on both the blind and learned paths.
// A chunk-dependent smoother (e.g. a running sum instead of the per-frame ring
// recompute) would diverge.
func TestStreamChunkInvarianceBitExact(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	x := whiteNoise(20000, dbToLin(-30), 99)
	tn := tone(len(x), sr, 3000, dbToLin(-18))
	for i := range x {
		x[i] += tn[i]
	}
	noise := whiteNoise(frame*8, dbToLin(-30), 5)
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}
	shuffled := []int{1, 7, 100, 999, 64, 256, 3000, 5, 128}
	for _, learned := range []bool{false, true} {
		var noiseArg []float32
		if learned {
			noiseArg = noise
		}
		ref := runStream(t, cfg, noiseArg, x, []int{len(x)})
		got := runStream(t, cfg, noiseArg, x, shuffled)
		if len(got) != len(ref) {
			t.Fatalf("learned=%v: chunked length %d != single-shot %d", learned, len(got), len(ref))
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("learned=%v: chunked output differs at %d: %g vs %g", learned, i, got[i], ref[i])
			}
		}
	}
}

// TestDenoiseWithNoiseMatchesManualSequence pins that DenoiseWithNoise is exactly
// New + LearnNoise + ProcessInto + FlushInto. Dropping the LearnNoise call inside
// DenoiseWithNoise would leave it on the blind path and diverge.
func TestDenoiseWithNoiseMatchesManualSequence(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	x := whiteNoise(15000, dbToLin(-30), 11)
	tn := tone(len(x), sr, 3000, dbToLin(-18))
	for i := range x {
		x[i] += tn[i]
	}
	noise := whiteNoise(frame*8, dbToLin(-30), 3)
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}

	auto, err := DenoiseWithNoise(x, noise, cfg)
	if err != nil {
		t.Fatal(err)
	}
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.LearnNoise(noise); err != nil {
		t.Fatal(err)
	}
	manual := streamAll(t, g, x)
	if len(auto) != len(manual) {
		t.Fatalf("length %d != %d", len(auto), len(manual))
	}
	for i := range auto {
		if auto[i] != manual[i] {
			t.Fatalf("DenoiseWithNoise differs from manual sequence at %d: %g vs %g", i, auto[i], manual[i])
		}
	}
}

// TestLearnNoiseMatchesSetNoiseFloor pins that LearnNoise (discovered through a
// dsp.Processor as denoiser.NoiseLearner) equals SetNoiseFloor of the same
// unfloored mean power, so both name the identical learned floor. Dropping the
// learned = true / noise = noiseBuf repoint in LearnNoise would run the blind path.
func TestLearnNoiseMatchesSetNoiseFloor(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	noise := whiteNoise(frame*8, dbToLin(-30), 7)
	x := whiteNoise(12000, dbToLin(-30), 8)
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}

	gA, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var proc dsp.Processor = gA
	nl, ok := proc.(denoiser.NoiseLearner)
	if !ok {
		t.Fatal("Gate does not satisfy denoiser.NoiseLearner through a dsp.Processor value")
	}
	if err := nl.LearnNoise(noise); err != nil {
		t.Fatal(err)
	}
	outA := streamAll(t, gA, x)

	plan, err := stft.New(stft.Config{FrameSize: frame, HopSize: hop, Window: stft.Hann})
	if err != nil {
		t.Fatal(err)
	}
	mp := make([]float32, plan.NumBins())
	plan.MeanPowerInto(mp, noise)
	gB, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := gB.SetNoiseFloor(mp); err != nil {
		t.Fatal(err)
	}
	outB := streamAll(t, gB, x)

	if len(outA) != len(outB) {
		t.Fatalf("length %d != %d", len(outA), len(outB))
	}
	for i := range outA {
		if outA[i] != outB[i] {
			t.Fatalf("LearnNoise vs SetNoiseFloor(MeanPowerInto) differ at %d: %g vs %g", i, outA[i], outB[i])
		}
	}
}

// TestAutoFrameSizeMatchesFlagship pins the duplicated auto-frame rule to the
// flagship denoiser's, so a given rate frames identically on either method.
func TestAutoFrameSizeMatchesFlagship(t *testing.T) {
	for _, sr := range []int{16000, 22050, 32000, 44100, 48000} {
		g, err := New(Config{SampleRate: sr})
		if err != nil {
			t.Fatal(err)
		}
		d, err := denoiser.New(denoiser.Config{SampleRate: sr})
		if err != nil {
			t.Fatal(err)
		}
		if g.FrameSize() != d.FrameSize() {
			t.Errorf("sr=%d: gate auto FrameSize %d != flagship %d", sr, g.FrameSize(), d.FrameSize())
		}
	}
}

// TestDenoiseWithFloorMatchesManualSequence pins that DenoiseWithFloor is exactly
// New + SetNoiseFloor + ProcessInto + FlushInto. Dropping the SetNoiseFloor call
// inside DenoiseWithFloor would leave it on the blind path and diverge.
func TestDenoiseWithFloorMatchesManualSequence(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	x := whiteNoise(12000, dbToLin(-30), 13)
	tn := tone(len(x), sr, 3000, dbToLin(-18))
	for i := range x {
		x[i] += tn[i]
	}
	plan, err := stft.New(stft.Config{FrameSize: frame, HopSize: hop, Window: stft.Hann})
	if err != nil {
		t.Fatal(err)
	}
	floor := make([]float32, plan.NumBins())
	plan.MeanPowerInto(floor, whiteNoise(frame*8, dbToLin(-30), 4))
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}

	auto, err := DenoiseWithFloor(x, floor, cfg)
	if err != nil {
		t.Fatal(err)
	}
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetNoiseFloor(floor); err != nil {
		t.Fatal(err)
	}
	manual := streamAll(t, g, x)
	if len(auto) != len(manual) {
		t.Fatalf("length %d != %d", len(auto), len(manual))
	}
	for i := range auto {
		if auto[i] != manual[i] {
			t.Fatalf("DenoiseWithFloor differs from manual sequence at %d: %g vs %g", i, auto[i], manual[i])
		}
	}
	assertFinite(t, "DenoiseWithFloor output", auto)
}

// TestBufferTooSmallConsumesNothing pins the ErrBufferTooSmall contract for both
// ProcessInto and FlushInto: a too-small out returns ErrBufferTooSmall, writes 0,
// and consumes nothing, so a correctly-sized retry emits the full expected count.
func TestBufferTooSmallConsumesNothing(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	cfg := Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium}
	in := whiteNoise(6000, 0.1, 21)

	// A fresh probe gate reports how many samples the first chunk emits.
	probe, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	full := make([]float32, probe.MaxOutputLen(len(in)))
	want, err := probe.ProcessInto(in, full)
	if err != nil {
		t.Fatal(err)
	}
	if want == 0 {
		t.Skip("first chunk emits nothing; cannot exercise the too-small path")
	}

	// ProcessInto with a buffer one short of the pending count.
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := g.ProcessInto(in, make([]float32, want-1)); !errors.Is(err, ErrBufferTooSmall) || n != 0 {
		t.Fatalf("ProcessInto too-small: (n=%d, err=%v), want (0, ErrBufferTooSmall)", n, err)
	}
	// Nothing consumed: a correctly-sized retry emits exactly the expected count.
	retry := make([]float32, g.MaxOutputLen(len(in)))
	if n, err := g.ProcessInto(in, retry); err != nil || n != want {
		t.Fatalf("retry after ErrBufferTooSmall emitted (n=%d, err=%v), want (%d, nil); input was consumed on the failed call", n, err, want)
	}

	// FlushInto with a buffer one short of the remaining tail.
	g2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	proc, err := g2.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	remaining := len(in) - len(proc)
	if remaining == 0 {
		return // whole stream already emitted; nothing to flush
	}
	if n, err := g2.FlushInto(make([]float32, remaining-1)); !errors.Is(err, ErrBufferTooSmall) || n != 0 {
		t.Fatalf("FlushInto too-small: (n=%d, err=%v), want (0, ErrBufferTooSmall)", n, err)
	}
	if n, err := g2.FlushInto(make([]float32, remaining)); err != nil || n != remaining {
		t.Fatalf("FlushInto retry emitted (n=%d, err=%v), want (%d, nil); the failed flush consumed state", n, err, remaining)
	}
}

// TestFlushAllocatingReturnsTail covers the allocating Process/Flush variants:
// together they emit exactly len(in) finite samples.
func TestFlushAllocatingReturnsTail(t *testing.T) {
	const sr, frame, hop = 48000, 256, 64
	g, err := New(Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Strength: denoiser.Medium})
	if err != nil {
		t.Fatal(err)
	}
	in := whiteNoise(5000, 0.1, 71)
	proc, err := g.Process(in)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := g.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if got := len(proc) + len(tail); got != len(in) {
		t.Fatalf("Process+Flush allocating variants emitted %d, want %d", got, len(in))
	}
	assertFinite(t, "Flush tail", tail)
}
