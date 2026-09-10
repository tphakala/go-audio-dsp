package denoiser

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

// testTone returns n samples of a deterministic multi-sine signal in [-1, 1].
func testTone(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		t := float64(i)
		x[i] = float32(0.4*math.Sin(0.05*t) + 0.3*math.Cos(0.31*t+1) + 0.2*math.Sin(1.7*t))
	}
	return x
}

func unityConfig(sr, frame, hop int) Config {
	p := ParamsFor(Medium)
	p.MaxAttenuationDB = 0 // unity gain: the stream is a pure analysis/synthesis round trip
	return Config{SampleRate: sr, FrameSize: frame, HopSize: hop, Params: &p}
}

// runStream pushes x through d in the given chunk sizes (cycling) and flushes.
func runStream(t *testing.T, d *Denoiser, x []float32, chunks []int) []float32 {
	t.Helper()
	var out []float32
	for i, c := 0, 0; i < len(x); c++ {
		n := min(chunks[c%len(chunks)], len(x)-i)
		y, err := d.Process(x[i : i+n])
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, y...)
		i += n
	}
	tail, err := d.Flush()
	if err != nil {
		t.Fatal(err)
	}
	return append(out, tail...)
}

func TestLatency(t *testing.T) {
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	if d.Latency() != 1024-256 {
		t.Fatalf("Latency = %d, want 768", d.Latency())
	}
}

func TestIdentityPassthrough(t *testing.T) {
	for _, fh := range [][2]int{{1024, 256}, {512, 128}, {256, 128}, {64, 16}, {128, 1}} {
		for _, n := range []int{0, 1, 100, fh[0] - 1, fh[0], fh[0] + 1, 3*fh[0] + 7, 48000} {
			x := testTone(n)
			d, err := New(unityConfig(48000, fh[0], fh[1]))
			if err != nil {
				t.Fatal(err)
			}
			y := runStream(t, d, x, []int{n + 1})
			if len(y) != len(x) {
				t.Fatalf("frame %d hop %d n %d: got %d samples", fh[0], fh[1], n, len(y))
			}
			for i := range x {
				if math.IsNaN(float64(y[i])) || math.IsInf(float64(y[i]), 0) {
					t.Fatalf("frame %d hop %d n %d: sample %d is non-finite (%g)", fh[0], fh[1], n, i, y[i])
				}
				if math.Abs(float64(y[i]-x[i])) > 1e-4 {
					t.Fatalf("frame %d hop %d n %d: sample %d = %g, want %g", fh[0], fh[1], n, i, y[i], x[i])
				}
			}
		}
	}
}

func TestDefaultParamsNegligibleNoiseProfileIsIdentity(t *testing.T) {
	// With a fixed near-silent noise profile, every bin with real energy sees a
	// huge SNR and unity gain, so the stream passes through. (The default with
	// no profile is an adaptive tracker, which treats a stationary multi-sine as
	// noise and attenuates it; that behaviour is covered elsewhere.)
	x := testTone(20000)
	d, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	p, err := d.NoiseProfileFromSamples(whiteNoise(4096, 1e-6, 1)) // -120 dBFS hiss
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetNoiseProfile(p); err != nil {
		t.Fatal(err)
	}
	y := runStream(t, d, x, []int{1000})
	for i := range x {
		if math.Abs(float64(y[i]-x[i])) > 1e-3 {
			t.Fatalf("sample %d = %g, want %g", i, y[i], x[i])
		}
	}
}

func TestStreamingEquivalence(t *testing.T) {
	x := testTone(30011)
	one, err := New(Config{SampleRate: 48000})
	if err != nil {
		t.Fatal(err)
	}
	ref := runStream(t, one, x, []int{len(x)})
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 5 {
		chunks := []int{1, 7, 255, 256, 257, 1023, 1024, 1025, 5000}
		rng.Shuffle(len(chunks), func(i, j int) { chunks[i], chunks[j] = chunks[j], chunks[i] })
		d, _ := New(Config{SampleRate: 48000})
		got := runStream(t, d, x, chunks)
		if len(got) != len(ref) {
			t.Fatalf("trial %d: %d samples, want %d", trial, len(got), len(ref))
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("trial %d: sample %d differs (%g vs %g) with chunks %v", trial, i, got[i], ref[i], chunks)
			}
		}
	}
}

func TestProcessIntoBufferTooSmall(t *testing.T) {
	x := testTone(5000)
	d, _ := New(Config{SampleRate: 48000})
	ref, _ := New(Config{SampleRate: 48000})
	want, _ := ref.Process(x)
	if _, err := d.ProcessInto(x, make([]float32, 10)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("err = %v, want ErrBufferTooSmall", err)
	}
	// Nothing was consumed: the next correctly sized call matches a fresh run.
	out := make([]float32, len(x)+256)
	n, err := d.ProcessInto(x, out)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(want) {
		t.Fatalf("wrote %d, want %d", n, len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("sample %d differs after a rejected call", i)
		}
	}
	// Flush sizing: FrameSize always suffices, shorter may not.
	if _, err := d.FlushInto(make([]float32, 1)); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("FlushInto err = %v, want ErrBufferTooSmall", err)
	}
	if _, err := d.FlushInto(make([]float32, 1024)); err != nil {
		t.Fatal(err)
	}
}

func TestFlushAndReuse(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	if tail, err := d.Flush(); err != nil || len(tail) != 0 {
		t.Fatalf("fresh Flush = %d samples, %v", len(tail), err)
	}
	x := testTone(12345)
	a := runStream(t, d, x, []int{4000})
	b := runStream(t, d, x, []int{4000}) // same Denoiser, second stream
	if len(a) != len(x) || len(b) != len(x) {
		t.Fatalf("lengths %d %d, want %d", len(a), len(b), len(x))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("second stream differs at %d", i)
		}
	}
	// Reset mid-stream restarts cleanly.
	if _, err := d.Process(x[:3000]); err != nil {
		t.Fatal(err)
	}
	d.Reset()
	c := runStream(t, d, x, []int{4000})
	for i := range a {
		if a[i] != c[i] {
			t.Fatalf("stream after Reset differs at %d", i)
		}
	}
}

func TestNaNChunkDoesNotPoisonStream(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	x := testTone(3000)
	bad := make([]float32, 1000)
	bad[500] = float32(math.NaN())
	for _, chunk := range [][]float32{x, bad, x, x} {
		if _, err := d.Process(chunk); err != nil {
			t.Fatal(err)
		}
	}
	// Frames overlapping the NaN sample are NaN by necessity; everything after
	// the overlap span (Latency() samples past the NaN) must be finite again.
	// Feed more clean input and check the tail.
	out, err := d.Process(testTone(8000))
	if err != nil {
		t.Fatal(err)
	}
	tail := out[len(out)-2000:]
	for i, v := range tail {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite output %g at tail index %d after a NaN chunk", v, i)
		}
	}
}

func TestProcessIntoZeroAlloc(t *testing.T) {
	d, _ := New(Config{SampleRate: 48000})
	x := testTone(4096)
	out := make([]float32, len(x)+256)
	if _, err := d.ProcessInto(x, out); err != nil { // warm up
		t.Fatal(err)
	}
	if a := testing.AllocsPerRun(20, func() {
		if _, err := d.ProcessInto(x, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("ProcessInto allocated %v per run, want 0", a)
	}
	if a := testing.AllocsPerRun(5, func() {
		if _, err := d.FlushInto(out); err != nil {
			t.Fatal(err)
		}
		if _, err := d.ProcessInto(x, out); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("FlushInto allocated %v per run, want 0", a)
	}
}

func TestDenoiseMatchesManualSequence(t *testing.T) {
	const sr = 48000
	x := whiteNoise(6*sr, 0.01, 40)
	toneAt(x, sr, 2*sr, 3*sr, 2000, 0.2)
	cfg := Config{SampleRate: sr, Strength: Heavy}
	got, err := Denoise(x, cfg)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := New(cfg)
	p, err := d.EstimateNoiseProfile(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetNoiseProfile(p); err != nil {
		t.Fatal(err)
	}
	want := runStream(t, d, x, []int{len(x)})
	if len(got) != len(x) || len(want) != len(x) {
		t.Fatalf("lengths %d %d, want %d", len(got), len(want), len(x))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Denoise differs from the manual sequence at %d", i)
		}
	}
	// Tier 2 variant.
	noise := x[:sr]
	got2, err := DenoiseWithNoise(x, noise, cfg)
	if err != nil {
		t.Fatal(err)
	}
	d2, _ := New(cfg)
	p2, _ := d2.NoiseProfileFromSamples(noise)
	_ = d2.SetNoiseProfile(p2)
	want2 := runStream(t, d2, x, []int{len(x)})
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Fatalf("DenoiseWithNoise differs from the manual sequence at %d", i)
		}
	}
}

func TestDenoiseFallbacksAndErrors(t *testing.T) {
	const sr = 48000
	// Featureless noise has no distinct quiet region: Denoise falls back to
	// adaptive tracking and still returns a full-length result.
	x := whiteNoise(3*sr, 0.01, 41)
	y, err := Denoise(x, Config{SampleRate: sr})
	if err != nil || len(y) != len(x) {
		t.Fatalf("fallback: len %d err %v", len(y), err)
	}
	// Tiny input (shorter than a frame) works too.
	y, err = Denoise(x[:100], Config{SampleRate: sr})
	if err != nil || len(y) != 100 {
		t.Fatalf("tiny input: len %d err %v", len(y), err)
	}
	if _, err := Denoise(x, Config{}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("invalid config err = %v", err)
	}
	if _, err := DenoiseWithNoise(x, x[:10], Config{SampleRate: sr}); !errors.Is(err, ErrProfileTooShort) {
		t.Errorf("short noise err = %v", err)
	}
	// DenoiseWithProfile: same result as the Tier 2 path with the same profile,
	// mismatch rejected, nil means adaptive.
	d, _ := New(Config{SampleRate: sr})
	p, _ := d.NoiseProfileFromSamples(x[:sr])
	a, err := DenoiseWithProfile(x, p, Config{SampleRate: sr})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := DenoiseWithNoise(x, x[:sr], Config{SampleRate: sr})
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("DenoiseWithProfile differs from DenoiseWithNoise at %d", i)
		}
	}
	if _, err := DenoiseWithProfile(x, p, Config{SampleRate: sr, FrameSize: 512}); !errors.Is(err, ErrProfileMismatch) {
		t.Errorf("mismatched profile err = %v", err)
	}
	if y, err := DenoiseWithProfile(x, nil, Config{SampleRate: sr}); err != nil || len(y) != len(x) {
		t.Errorf("nil profile (adaptive): len %d err %v", len(y), err)
	}
}
