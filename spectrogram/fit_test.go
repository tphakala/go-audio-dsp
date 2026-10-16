package spectrogram

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

var (
	hannShape  = stft.WindowShape{Family: stft.ShapeHann}
	dolph106p5 = stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: 106.5}
)

func dolphShape(drDB float64) stft.WindowShape {
	return stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: DolphAttenuationDB(drDB, 0, 0)}
}

func newTestFit(tb testing.TB, rate, n, columns int, shape stft.WindowShape) *Fit {
	tb.Helper()
	f, err := NewFit(FitConfig{SampleRate: rate, FrameSize: n, Columns: columns, Window: shape})
	if err != nil {
		tb.Fatal(err)
	}
	return f
}

func TestDolphAttenuationDB(t *testing.T) {
	for _, c := range []struct{ dr, gain, adj, want float64 }{
		{80, 0, 0, 86.4},
		{100, 0, 0, 106.5},
		{120, 0, 0, 126.6},
		{20, 0, 0, 26.1},
		{100, 3, 5, 103*1.105 + 6},
	} {
		if got := DolphAttenuationDB(c.dr, c.gain, c.adj); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("DolphAttenuationDB(%g, %g, %g) = %.15g, want %.15g", c.dr, c.gain, c.adj, got, c.want)
		}
	}
}

// TestFitGeometry pins the step and block rule at 24 kHz over a 15 s span against
// the values sox 14.4.2 prints for the same settings (-V4, "step" and "block").
// The Dolph rows at 20 and 120 dB fail if the window sum is peak normalized or a
// rounding in the rule changes.
func TestFitGeometry(t *testing.T) {
	const span = 15 * 24000
	cases := []struct {
		name         string
		x, n         int
		shape        stft.WindowShape
		hop, blockSt int
	}{
		{"hann sm", 258, 256, hannShape, 127, 11},
		{"hann md", 514, 512, hannShape, 233, 3},
		{"hann lg", 1026, 1024, hannShape, 350, 1},
		{"hann xl", 2050, 2048, hannShape, 175, 1},
		{"dolph sm z80", 258, 256, dolphShape(80), 100, 14},
		{"dolph sm z100", 258, 256, dolphShape(100), 87, 16},
		{"dolph sm z120", 258, 256, dolphShape(120), 82, 17},
		{"dolph sm z20", 258, 256, dolphShape(20), 174, 8},
		// The window sums here have a fractional part of 0.5 or more (139.786 and
		// 116.658), so rounding them to the nearest integer matters.
		{"dolph sm z40", 258, 256, dolphShape(40), 140, 10},
		{"dolph sm z60", 258, 256, dolphShape(60), 116, 12},
		{"dolph lg z80", 1026, 1024, dolphShape(80), 350, 1},
		{"dolph lg z100", 1026, 1024, dolphShape(100), 350, 1},
		{"dolph lg z120", 1026, 1024, dolphShape(120), 175, 2},
	}
	// The columns per second are capped at 5000: 10000 columns in a 1 s span would
	// ask for 2 samples per column; the cap gives 4, and the Hann step of 128 does
	// not apply because 4 is below it.
	capped := newTestFit(t, 24000, 256, 10000, hannShape)
	if err := capped.Reset(24000); err != nil {
		t.Fatal(err)
	}
	if capped.Hop() != 4 || capped.BlockSteps() != 1 {
		t.Errorf("capped geometry %d/%d, want 4/1", capped.Hop(), capped.BlockSteps())
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newTestFit(t, 24000, c.n, c.x, c.shape)
			if err := f.Reset(span); err != nil {
				t.Fatal(err)
			}
			if f.Hop() != c.hop || f.BlockSteps() != c.blockSt {
				t.Errorf("Hop/BlockSteps = %d/%d, want %d/%d", f.Hop(), f.BlockSteps(), c.hop, c.blockSt)
			}
		})
	}
	f := newTestFit(t, 24000, 256, 258, hannShape)
	if err := f.Reset(span); err != nil {
		t.Fatal(err)
	}
	if got, want := f.SecondsSpanned(258), 258.0*127*11/24000; math.Abs(got-want) > 1e-12 {
		t.Errorf("SecondsSpanned(258) = %.12g, want %.12g", got, want)
	}
}

func TestFitAccessors(t *testing.T) {
	f := newTestFit(t, 24000, 1024, 1026, hannShape)
	if f.FrameSize() != 1024 || f.Bins() != 513 {
		t.Errorf("FrameSize/Bins = %d/%d, want 1024/513", f.FrameSize(), f.Bins())
	}
	if got, want := f.BinHz(3), 3*24000.0/1024; got != want {
		t.Errorf("BinHz(3) = %g, want %g", got, want)
	}
	if f.NumColumns(48000) != 0 || f.Hop() != 0 || f.BlockSteps() != 0 {
		t.Errorf("before Reset: NumColumns %d, Hop %d, BlockSteps %d, want 0", f.NumColumns(48000), f.Hop(), f.BlockSteps())
	}
	calls := 0
	f.Feed(make([]float32, 5000), func([]float32, int) { calls++ })
	f.Finish(func([]float32, int) { calls++ })
	if calls != 0 {
		t.Errorf("Feed and Finish before the first Reset called back %d times, want 0", calls)
	}
}

// TestFitLevel checks the amplitude level: a full-scale sinusoid on a bin centre
// reads 0 dB in a column with no padding, at every geometry (the small frame sizes
// average several frames per column).
func TestFitLevel(t *testing.T) {
	for _, shape := range []stft.WindowShape{hannShape, dolph106p5} {
		for _, n := range []int{256, 512, 1024, 2048} {
			const rate, samples = 24000, 48000
			k0 := n / 8
			sig := make([]float32, samples)
			for i := range sig {
				sig[i] = float32(math.Sin(2 * math.Pi * float64(k0) * float64(i) / float64(n)))
			}
			f := newTestFit(t, rate, n, n+2, shape)
			m, err := f.Compute(sig, 0)
			if err != nil {
				t.Fatal(err)
			}
			if m.Frames < 8 {
				t.Fatalf("%s N=%d: %d columns, want at least 8", shape.Family, n, m.Frames)
			}
			col := m.Column(m.Frames / 2)
			if got := float64(col[k0]); math.Abs(got) > 0.1 {
				t.Errorf("%s N=%d: mid-clip level at bin %d = %.4f dB, want 0 within 0.1", shape.Family, n, k0, got)
			}
			if peak := slices.Max(col); peak != col[k0] {
				t.Errorf("%s N=%d: peak %.3f dB is not at bin %d (%.3f dB)", shape.Family, n, peak, k0, col[k0])
			}
		}
	}
}

// fitColumns streams signal through f in the given chunk sizes (cycled) and
// returns copies of the emitted columns.
func fitColumns(f *Fit, signal []float32, span int64, chunks []int) ([][]float32, error) {
	if err := f.Reset(span); err != nil {
		return nil, err
	}
	var cols [][]float32
	onColumn := func(col []float32, index int) {
		if index != len(cols) {
			panic("column index out of order")
		}
		cols = append(cols, slices.Clone(col))
	}
	for i, c := 0, 0; i < len(signal); c++ {
		n := min(chunks[c%len(chunks)], len(signal)-i)
		f.Feed(signal[i:i+n], onColumn)
		i += n
	}
	f.Finish(onColumn)
	return cols, nil
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// TestFitStreamEqualsCompute checks that the columns do not depend on how the
// signal is chunked: 20 seeded random chunkings, one-sample chunks and one chunk
// longer than the clip all match ComputeInto bit for bit.
func TestFitStreamEqualsCompute(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	cases := []struct {
		name  string
		n, x  int
		shape stft.WindowShape
		span  int64
	}{
		{"sm dolph partial block", 256, 258, dolphShape(100), 15 * 24000},
		{"xl dolph edge windows", 2048, 2050, dolphShape(100), 15 * 24000},
		{"sm hann truncated", 256, 100, hannShape, 6000},
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newTestFit(t, 24000, c.n, c.x, c.shape)
			var m Matrix
			m.Data = make([]float32, f.Bins()*c.x)
			got, err := f.ComputeInto(&m, clip, c.span)
			if err != nil || got == 0 {
				t.Fatalf("ComputeInto = (%d, %v)", got, err)
			}
			chunkings := make([][]int, 0, 22)
			chunkings = append(chunkings, []int{1}, []int{len(clip) + 1000})
			for range 20 {
				chunks := make([]int, 7)
				for i := range chunks {
					chunks[i] = 1 + rng.IntN(4000)
				}
				chunkings = append(chunkings, chunks)
			}
			for _, chunks := range chunkings {
				cols, err := fitColumns(f, clip, c.span, chunks)
				if err != nil {
					t.Fatal(err)
				}
				if len(cols) != m.Frames {
					t.Fatalf("chunks %v: %d columns, ComputeInto gave %d", chunks[:min(3, len(chunks))], len(cols), m.Frames)
				}
				for i, col := range cols {
					if !sameBits(col, m.Column(i)) {
						t.Fatalf("chunks %v: column %d differs from ComputeInto", chunks[:min(3, len(chunks))], i)
					}
				}
			}
		})
	}
}

// TestFitReuse runs one Fit over clips of different lengths and spans, so the
// window cache sees a new drain length and a new hop in turn, and checks that a
// repeat of the first run is bit-identical to it and to a fresh Fit.
func TestFitReuse(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	short := clip[:16800] // 0.7 s: a different L mod hop
	const longSpan, shortSpan = 15 * 24000, 24000
	shape := dolphShape(100)
	run := func(f *Fit, sig []float32, span int64) []float32 {
		t.Helper()
		m, err := f.Compute(sig, span)
		if err != nil {
			t.Fatal(err)
		}
		return m.Data
	}
	fresh := run(newTestFit(t, 24000, 2048, 2050, shape), clip, longSpan)
	f := newTestFit(t, 24000, 2048, 2050, shape)
	a1 := run(f, clip, longSpan)
	run(f, short, longSpan)  // same hop, new drain length
	run(f, clip, shortSpan)  // new hop (more start-edge windows than the cache holds)
	run(f, short, shortSpan) // new hop and drain length
	// An abandoned stream: a block is half full when the next Reset comes.
	if err := f.Reset(longSpan); err != nil {
		t.Fatal(err)
	}
	f.Feed(clip[:5000], func([]float32, int) {})
	a2 := run(f, clip, longSpan)
	if !sameBits(a1, fresh) {
		t.Error("first run differs from a fresh Fit")
	}
	if !sameBits(a2, a1) {
		t.Error("repeat run after other clips and spans differs from the first run")
	}
}

// TestFitNumColumns checks NumColumns, min(Columns, ceil(frames/BlockSteps)),
// against the number of columns a run emits, and that no run emits more than
// Columns.
func TestFitNumColumns(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	big := make([]float32, 60000)
	for i := range big {
		big[i] = clip[i%len(clip)]
	}
	cases := []struct {
		name  string
		n, x  int
		shape stft.WindowShape
		span  int64
	}{
		{"sm hann blocks", 256, 258, hannShape, 15 * 24000},
		{"sm dolph blocks", 256, 258, dolphShape(100), 15 * 24000},
		{"lg hann hop 350", 1024, 1026, hannShape, 15 * 24000},
		{"sm hann truncated", 256, 100, hannShape, 6000},
		{"lg hann truncated", 1024, 20, hannShape, 24000},
	}
	lengths := []int{0, 1, 2, 60, 127, 128, 129, 174, 175, 176, 255, 256, 300, 349, 350, 351, 524, 525, 1000, 1023, 1024, 1500, 3000, 6000, 6040, 6050, 6097, 6098, 7000, 24000, 60000}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newTestFit(t, 24000, c.n, c.x, c.shape)
			if err := f.Reset(c.span); err != nil {
				t.Fatal(err)
			}
			for _, l := range lengths {
				want := f.NumColumns(int64(l))
				cols, err := fitColumns(f, big[:l], c.span, []int{3000})
				if err != nil {
					t.Fatal(err)
				}
				if len(cols) != want || len(cols) > c.x {
					t.Errorf("L=%d: %d columns emitted, NumColumns %d, limit %d", l, len(cols), want, c.x)
				}
			}
		})
	}
}

// TestFitIgnoresSamplesPastTheKeptColumns checks the useful-sample limit: the
// columns kept at the limit depend on the signal up to a point, and not beyond it.
// A change to one sample before the point changes the columns, a change to the
// first sample at or after it does not, and Feed stops there.
func TestFitIgnoresSamplesPastTheKeptColumns(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	long := make([]float32, 362000)
	for i := range long {
		long[i] = clip[i%len(clip)]
	}
	cases := []struct {
		name  string
		n, x  int
		span  int64
		hop   int
		block int
	}{
		{"even lead", 256, 100, 6000, 60, 1},
		{"odd lead", 256, 258, 15 * 24000, 127, 11},
		{"blocks of 10", 256, 20, 24000, 120, 10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newTestFit(t, 24000, c.n, c.x, hannShape)
			if err := f.Reset(c.span); err != nil {
				t.Fatal(err)
			}
			if f.Hop() != c.hop || f.BlockSteps() != c.block {
				t.Fatalf("geometry %d/%d, want %d/%d", f.Hop(), f.BlockSteps(), c.hop, c.block)
			}
			sig := long
			want, err := fitColumns(f, sig, c.span, []int{1000})
			if err != nil {
				t.Fatal(err)
			}
			if len(want) != c.x {
				t.Fatalf("%d columns, want the limit %d", len(want), c.x)
			}
			if f.consumed != f.useful {
				t.Errorf("Feed accepted %d samples, want to stop at %d", f.consumed, f.useful)
			}
			for _, i := range []int64{f.useful - 2, f.useful - 1, f.useful, f.useful + 1} {
				other := slices.Clone(sig)
				other[i] += 0.5
				got, err := fitColumns(f, other, c.span, []int{1000})
				if err != nil {
					t.Fatal(err)
				}
				same := len(got) == len(want) && slices.EqualFunc(got, want, sameBits)
				if i < f.useful && same {
					t.Errorf("changing sample %d, which a kept frame reads, left the columns unchanged", i)
				}
				if i >= f.useful && !same {
					t.Errorf("changing sample %d, past the last kept frame, changed the columns", i)
				}
			}
		})
	}
}

// TestFitResetDropsPartialBlock checks that a stream abandoned part way through
// a block leaves nothing in the next one.
func TestFitResetDropsPartialBlock(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	const span = 15 * 24000
	fresh, err := newTestFit(t, 24000, 256, 258, dolphShape(100)).Compute(clip, span)
	if err != nil {
		t.Fatal(err)
	}
	f := newTestFit(t, 24000, 256, 258, dolphShape(100))
	if err := f.Reset(span); err != nil {
		t.Fatal(err)
	}
	f.Feed(clip[:3000], func([]float32, int) {}) // two columns and a partial block
	got, err := f.Compute(clip, span)
	if err != nil {
		t.Fatal(err)
	}
	if !sameBits(got.Data, fresh.Data) {
		t.Error("columns after an abandoned stream differ from a fresh Fit")
	}
}

func TestFitConfig(t *testing.T) {
	good := FitConfig{SampleRate: 24000, FrameSize: 256, Columns: 100, Window: hannShape}
	bad := []struct {
		name string
		mod  func(*FitConfig)
	}{
		{"SampleRate 0", func(c *FitConfig) { c.SampleRate = 0 }},
		{"Columns 0", func(c *FitConfig) { c.Columns = 0 }},
		{"GainDB NaN", func(c *FitConfig) { c.GainDB = math.NaN() }},
		{"GainDB Inf", func(c *FitConfig) { c.GainDB = math.Inf(1) }},
		{"FrameSize not a power of two", func(c *FitConfig) { c.FrameSize = 300 }},
		{"Hann with attenuation", func(c *FitConfig) { c.Window.AttenuationDB = 40 }},
		{"Dolph attenuation too low", func(c *FitConfig) { c.Window = stft.WindowShape{Family: stft.ShapeDolphChebyshev, AttenuationDB: 5} }},
		{"matrix size overflow", func(c *FitConfig) { c.Columns = math.MaxInt32 }},
	}
	for _, c := range bad {
		cfg := good
		c.mod(&cfg)
		if _, err := NewFit(cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("%s: NewFit error = %v, want ErrInvalidConfig", c.name, err)
		}
	}
	if _, err := NewFit(good); err != nil {
		t.Fatal(err)
	}

	f := newTestFit(t, 24000, 256, 100, hannShape)
	for _, span := range []int64{0, -1} {
		if err := f.Reset(span); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("Reset(%d) = %v, want ErrInvalidConfig", span, err)
		}
	}
	m := Matrix{Data: make([]float32, 1<<16)}
	if _, err := f.ComputeInto(&m, make([]float32, 100), -5); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("ComputeInto with span -5 = %v, want ErrInvalidConfig", err)
	}
	if _, err := f.Compute(make([]float32, 100), -5); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("Compute with span -5 = %v, want ErrInvalidConfig", err)
	}
	// 10000 columns in one second at 4 kHz asks for less than one sample per column.
	f = newTestFit(t, 4000, 256, 10000, hannShape)
	if err := f.Reset(4000); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("columns per second above the sample rate: Reset = %v, want ErrInvalidConfig", err)
	}
	// A span so long that one column covers more samples than an int32 holds.
	f = newTestFit(t, 24000, 256, 1, hannShape)
	if err := f.Reset(1 << 40); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("span 1<<40: Reset = %v, want ErrInvalidConfig", err)
	}
}

// TestFitFailedResetKeepsTheStream checks that a rejected Reset leaves the
// previous geometry and stream in place.
func TestFitFailedResetKeepsTheStream(t *testing.T) {
	f := newTestFit(t, 24000, 256, 258, hannShape)
	if err := f.Reset(15 * 24000); err != nil {
		t.Fatal(err)
	}
	hop, block := f.Hop(), f.BlockSteps()
	if err := f.Reset(-5); err == nil {
		t.Fatal("Reset(-5) succeeded")
	}
	if f.Hop() != hop || f.BlockSteps() != block {
		t.Errorf("geometry changed to %d/%d after a rejected Reset, want %d/%d", f.Hop(), f.BlockSteps(), hop, block)
	}
}

func TestFitComputeIntoBufferTooSmall(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	f := newTestFit(t, 24000, 256, 258, hannShape)
	need := f.Bins() * 18
	sentinel := float32(-12345)
	data := make([]float32, need-1)
	for i := range data {
		data[i] = sentinel
	}
	m := Matrix{Data: data}
	got, err := f.ComputeInto(&m, clip, 15*24000)
	if !errors.Is(err, ErrBufferTooSmall) || got != 0 {
		t.Fatalf("ComputeInto = (%d, %v), want (0, ErrBufferTooSmall)", got, err)
	}
	for i, v := range data {
		if v != sentinel {
			t.Fatalf("value %d written (%g) on ErrBufferTooSmall", i, v)
		}
	}
	m = Matrix{Data: make([]float32, need)}
	if got, err := f.ComputeInto(&m, clip, 15*24000); err != nil || got != 18 || m.Frames != 18 || m.Bins != f.Bins() || len(m.Data) != need {
		t.Fatalf("ComputeInto = (%d, %v), Frames %d Bins %d len %d, want 18 columns of %d", got, err, m.Frames, m.Bins, len(m.Data), f.Bins())
	}
}

// TestFitComputeIntoDefaultSpan checks span 0: the span is the signal length, or
// one second of samples for an empty signal.
func TestFitComputeIntoDefaultSpan(t *testing.T) {
	clip := soxSignal(t, soxRef{})
	f := newTestFit(t, 24000, 256, 100, hannShape)
	a, err := f.Compute(clip, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, err := f.Compute(clip, int64(len(clip)))
	if err != nil {
		t.Fatal(err)
	}
	if a.Frames == 0 || a.Frames != b.Frames || !sameBits(a.Data, b.Data) {
		t.Errorf("span 0 differs from span len(signal): %d and %d columns", a.Frames, b.Frames)
	}
	e, err := f.Compute(nil, 0)
	if err != nil || e.Frames != 0 {
		t.Errorf("empty signal: (%d columns, %v), want none", e.Frames, err)
	}
	// One second at 24 kHz and 100 columns: 240 samples per column, two steps of 120.
	if f.Hop() != 120 || f.BlockSteps() != 2 {
		t.Errorf("empty signal: geometry %d/%d, want the one-second geometry 120/2", f.Hop(), f.BlockSteps())
	}
}

// TestFitSilenceIsFinite checks the dB floor: silence reads a finite value,
// -200 dB plus the gain, never -Inf or NaN.
func TestFitSilenceIsFinite(t *testing.T) {
	f, err := NewFit(FitConfig{SampleRate: 24000, FrameSize: 256, Columns: 50, Window: hannShape, GainDB: 7})
	if err != nil {
		t.Fatal(err)
	}
	m, err := f.Compute(make([]float32, 24000), 0)
	if err != nil || m.Frames == 0 {
		t.Fatalf("Compute = (%d columns, %v)", m.Frames, err)
	}
	for i, v := range m.Data {
		if math.Abs(float64(v)-(-200+7)) > 1e-3 {
			t.Fatalf("value %d = %g, want -193", i, v)
		}
	}
}
