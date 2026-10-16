package stft

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// The tests below derive the expected frame schedule and windows from the
// behaviour model in the EdgeAnalyzer doc, written out independently of edge.go:
// the padded stream is Z0 zeros, the signal, then Dz zeros, and frame k is
// P[k*S : k*S+N].

// edgeLead returns the lead zero count ceil((n-s)/2).
func edgeLead(n, s int) int { return (n - s + 1) / 2 }

// edgeDrain returns the drain zero count for a signal of l samples.
func edgeDrain(n, s, l int) int {
	r := l % s
	dz := (n - s) / 2
	if r >= s/2 {
		dz += s - r
	}
	return dz
}

// edgePadded builds the padded stream for sig.
func edgePadded(sig []float32, n, s int) []float32 {
	p := make([]float32, 0, edgeLead(n, s)+len(sig)+edgeDrain(n, s, len(sig)))
	p = append(p, make([]float32, edgeLead(n, s))...)
	p = append(p, sig...)
	return append(p, make([]float32, edgeDrain(n, s, len(sig)))...)
}

// edgeFrameCount is the number of frames the padded stream holds.
func edgeFrameCount(n, s, l int) int {
	total := edgeLead(n, s) + l + edgeDrain(n, s, l)
	if total < n {
		return 0
	}
	return (total-n)/s + 1
}

func noErr(tb testing.TB, err error) {
	tb.Helper()
	if err != nil {
		tb.Fatal(err)
	}
}

func isInvalid(tb testing.TB, err error, ctx string) {
	tb.Helper()
	if !errors.Is(err, ErrInvalidConfig) {
		tb.Errorf("%s: err = %v, want it to wrap ErrInvalidConfig", ctx, err)
	}
}

func closeRel(tb testing.TB, got, want, rel float64, ctx string) {
	tb.Helper()
	if math.Abs(got-want) > rel*math.Abs(want) {
		tb.Errorf("%s: got %.9g, want %.9g (relative tolerance %g)", ctx, got, want, rel)
	}
}

func sameInt[T ~int | ~int64 | ~uint64](tb testing.TB, got, want T, ctx string) {
	tb.Helper()
	if got != want {
		tb.Errorf("%s: got %d, want %d", ctx, got, want)
	}
}

func rampSignal(l int) []float32 {
	x := make([]float32, l)
	for i := range x {
		x[i] = float32(i + 1)
	}
	return x
}

func noiseSignal(l int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 99))
	x := make([]float32, l)
	for i := range x {
		x[i] = float32(r.Float64()*2 - 1)
	}
	return x
}

func newEdge(tb testing.TB, n int, shape WindowShape) *EdgeAnalyzer {
	tb.Helper()
	a, err := NewEdgeAnalyzer(EdgeConfig{FrameSize: n, Shape: shape})
	noErr(tb, err)
	return a
}

var (
	hannShape  = WindowShape{Family: ShapeHann}
	dolphShape = func(att float64) WindowShape { return WindowShape{Family: ShapeDolphChebyshev, AttenuationDB: att} }
)

// runEdge feeds sig in one call, finishes, and returns every frame's power.
func runEdge(a *EdgeAnalyzer, hop int, sig []float32) ([][]float32, error) {
	if err := a.Reset(hop); err != nil {
		return nil, err
	}
	var frames [][]float32
	collect := func(power []float32) { frames = append(frames, slices.Clone(power)) }
	a.Feed(sig, collect)
	a.Finish(collect)
	return frames, nil
}

// A wrong lead count, slide or drain length shows up as a frame whose contents
// differ from the padded stream.
func TestEdgeFrameContents(t *testing.T) {
	for _, g := range []struct{ n, s int }{{16, 5}, {16, 7}, {16, 16}, {16, 1}, {256, 127}, {1024, 350}} {
		a := newEdge(t, g.n, hannShape)
		for l := 0; l <= 3*g.n; l++ {
			sig := rampSignal(l)
			p := edgePadded(sig, g.n, g.s)
			noErr(t, a.Reset(g.s))
			k := 0
			check := func([]float32) {
				lo := k * g.s
				if lo+g.n > len(p) {
					t.Fatalf("N=%d S=%d L=%d: frame %d runs past the padded stream (len %d)", g.n, g.s, l, k, len(p))
				}
				if !slices.Equal(a.buf, p[lo:lo+g.n]) {
					t.Fatalf("N=%d S=%d L=%d: frame %d holds %v, want %v", g.n, g.s, l, k, a.buf, p[lo:lo+g.n])
				}
				k++
			}
			a.Feed(sig, check)
			a.Finish(check)
			if want := edgeFrameCount(g.n, g.s, l); k != want {
				t.Fatalf("N=%d S=%d L=%d: %d frames, want %d", g.n, g.s, l, k, want)
			}
		}
	}
}

// The frame count agrees across the emitted count, NumFrames and the closed form
// floor(L/S) + (L mod S >= floor(S/2)).
func TestEdgeNumFrames(t *testing.T) {
	for _, g := range []struct{ n, s int }{{16, 5}, {16, 7}, {16, 16}, {16, 1}, {256, 127}, {1024, 350}} {
		a := newEdge(t, g.n, hannShape)
		for l := 0; l <= 3*g.n; l++ {
			frames, err := runEdge(a, g.s, make([]float32, l))
			noErr(t, err)
			closed := l / g.s
			if l%g.s >= g.s/2 {
				closed++
			}
			if len(frames) != closed || a.NumFrames(int64(l)) != int64(closed) || edgeFrameCount(g.n, g.s, l) != closed {
				t.Fatalf("N=%d S=%d L=%d: emitted %d, NumFrames %d, closed form %d, padded-stream count %d",
					g.n, g.s, l, len(frames), a.NumFrames(int64(l)), closed, edgeFrameCount(g.n, g.s, l))
			}
		}
	}
}

// Widths measured with sox 14.4.2 (Homebrew 14.4.2_6, -V4): the PNG width equals
// the frame count because one frame per column holds at these geometries.
//
//	sox clip.wav -n trim 0 <L>s spectrogram -x 1026 -y 513 -d 15 -r      (N=1024, S=350)
//	sox clip.wav -n trim 0 <L>s spectrogram -x 2050 -y 1025 -d 15 -w dolph -r   (N=2048, S=175)
//
// The short lengths include cases where sox's internal read position is
// negative, so this pins the drain length modulo L mod S.
func TestEdgeNumFramesSoxTable(t *testing.T) {
	tables := []struct {
		n, s int
		want map[int]int
	}{
		{1024, 350, map[int]int{1: 0, 100: 0, 174: 0, 175: 1, 200: 1, 336: 1, 337: 1, 338: 1, 349: 1, 350: 1, 524: 1, 525: 2, 1000: 3, 48000: 137, 72000: 206}},
		{2048, 175, map[int]int{1: 0, 100: 1, 174: 1, 175: 1, 200: 1, 336: 2, 350: 2, 524: 3, 525: 3, 1000: 6, 48000: 274, 72000: 411}},
	}
	for _, tb := range tables {
		a := newEdge(t, tb.n, hannShape)
		noErr(t, a.Reset(tb.s))
		for l, want := range tb.want {
			if got := a.NumFrames(int64(l)); got != int64(want) {
				t.Errorf("N=%d S=%d L=%d: NumFrames = %d, sox width %d", tb.n, tb.s, l, got, want)
			}
			frames, err := runEdge(a, tb.s, make([]float32, l))
			noErr(t, err)
			if len(frames) != want {
				t.Errorf("N=%d S=%d L=%d: emitted %d frames, sox width %d", tb.n, tb.s, l, len(frames), want)
			}
		}
	}
}

// Drain lengths printed by sox 14.4.2_6 (-V4, "flushed end") for (L, N, S).
func TestEdgeDrainLength(t *testing.T) {
	for _, c := range []struct{ l, n, s, want int }{
		{48000, 1024, 350, 337}, {48000, 256, 127, 70}, {48000, 512, 233, 139},
		{175, 1024, 350, 512}, {200, 1024, 350, 487}, {1000, 1024, 350, 387},
		{1000, 2048, 175, 986}, {100, 2048, 175, 1011},
	} {
		a := newEdge(t, c.n, hannShape)
		noErr(t, a.Reset(c.s))
		if got := a.drainLen(int64(c.l)); got != c.want {
			t.Errorf("L=%d N=%d S=%d: drainLen = %d, sox drained %d", c.l, c.n, c.s, got, c.want)
		}
	}
}

// symmetricShape returns the shape_m window from public pieces, in float64
// precision of the float32 generators: Hann from GenerateWindow, Dolph from
// DolphChebyshevSymmetric divided by its centre sample.
func symmetricShape(t *testing.T, s WindowShape, m int) []float64 {
	t.Helper()
	out := make([]float64, m)
	switch s.Family {
	case ShapeHann:
		for i, v := range GenerateWindow(HannSymmetric, m) {
			out[i] = float64(v)
		}
	case ShapeDolphChebyshev:
		w, err := DolphChebyshevSymmetric(m, s.AttenuationDB)
		noErr(t, err)
		c := float64(w[(m-1)/2])
		for i, v := range w {
			out[i] = float64(v) / c
		}
	}
	return out
}

// expectedEdgeWindow builds a frame's full-length window from the model: a drain
// frame (d >= 1 drain zeros) takes shape_m, m = N-d+1, at slot 0 with all m
// points summed; a flow frame with e lead zeros takes shape_m, m = N-e+1, at slot
// e with the first m-1 points summed. The factor is 2/sum * ((m-1)/N)^2.
func expectedEdgeWindow(t *testing.T, s WindowShape, n, e, d int) []float64 {
	t.Helper()
	m, points, at := n-e+1, n-e, e
	if d >= 1 {
		m, points, at = n-d+1, n-d+1, 0
	}
	shape := symmetricShape(t, s, m)
	var sum float64
	for _, v := range shape[:points] {
		sum += v
	}
	ratio := float64(m-1) / float64(n)
	amp := 2 / sum * ratio * ratio
	win := make([]float64, n)
	for i, v := range shape[:points] {
		win[at+i] = v * amp
	}
	return win
}

// windowTol is the relative tolerance between an analyzer window and one built
// from the public float32 generators. The worst deviation measured is 1.05e-7 on
// arm64 NEON, arm64 scalar, amd64 SSE2 and amd64 scalar alike (the float32
// rounding of the reference), so 5e-7 leaves a margin of about 5x.
const windowTol = 5e-7

// requireWindowsClose compares got with want element by element, relative to the
// larger of |want| and a thousandth of the window peak (so near-zero Hann tails
// are not judged on their last bits), and returns the worst deviation seen.
func requireWindowsClose(t *testing.T, got []float32, want []float64, ctx string) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: window length %d, want %d", ctx, len(got), len(want))
	}
	peak := 0.0
	for _, v := range want {
		peak = max(peak, math.Abs(v))
	}
	worst := 0.0
	for i := range want {
		dev := math.Abs(float64(got[i])-want[i]) / max(math.Abs(want[i]), 1e-3*peak)
		worst = max(worst, dev)
		if dev > windowTol {
			t.Fatalf("%s: window[%d] = %g, want %g", ctx, i, got[i], want[i])
		}
	}
	return worst
}

// The window each frame uses matches the model: placement at slot e or slot 0,
// the flow sum of m-1 points against the drain sum of m points, the factor, and
// the centre normalization. Dolph is the family that can see the sum rule: the
// Hann endpoint is 0, so adding it changes nothing.
func TestEdgeWindowRules(t *testing.T) {
	cases := []struct {
		name  string
		shape WindowShape
		n, s  int
	}{
		{"hann-n64", hannShape, 64, 21},
		{"dolph-40dB-n64", dolphShape(40), 64, 21},
		{"dolph-106.5dB-n256", dolphShape(106.5), 256, 127},
		{"dolph-26.1dB-n128-hop1", dolphShape(26.1), 128, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := newEdge(t, c.n, c.shape)
			var lead, drain, full int // frames seen of each kind
			worst := 0.0
			for _, l := range []int{0, 1, c.s - 1, c.s, 2 * c.s, c.n - c.s, c.n, c.n + 5, 3*c.n + 7} {
				z0 := edgeLead(c.n, c.s)
				noErr(t, a.Reset(c.s))
				k := 0
				check := func([]float32) {
					d := k*c.s + c.n - z0 - l
					e := max(0, z0-k*c.s)
					ctx := fmt.Sprintf("L=%d frame %d (e=%d d=%d)", l, k, e, d)
					worst = max(worst, requireWindowsClose(t, a.win, expectedEdgeWindow(t, c.shape, c.n, e, d), ctx))
					switch {
					case d >= 1:
						drain++
					case e > 0:
						lead++
					default:
						full++
					}
					k++
				}
				a.Feed(make([]float32, l), check)
				a.Finish(check)
			}
			if lead == 0 || drain == 0 || full == 0 {
				t.Errorf("fixture covers %d lead, %d drain and %d full-window frames, want all three", lead, drain, full)
			}
			t.Logf("worst window deviation %.3g (tolerance %g)", worst, windowTol)
		})
	}
}

// A clip shorter than a frame has its only frames in the drain, and they still
// hold lead zeros. The model windows them with the drain rule (shape_m at slot
// 0, m = N-d+1), not the lead rule. N=1024, S=350, L=200: the frame has e = 337
// and d = 487, so m = 538.
func TestEdgeShortClipUsesDrainRule(t *testing.T) {
	const n, s, l = 1024, 350, 200
	a := newEdge(t, n, dolphShape(60))
	noErr(t, a.Reset(s))
	frames := 0
	a.Feed(rampSignal(l), func([]float32) { t.Fatal("no frame completes inside Feed") })
	a.Finish(func([]float32) {
		frames++
		if a.win[0] == 0 || a.win[537] == 0 {
			t.Errorf("drain window should span slots 0..537, got first %g last %g", a.win[0], a.win[537])
		}
		for i, v := range a.win[538:] {
			if v != 0 {
				t.Fatalf("window[%d] = %g, want 0 after the drain window", 538+i, v)
			}
		}
		requireWindowsClose(t, a.win, expectedEdgeWindow(t, dolphShape(60), n, 337, 487), "short clip")
	})
	sameInt(t, frames, 1, "frames")
}

// A full-scale cosine centred on a bin reads power 1 (0 dBFS) in a frame with no
// padding, for every window, size and attenuation. Without the amplitude factor
// the level is several tens of dB higher. The measured worst error is 0.0002 dB
// on every kernel tier tried (it is leakage from the cosine's negative-frequency
// image, a property of the window), so the 0.01 dB bound leaves a 50x margin.
func TestEdgeLevel(t *testing.T) {
	const levelTolDB = 0.01
	worst := 0.0
	for _, shape := range []WindowShape{hannShape, dolphShape(86.4), dolphShape(106.5), dolphShape(126.6)} {
		for _, n := range []int{256, 512, 1024, 2048} {
			a := newEdge(t, n, shape)
			k0 := n / 8
			sig := make([]float32, 4*n)
			for i := range sig {
				sig[i] = float32(math.Cos(2 * math.Pi * float64(k0) * float64(i) / float64(n)))
			}
			noErr(t, a.Reset(0))
			z0 := edgeLead(n, a.HopSize())
			checked := 0
			k := 0
			a.Feed(sig, func(power []float32) {
				// Frames with e = 0 hold no lead zeros; the first drain frame comes after Feed.
				if k*a.HopSize() >= z0 {
					db := 10 * math.Log10(float64(power[k0]))
					worst = max(worst, math.Abs(db))
					if math.Abs(db) > levelTolDB {
						t.Errorf("%v N=%d frame %d: bin %d reads %.4f dB, want 0 within %g", shape, n, k, k0, db, levelTolDB)
					}
					checked++
				}
				k++
			})
			if checked == 0 {
				t.Fatalf("%v N=%d: no mid-stream frame checked", shape, n)
			}
		}
	}
	t.Logf("worst |level error| = %.3g dB", worst)
}

// FullWindowSum is the float64 sum of the first N points of the symmetric
// (N+1)-point window. Hann sums to N/2. The Dolph values are sox 14.4.2_6
// window_density (sum/N, printed with 6 significant digits by -V4); the 26.1 dB
// rows only match a centre-normalized window, where a peak-normalized one gives a
// different density.
func TestEdgeFullWindowSum(t *testing.T) {
	for _, n := range []int{64, 256, 1024} {
		a := newEdge(t, n, hannShape)
		closeRel(t, a.FullWindowSum(), float64(n)/2, 1e-9, fmt.Sprintf("Hann N=%d", n))
	}
	table := []struct {
		n    int
		att  float64
		want float64
	}{
		{256, 86.4, 0.39883}, {256, 106.5, 0.359028}, {256, 126.6, 0.329197}, {256, 26.1, 0.717166},
		{1024, 86.4, 0.398724}, {1024, 106.5, 0.358885}, {1024, 126.6, 0.329015}, {1024, 26.1, 0.717514},
	}
	worst := 0.0
	for _, c := range table {
		a := newEdge(t, c.n, dolphShape(c.att))
		got := a.FullWindowSum() / float64(c.n)
		worst = max(worst, math.Abs(got-c.want)/c.want)
		closeRel(t, got, c.want, 5e-6, fmt.Sprintf("Dolph N=%d A=%g density", c.n, c.att))
	}
	t.Logf("worst density deviation from sox %.3g (sox prints 6 significant digits)", worst)
}

// Reset(0) picks round(FullWindowSum) clamped to [1, N].
func TestEdgeDefaultHop(t *testing.T) {
	a := newEdge(t, 256, hannShape)
	sameInt(t, a.HopSize(), 128, "default hop of a fresh Hann N=256")
	b := newEdge(t, 1024, dolphShape(106.5))
	want := int(math.Floor(b.FullWindowSum() + 0.5))
	sameInt(t, b.HopSize(), want, "default hop of a fresh Dolph N=1024")
	noErr(t, b.Reset(60))
	noErr(t, b.Reset(0))
	sameInt(t, b.HopSize(), want, "hop after Reset(0)")
	// Sums of 336.91 and 91.91 round up; truncating them would give 336 and 91.
	sameInt(t, newEdge(t, 1024, dolphShape(126.6)).HopSize(), 337, "default hop at N=1024, 126.6 dB")
	sameInt(t, newEdge(t, 256, dolphShape(106.5)).HopSize(), 92, "default hop at N=256, 106.5 dB")
	// The N=2 Hann window is [0, 1], so the rounded sum is 1.
	sameInt(t, newEdge(t, 2, hannShape).HopSize(), 1, "default hop at N=2")
}

// Frame powers do not depend on how the input is chunked.
func TestEdgeChunkInvariance(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, shape := range []WindowShape{hannShape, dolphShape(86.4)} {
		for _, g := range []struct{ n, s, l int }{{64, 21, 700}, {256, 127, 1500}, {64, 1, 300}, {128, 128, 400}} {
			a := newEdge(t, g.n, shape)
			sig := noiseSignal(g.l, 11)
			want, err := runEdge(a, g.s, sig)
			noErr(t, err)
			for trial := range 8 {
				noErr(t, a.Reset(g.s))
				var got [][]float32
				collect := func(power []float32) { got = append(got, slices.Clone(power)) }
				for rest := sig; len(rest) > 0; {
					c := 1 + r.IntN(2*g.n)
					if trial == 0 {
						c = 1
					}
					c = min(c, len(rest))
					a.Feed(rest[:c], collect)
					rest = rest[c:]
				}
				a.Finish(collect)
				requireFramesEqual(t, want, got, fmt.Sprintf("%v N=%d S=%d trial %d", shape, g.n, g.s, trial))
			}
		}
	}
}

func requireFramesEqual(t *testing.T, want, got [][]float32, ctx string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d frames, want %d", ctx, len(got), len(want))
	}
	for k := range want {
		for i := range want[k] {
			if math.Float32bits(want[k][i]) != math.Float32bits(got[k][i]) {
				t.Fatalf("%s: frame %d bin %d = %g, want %g", ctx, k, i, got[k][i], want[k][i])
			}
		}
	}
}

// A warmed analyzer rebuilds no window for a repeat geometry. N=2048, Dolph
// 106.5, hop 175, a 48000-sample clip. The drain windows depend on the clip
// length modulo the hop, the lead windows only on the hop.
func TestEdgeWindowCacheReuse(t *testing.T) {
	const n, hop = 2048, 175
	a := newEdge(t, n, dolphShape(106.5))
	sig := noiseSignal(48001, 3)

	// run feeds sig[:l], returns the builds it caused and the number of drain frames.
	run := func(h, l int) (builds uint64, drainFrames int, frames [][]float32) {
		before := a.windowBuilds
		noErr(t, a.Reset(h))
		collect := func(p []float32) { frames = append(frames, slices.Clone(p)) }
		a.Feed(sig[:l], collect)
		inFeed := len(frames)
		a.Finish(collect)
		return a.windowBuilds - before, len(frames) - inFeed, frames
	}
	// leadWindows counts frames with lead zeros; drain windows cap at the cache size.
	leadWindows := func(h int) int { return min((edgeLead(n, h)+h-1)/h, edgeCacheWindows) }

	b0, d0, first := run(hop, 48000)
	sameInt(t, b0, uint64(leadWindows(hop)+min(d0, edgeCacheWindows)), "first run builds the lead and drain windows")
	b1, _, second := run(hop, 48000)
	sameInt(t, b1, 0, "same hop and length rebuilds nothing")
	requireFramesEqual(t, first, second, "repeat run")

	b2, d2, _ := run(hop, 48001)
	sameInt(t, b2, uint64(min(d2, edgeCacheWindows)), "a new length modulo the hop rebuilds only the drain windows")

	b3, d3, got := run(174, 48000)
	sameInt(t, b3, uint64(leadWindows(174)+min(d3, edgeCacheWindows)), "a new hop rebuilds the lead and drain windows")
	fresh := newEdge(t, n, dolphShape(106.5))
	want, err := runEdge(fresh, 174, sig[:48000])
	noErr(t, err)
	requireFramesEqual(t, want, got, "hop 174 after hop 175 equals a fresh analyzer")
}

// The drain windows depend on the drain zeros fed when the first drain frame
// fires, which for a clip shorter than a frame is not a function of the clip
// length modulo the hop. N=1024, S=350: L=200 and L=550 share L mod S = 200 but
// fire their first drain frame after 487 and 137 zeros. L=200 and L=190 both emit
// one drain frame, after 487 and 497 zeros, so a key of the frame count alone would
// reuse the wrong window.
func TestEdgeDrainKeyShortClips(t *testing.T) {
	a := newEdge(t, 1024, dolphShape(100))
	sig := noiseSignal(550, 8)
	for _, l := range []int{200, 190, 550, 200, 550} {
		got, err := runEdge(a, 350, sig[:l])
		noErr(t, err)
		want, err := runEdge(newEdge(t, 1024, dolphShape(100)), 350, sig[:l])
		noErr(t, err)
		requireFramesEqual(t, want, got, fmt.Sprintf("L=%d on a reused analyzer", l))
	}
}

// Two geometries can share the first drain length and the drain frame count while
// differing in hop, and then in every later drain frame's zero count. N=64, L=4:
// hop 2 and hop 3 both fire their first drain frame after 29 zeros and emit two.
func TestEdgeDrainKeyHop(t *testing.T) {
	a := newEdge(t, 64, dolphShape(60))
	sig := noiseSignal(4, 12)
	for _, hop := range []int{2, 3, 2, 3} {
		got, err := runEdge(a, hop, sig)
		noErr(t, err)
		want, err := runEdge(newEdge(t, 64, dolphShape(60)), hop, sig)
		noErr(t, err)
		requireFramesEqual(t, want, got, fmt.Sprintf("hop %d on a reused analyzer", hop))
		sameInt(t, len(got), 2, "drain frames")
	}
}

// A window whose summed points are all zero must come out zero, not NaN. At N=2
// and hop 1 the lead frame has one lead zero, so its window is the 2-point Hann
// shape [0, 0] summed over its first point: a zero sum.
func TestEdgeZeroSumWindow(t *testing.T) {
	a := newEdge(t, 2, hannShape)
	frames, err := runEdge(a, 1, []float32{1, 1, 1, 1})
	noErr(t, err)
	if len(frames) == 0 {
		t.Fatal("no frames")
	}
	for k, f := range frames {
		for i, v := range f {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("frame %d bin %d = %g, want finite", k, i, v)
			}
		}
	}
	for i, v := range frames[0] {
		if v != 0 {
			t.Errorf("lead frame bin %d = %g, want 0 (the window is all zero)", i, v)
		}
	}
}

// One reused analyzer, walked through random (hop, length) steps, produces the
// same bits as a fresh analyzer for each step. Small hops exceed the cache cap,
// so the per-frame rebuild buffer is exercised too.
func TestEdgeCacheTransitions(t *testing.T) {
	const n, steps = 512, 40
	for _, shape := range []WindowShape{hannShape, dolphShape(106.5)} {
		r := rand.New(rand.NewPCG(21, 22))
		hops := []int{1, 3, 60, 174, 175, 350, n}
		sig := noiseSignal(3*n, 4)
		reused := newEdge(t, n, shape)
		for step := range steps {
			hop, l := hops[r.IntN(len(hops))], r.IntN(3*n+1)
			got, err := runEdge(reused, hop, sig[:l])
			noErr(t, err)
			want, err := runEdge(newEdge(t, n, shape), hop, sig[:l])
			noErr(t, err)
			requireFramesEqual(t, want, got, fmt.Sprintf("%v step %d hop %d L %d", shape, step, hop, l))
		}
	}
}

func TestEdgeAnalyzerConfig(t *testing.T) {
	bad := []struct {
		name string
		cfg  EdgeConfig
	}{
		{"frame size 0", EdgeConfig{FrameSize: 0}},
		{"frame size 3", EdgeConfig{FrameSize: 3}},
		{"frame size 1", EdgeConfig{FrameSize: 1}},
		{"dolph frame size 65536", EdgeConfig{FrameSize: 65536, Shape: dolphShape(100)}},
		{"hann with attenuation", EdgeConfig{FrameSize: 256, Shape: WindowShape{Family: ShapeHann, AttenuationDB: 5}}},
		{"hann with NaN attenuation", EdgeConfig{FrameSize: 256, Shape: WindowShape{Family: ShapeHann, AttenuationDB: math.NaN()}}},
		{"dolph NaN", EdgeConfig{FrameSize: 256, Shape: dolphShape(math.NaN())}},
		{"dolph 19", EdgeConfig{FrameSize: 256, Shape: dolphShape(19)}},
		{"dolph 201", EdgeConfig{FrameSize: 256, Shape: dolphShape(201)}},
		{"dolph unset attenuation", EdgeConfig{FrameSize: 256, Shape: WindowShape{Family: ShapeDolphChebyshev}}},
		{"unknown family", EdgeConfig{FrameSize: 256, Shape: WindowShape{Family: ShapeFamily(7)}}},
	}
	for _, c := range bad {
		a, err := NewEdgeAnalyzer(c.cfg)
		isInvalid(t, err, c.name)
		if a != nil {
			t.Errorf("%s: got an analyzer with the error", c.name)
		}
	}

	// A rejected Reset leaves the stream exactly as it was.
	a := newEdge(t, 256, dolphShape(86.4))
	sig := noiseSignal(1000, 2)
	noErr(t, a.Reset(100))
	var got [][]float32
	collect := func(p []float32) { got = append(got, slices.Clone(p)) }
	a.Feed(sig[:300], collect)
	for _, h := range []int{-1, 257} {
		isInvalid(t, a.Reset(h), fmt.Sprintf("Reset(%d)", h))
		sameInt(t, a.HopSize(), 100, fmt.Sprintf("hop after Reset(%d)", h))
	}
	a.Feed(sig[300:], collect)
	a.Finish(collect)
	want, err := runEdge(newEdge(t, 256, dolphShape(86.4)), 100, sig)
	noErr(t, err)
	requireFramesEqual(t, want, got, "stream interrupted by rejected Resets")

	for f, want := range map[ShapeFamily]string{ShapeHann: "hann", ShapeDolphChebyshev: "dolph-chebyshev", ShapeFamily(9): "ShapeFamily(9)"} {
		if got := f.String(); got != want {
			t.Errorf("ShapeFamily(%d).String() = %q, want %q", int(f), got, want)
		}
	}
}

func TestEdgeFinishIdempotent(t *testing.T) {
	a := newEdge(t, 128, hannShape)
	sig := noiseSignal(500, 1)
	want, err := runEdge(a, 40, sig)
	noErr(t, err)
	if len(want) == 0 {
		t.Fatal("fixture emits no frames")
	}

	count := 0
	fail := func([]float32) { count++ }
	a.Finish(fail)
	a.Feed(sig, fail)
	a.Finish(fail)
	sameInt(t, count, 0, "frames emitted by Feed and Finish after Finish")

	got, err := runEdge(a, 40, sig)
	noErr(t, err)
	requireFramesEqual(t, want, got, "Reset restores the stream")
}

// Accessors report the construction and Reset state.
func TestEdgeAccessors(t *testing.T) {
	a := newEdge(t, 512, hannShape)
	sameInt(t, a.FrameSize(), 512, "FrameSize")
	sameInt(t, a.NumBins(), 257, "NumBins")
	noErr(t, a.Reset(100))
	sameInt(t, a.HopSize(), 100, "HopSize")
	sameInt(t, a.NumFrames(-5), 0, "NumFrames of a negative length")
}

// dolphCentreInto agrees with dolph64 up to the choice of normalizer: both are
// the same unnormalized window, divided by the peak or by the centre.
func TestDolphCentreIntoMatchesDolph64(t *testing.T) {
	for _, m := range []int{1, 2, 3, 8, 255, 256, 257, 1025} {
		for _, att := range []float64{20, 26.1, 86.4, 126.6, 200} {
			peakNorm, err := dolph64(m, att)
			noErr(t, err)
			w := make([]float64, m)
			noErr(t, dolphCentreInto(w, make([]float64, (m-1)/2+1), make([]float64, 2*m), att))
			if w[(m-1)/2] != 1 {
				t.Fatalf("m=%d A=%g: centre sample = %g, want exactly 1", m, att, w[(m-1)/2])
			}
			ratio := w[0] / peakNorm[0]
			for i := range w {
				if math.Abs(w[i]-ratio*peakNorm[i]) > 1e-12*math.Abs(w[i]) {
					t.Fatalf("m=%d A=%g: w[%d] = %g, peak-normalized %g * %g", m, att, i, w[i], peakNorm[i], ratio)
				}
			}
		}
	}
}

// naivePower returns |DFT|^2 of x*w for bins 0..n/2, in float64.
func naivePower(x, w []float32) []float64 {
	n := len(x)
	out := make([]float64, n/2+1)
	for k := range out {
		var re, im float64
		for i := range x {
			v := float64(x[i]) * float64(w[i])
			ang := -2 * math.Pi * float64((k*i)%n) / float64(n)
			re += v * math.Cos(ang)
			im += v * math.Sin(ang)
		}
		out[k] = re*re + im*im
	}
	return out
}

// dftTol bounds the difference between an analyzer frame power and a float64 DFT
// of the same windowed frame, relative to the frame's strongest bin. Measured
// worst: 3.5e-7 arm64 NEON, 3.6e-7 arm64 scalar, 4.1e-7 amd64 SSE2 and scalar, so
// 5e-6 leaves a margin of about 12x for the float32 transform on any kernel tier.
const dftTol = 5e-6

// Every frame's power agrees with a direct float64 DFT of the frame times the
// window the analyzer used, so the transform, the window placement and the power
// step are checked end to end, on whichever simd kernel tier runs.
func TestEdgePowerMatchesDFT(t *testing.T) {
	worst := 0.0
	for _, shape := range []WindowShape{hannShape, dolphShape(106.5)} {
		for _, g := range []struct{ n, s, l int }{{64, 21, 300}, {256, 127, 700}, {128, 1, 60}} {
			a := newEdge(t, g.n, shape)
			noErr(t, a.Reset(g.s))
			k := 0
			check := func(power []float32) {
				want := naivePower(a.buf, a.win)
				peak := 0.0
				for _, v := range want {
					peak = max(peak, v)
				}
				for i, v := range power {
					dev := math.Abs(float64(v)-want[i]) / max(peak, 1e-30)
					worst = max(worst, dev)
					if dev > dftTol {
						t.Fatalf("%v N=%d S=%d frame %d bin %d: power %g, DFT %g (peak %g)", shape, g.n, g.s, k, i, v, want[i], peak)
					}
				}
				k++
			}
			sig := noiseSignal(g.l, 17)
			a.Feed(sig, check)
			a.Finish(check)
		}
	}
	t.Logf("worst |power - DFT| / frame peak = %.3g (tolerance %g)", worst, dftTol)
}

// The precision guards turn a non-finite evaluation into an error instead of a
// window. A NaN attenuation drives every sample to NaN; the public constructors
// reject it earlier, so this reaches the guards directly.
func TestDolphGuardsRejectNonFinite(t *testing.T) {
	_, err := dolph64(9, math.NaN())
	isInvalid(t, err, "dolph64 with NaN attenuation")
	w := make([]float64, 9)
	isInvalid(t, dolphCentreInto(w, make([]float64, 5), make([]float64, 18), math.NaN()), "dolphCentreInto with NaN attenuation")
}

// A shape that cannot be evaluated yields an all-zero window and ok = false
// rather than stale or non-finite samples.
func TestEdgeUnevaluableShapeGivesZeroWindow(t *testing.T) {
	a := newEdge(t, 16, hannShape)
	a.shape.Family = ShapeFamily(9) // bypasses validate, which the constructor applies
	dst := make([]float32, 16)
	for i := range dst {
		dst[i] = 1
	}
	if _, ok := a.buildWindow(dst, 9, 8, 0); ok {
		t.Error("buildWindow reported ok for an unknown family")
	}
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("window[%d] = %g, want 0", i, v)
		}
	}
}
