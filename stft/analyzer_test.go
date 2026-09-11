package stft

import (
	"math"
	"slices"
	"testing"
)

// collectFrames feeds sig through a in the given chunk size and returns a clone
// of every frame's spectrum.
func collectFrames(a *Analyzer, sig []float32, chunk int) [][]complex64 {
	var frames [][]complex64
	for i := 0; i < len(sig); i += chunk {
		end := min(i+chunk, len(sig))
		a.Feed(sig[i:end], func(spec []complex64, _ []float32) {
			frames = append(frames, slices.Clone(spec))
		})
	}
	return frames
}

// TestAnalyzerMatchesWholeClip: streaming with no preroll produces the same
// per-frame spectra as the whole-clip NoPad Spectrum.
func TestAnalyzerMatchesWholeClip(t *testing.T) {
	const n, hop = 256, 64
	sig := testSignal(10 * n)
	p, err := New(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	nf := p.NumFrames(len(sig), NoPad)
	whole := make([][]complex64, nf)
	for f := range whole {
		whole[f] = make([]complex64, p.NumBins())
	}
	p.Spectrum(whole, sig, NoPad)

	a, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	stream := collectFrames(a, sig, 100) // odd chunk
	if len(stream) != nf {
		t.Fatalf("streaming produced %d frames, want %d", len(stream), nf)
	}
	for f := range whole {
		for k := range whole[f] {
			dr := math.Abs(float64(real(stream[f][k]) - real(whole[f][k])))
			di := math.Abs(float64(imag(stream[f][k]) - imag(whole[f][k])))
			if !(dr <= 1e-3) || !(di <= 1e-3) { // fail-closed: a NaN operand fails, not passes
				t.Fatalf("frame %d bin %d: stream %v vs whole %v", f, k, stream[f][k], whole[f][k])
			}
		}
	}
}

// TestAnalyzerChunkInvariance: the frame spectra are bit-identical regardless of
// how the input is chunked (same buffer contents, same transform).
func TestAnalyzerChunkInvariance(t *testing.T) {
	const n, hop = 512, 128
	sig := testSignal(12 * n)
	ref := func() [][]complex64 {
		a, _ := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
		return collectFrames(a, sig, len(sig)) // one shot
	}()
	for _, chunk := range []int{1, 7, 63, 129, 500, 4096} {
		a, _ := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
		got := collectFrames(a, sig, chunk)
		if len(got) != len(ref) {
			t.Fatalf("chunk %d: %d frames, want %d", chunk, len(got), len(ref))
		}
		for f := range ref {
			if !slices.Equal(got[f], ref[f]) {
				t.Fatalf("chunk %d frame %d differs from one-shot (not bit-identical)", chunk, f)
			}
		}
	}
}

// TestAnalyzerInverseRoundTrip: with a rectangular window, Inverse(RFFT(frame))
// reproduces the frame within float32 tolerance.
func TestAnalyzerInverseRoundTrip(t *testing.T) {
	const n, hop = 256, 64
	a, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop, Window: Rectangular})
	if err != nil {
		t.Fatal(err)
	}
	sig := testSignal(n)
	got := make([]float32, n)
	var ran bool
	a.Feed(sig, func(spec []complex64, _ []float32) {
		a.Inverse(got, spec)
		ran = true
	})
	if !ran {
		t.Fatal("no frame completed for a full-frame feed")
	}
	for i := range sig {
		if !(math.Abs(float64(got[i]-sig[i])) <= 1e-4) { // fail-closed on NaN
			t.Fatalf("round trip at %d: got %g want %g", i, got[i], sig[i])
		}
	}
}

func TestAnalyzerInFillAndReset(t *testing.T) {
	const n, hop = 128, 32
	a, err := NewAnalyzer(Config{FrameSize: n, HopSize: hop})
	if err != nil {
		t.Fatal(err)
	}
	if a.InFill() != 0 {
		t.Fatalf("fresh InFill = %d, want 0", a.InFill())
	}
	// Feed less than a frame: no callback, InFill tracks it.
	calls := 0
	a.Feed(make([]float32, hop), func([]complex64, []float32) { calls++ })
	if calls != 0 || a.InFill() != hop {
		t.Fatalf("partial feed: calls=%d InFill=%d, want 0 and %d", calls, a.InFill(), hop)
	}
	// Fill to a full frame: one callback, then buffer slides to n-hop.
	a.Feed(make([]float32, n-hop), func([]complex64, []float32) { calls++ })
	if calls != 1 {
		t.Fatalf("completing feed: calls=%d, want 1", calls)
	}
	if a.InFill() != n-hop {
		t.Fatalf("after one frame InFill = %d, want %d", a.InFill(), n-hop)
	}
	a.Reset()
	if a.InFill() != 0 {
		t.Fatalf("after Reset InFill = %d, want 0", a.InFill())
	}
}

func TestAnalyzerAccessors(t *testing.T) {
	a, err := NewAnalyzer(Config{FrameSize: 512, HopSize: 128})
	if err != nil {
		t.Fatal(err)
	}
	if a.FrameSize() != 512 || a.HopSize() != 128 || a.NumBins() != 257 {
		t.Errorf("accessors = %d/%d/%d, want 512/128/257", a.FrameSize(), a.HopSize(), a.NumBins())
	}
	if len(a.Window()) != 512 {
		t.Errorf("Window len = %d, want 512", len(a.Window()))
	}
}

// TestAnalyzerResetClearsBuffer pins that Reset zeroes the analysis buffer (not
// just the fill counter): feed non-zero data short of a frame, Reset, and the
// backing buffer must be all zero. Removing clear(a.buf) from Reset turns this red.
func TestAnalyzerResetClearsBuffer(t *testing.T) {
	a, err := NewAnalyzer(Config{FrameSize: 128, HopSize: 32})
	if err != nil {
		t.Fatal(err)
	}
	in := make([]float32, 64) // less than a frame: no slide, no callback
	for i := range in {
		in[i] = float32(i + 1)
	}
	a.Feed(in, func([]complex64, []float32) { t.Fatal("no frame should complete for a partial feed") })
	if a.InFill() != 64 {
		t.Fatalf("InFill = %d, want 64", a.InFill())
	}
	a.Reset()
	if a.InFill() != 0 {
		t.Fatalf("after Reset InFill = %d, want 0", a.InFill())
	}
	for i, v := range a.buf {
		if v != 0 {
			t.Fatalf("Reset left buf[%d] = %g, want 0", i, v)
		}
	}
}
