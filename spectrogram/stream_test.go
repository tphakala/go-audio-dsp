package spectrogram

import "testing"

// feedChunked feeds sig to s in irregular chunk sizes and returns each emitted
// column (copied, since the callback slice is reused) and its center sample.
func feedChunked(s *ColumnSource, sig []float32, chunk int) (cols [][]float32, centers []int64) {
	for off := 0; off < len(sig); off += chunk {
		end := min(off+chunk, len(sig))
		s.Feed(sig[off:end], func(col []float32, centerSample int64) {
			cp := make([]float32, len(col))
			copy(cp, col)
			cols = append(cols, cp)
			centers = append(centers, centerSample)
		})
	}
	return cols, centers
}

func TestStreamMatchesBatch(t *testing.T) {
	const sr, n, hop = 16000, 512, 128
	sig := sine(6000, sr, 900, 0.7)
	for i, v := range sine(6000, sr, 2500, 0.4) {
		sig[i] += v
	}

	cfgs := []Config{
		{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: Magnitude},
		{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: Power, MinHz: 500, MaxHz: 6000},
		{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: DB, GainDB: 3, DynamicRangeDB: 100},
	}
	for _, cfg := range cfgs {
		t.Run(cfg.Scale.String(), func(t *testing.T) {
			batch, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			m := batch.Compute(sig)

			cs, err := NewColumnSource(cfg)
			if err != nil {
				t.Fatal(err)
			}
			cols, centers := feedChunked(cs, sig, 111) // deliberately not a hop multiple

			if len(cols) != m.Frames {
				t.Fatalf("stream produced %d columns, batch %d frames", len(cols), m.Frames)
			}
			for f := range m.Frames {
				bcol := m.Column(f)
				scol := cols[f]
				for b := range bcol {
					// Batch and stream run the identical framing and scale on the
					// same samples, so every scale (DB included) agrees bit for bit.
					if scol[b] != bcol[b] {
						t.Fatalf("frame %d bin %d (%s): stream %v != batch %v (want bit-exact)", f, b, cfg.Scale, scol[b], bcol[b])
					}
				}
				if want := int64(f*hop + n/2); centers[f] != want {
					t.Fatalf("frame %d centerSample = %d, want %d", f, centers[f], want)
				}
			}
		})
	}
}

func TestStreamResetAndShortFeed(t *testing.T) {
	const sr, n, hop = 8000, 256, 64
	s, err := NewColumnSource(Config{SampleRate: sr, FrameSize: n, HopSize: hop, Scale: Power})
	if err != nil {
		t.Fatal(err)
	}
	// A feed shorter than one frame emits nothing.
	got := 0
	s.Feed(make([]float32, n-1), func(col []float32, center int64) { got++ })
	if got != 0 {
		t.Fatalf("short feed emitted %d columns, want 0", got)
	}
	// Completing the frame emits one column at center n/2.
	var firstCenter int64 = -1
	s.Feed(make([]float32, 1), func(col []float32, center int64) {
		got++
		if firstCenter < 0 {
			firstCenter = center
		}
	})
	if got != 1 || firstCenter != int64(n/2) {
		t.Fatalf("got %d columns, firstCenter %d; want 1 column at %d", got, firstCenter, n/2)
	}
	// After Reset the frame counter restarts, so the next completed frame is
	// again centered at n/2.
	s.Reset()
	firstCenter = -1
	s.Feed(make([]float32, n), func(col []float32, center int64) {
		if firstCenter < 0 {
			firstCenter = center
		}
	})
	if firstCenter != int64(n/2) {
		t.Fatalf("after Reset firstCenter = %d, want %d", firstCenter, n/2)
	}
}

func TestStreamAccessors(t *testing.T) {
	const sr, n, hop = 44100, 2048, 512
	s, err := NewColumnSource(Config{SampleRate: sr, FrameSize: n, HopSize: hop, MinHz: 100, MaxHz: 8000})
	if err != nil {
		t.Fatal(err)
	}
	if s.FrameSize() != n || s.HopSize() != hop {
		t.Errorf("FrameSize/HopSize = %d/%d, want %d/%d", s.FrameSize(), s.HopSize(), n, hop)
	}
	// Accessors must agree with an equivalent Spectrogram.
	b, err := New(Config{SampleRate: sr, FrameSize: n, HopSize: hop, MinHz: 100, MaxHz: 8000})
	if err != nil {
		t.Fatal(err)
	}
	if s.Bins() != b.Bins() || s.BinHz(0) != b.BinHz(0) {
		t.Errorf("stream Bins/BinHz(0) = %d/%g, batch %d/%g", s.Bins(), s.BinHz(0), b.Bins(), b.BinHz(0))
	}
}
