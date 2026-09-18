package mel

import (
	"testing"

	"github.com/tphakala/go-audio-dsp/stft"
)

// collectStream feeds sig through a ColumnSource in chunks of chunk samples and
// returns the emitted columns (copied) and their center samples.
func collectStream(t *testing.T, cfg Config, sig []float32, chunk int) (cols [][]float32, centers []int64) {
	t.Helper()
	cs, err := NewColumnSource(cfg)
	if err != nil {
		t.Fatal(err)
	}
	onCol := func(col []float32, center int64) {
		cp := make([]float32, len(col))
		copy(cp, col)
		cols = append(cols, cp)
		centers = append(centers, center)
	}
	for i := 0; i < len(sig); i += chunk {
		end := min(i+chunk, len(sig))
		cs.Feed(sig[i:end], onCol)
	}
	return cols, centers
}

// TestStreamMatchesBatchBitExact checks that a streamed column equals the
// whole-clip (NoPad) column at the same frame offset bit for bit, for every
// config, with the center sample at f*hop + FrameSize/2.
func TestStreamMatchesBatchBitExact(t *testing.T) {
	sig := toneSig(6000, 48000, []float64{1200, 4300, 9100}, 0.4)
	for name, cfg := range melTestConfigs(t) {
		t.Run(name, func(t *testing.T) {
			ex, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			m := ex.Compute(sig, stft.NoPad)
			// Feed in chunks of a non-hop-multiple so frames complete mid-chunk.
			cols, centers := collectStream(t, cfg, sig, 111)
			if len(cols) != m.Frames {
				t.Fatalf("stream frames = %d, batch frames = %d", len(cols), m.Frames)
			}
			for f := range m.Frames {
				bcol := m.Column(f)
				for b := range bcol {
					if cols[f][b] != bcol[b] {
						t.Fatalf("frame %d mel %d: stream %g != batch %g", f, b, cols[f][b], bcol[b])
					}
				}
				if want := int64(f*ex.HopSize() + ex.FrameSize()/2); centers[f] != want {
					t.Fatalf("frame %d center = %d, want %d", f, centers[f], want)
				}
			}
		})
	}
}

// TestStreamChunkInvariance checks that any chunking yields bit-identical columns.
func TestStreamChunkInvariance(t *testing.T) {
	cfg := melTestConfigs(t)["power-log10"]
	sig := toneSig(5000, 48000, []float64{1000, 5000}, 0.5)
	ref, _ := collectStream(t, cfg, sig, 1000)
	for _, chunk := range []int{1, 7, 111, 256, 4096, 5000} {
		cols, _ := collectStream(t, cfg, sig, chunk)
		if len(cols) != len(ref) {
			t.Fatalf("chunk %d: %d frames, want %d", chunk, len(cols), len(ref))
		}
		for f := range ref {
			for b := range ref[f] {
				if cols[f][b] != ref[f][b] {
					t.Fatalf("chunk %d frame %d mel %d: %g != %g", chunk, f, b, cols[f][b], ref[f][b])
				}
			}
		}
	}
}

// TestStreamReset checks that Reset restarts the center counter and frame buffer,
// so a second stream reproduces the first.
func TestStreamReset(t *testing.T) {
	cfg := melTestConfigs(t)["power-log10"]
	sig := toneSig(4000, 48000, []float64{2000}, 0.5)
	cs, err := NewColumnSource(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var first [][]float32
	var firstCenters []int64
	cs.Feed(sig, func(col []float32, center int64) {
		cp := make([]float32, len(col))
		copy(cp, col)
		first = append(first, cp)
		firstCenters = append(firstCenters, center)
	})
	cs.Reset()
	idx := 0
	cs.Feed(sig, func(col []float32, center int64) {
		if center != firstCenters[idx] {
			t.Fatalf("after Reset frame %d center = %d, want %d", idx, center, firstCenters[idx])
		}
		for b := range col {
			if col[b] != first[idx][b] {
				t.Fatalf("after Reset frame %d mel %d: %g != %g", idx, b, col[b], first[idx][b])
			}
		}
		idx++
	})
	if idx != len(first) {
		t.Fatalf("after Reset emitted %d frames, want %d", idx, len(first))
	}
}

func TestStreamAccessors(t *testing.T) {
	cfg := Config{SampleRate: 48000, FrameSize: 1024, HopSize: 256, NumMels: 40, MinHz: 500, MaxHz: 20000, Input: InputPower}
	cs, err := NewColumnSource(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if cs.FrameSize() != 1024 || cs.HopSize() != 256 || cs.NumMels() != 40 {
		t.Errorf("accessors = %d/%d/%d, want 1024/256/40", cs.FrameSize(), cs.HopSize(), cs.NumMels())
	}
	if cs.Filterbank() == nil || cs.Filterbank().NumMels() != 40 {
		t.Errorf("Filterbank() = %v, want 40-row bank", cs.Filterbank())
	}
}

// TestStreamPartialFrameNoEmit checks a Feed that does not complete a frame emits
// nothing.
func TestStreamPartialFrameNoEmit(t *testing.T) {
	cs, err := NewColumnSource(Config{SampleRate: 16000, FrameSize: 256, HopSize: 64, NumMels: 20, Input: InputPower})
	if err != nil {
		t.Fatal(err)
	}
	emitted := 0
	cs.Feed(make([]float32, 100), func([]float32, int64) { emitted++ })
	if emitted != 0 {
		t.Fatalf("partial frame emitted %d columns, want 0", emitted)
	}
}
