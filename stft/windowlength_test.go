package stft

import (
	"slices"
	"testing"
)

// TestWindowLengthPlacement checks that a window shorter than FrameSize is
// zero-padded into the frame at the WindowAlign offset and equals the short
// window inside the placement.
func TestWindowLengthPlacement(t *testing.T) {
	const frame, win = 1024, 768
	short := GenerateWindow(Hann, win)

	t.Run("center", func(t *testing.T) {
		p, err := New(Config{FrameSize: frame, WindowLength: win, WindowAlign: AlignCenter})
		if err != nil {
			t.Fatal(err)
		}
		w := p.Window()
		if len(w) != frame {
			t.Fatalf("Window len = %d, want %d", len(w), frame)
		}
		off := (frame - win) / 2 // 128
		for i := range off {
			if w[i] != 0 {
				t.Fatalf("leading pad w[%d] = %g, want 0", i, w[i])
			}
		}
		for i := off + win; i < frame; i++ {
			if w[i] != 0 {
				t.Fatalf("trailing pad w[%d] = %g, want 0", i, w[i])
			}
		}
		for i := range short {
			if w[off+i] != short[i] {
				t.Fatalf("placed w[%d] = %g, want %g", off+i, w[off+i], short[i])
			}
		}
	})

	t.Run("left", func(t *testing.T) {
		p, err := New(Config{FrameSize: frame, WindowLength: win, WindowAlign: AlignLeft})
		if err != nil {
			t.Fatal(err)
		}
		w := p.Window()
		for i := range short {
			if w[i] != short[i] {
				t.Fatalf("placed w[%d] = %g, want %g", i, w[i], short[i])
			}
		}
		for i := win; i < frame; i++ {
			if w[i] != 0 {
				t.Fatalf("trailing pad w[%d] = %g, want 0", i, w[i])
			}
		}
	})

	t.Run("center odd pad floors left", func(t *testing.T) {
		// FrameSize 8, window 5: total pad 3, floor(3/2)=1 leading zero, 2 trailing.
		p, err := New(Config{FrameSize: 8, WindowLength: 5, WindowAlign: AlignCenter})
		if err != nil {
			t.Fatal(err)
		}
		w := p.Window()
		short5 := GenerateWindow(Hann, 5)
		if w[0] != 0 {
			t.Fatalf("leading pad w[0] = %g, want 0", w[0])
		}
		for i := range short5 {
			if w[1+i] != short5[i] {
				t.Fatalf("placed w[%d] = %g, want %g", 1+i, w[1+i], short5[i])
			}
		}
		if w[6] != 0 || w[7] != 0 {
			t.Fatalf("trailing pad w[6:8] = %g, %g, want 0, 0", w[6], w[7])
		}
	})
}

// TestWindowLengthOne covers the degenerate WindowLength == 1 path: GenerateWindow
// special-cases n == 1 to a single unit sample [1], which buildWindow then places
// at the floor-centered offset with the rest of the frame zero.
func TestWindowLengthOne(t *testing.T) {
	const fr = 8
	p, err := New(Config{FrameSize: fr, WindowLength: 1, WindowAlign: AlignCenter})
	if err != nil {
		t.Fatal(err)
	}
	w := p.Window()
	off := (fr - 1) / 2 // 3
	for i, v := range w {
		want := float32(0)
		if i == off {
			want = 1
		}
		if v != want {
			t.Fatalf("w[%d] = %g, want %g", i, v, want)
		}
	}
}

// TestWindowLengthCustomShortIsPadded checks a CustomWindow shorter than
// FrameSize is placed and zero-padded like a built-in window, for both
// alignments (the cloned-custom branch shares the padding code with the
// built-in path, but the clone itself is custom-specific).
func TestWindowLengthCustomShortIsPadded(t *testing.T) {
	const frame, win = 16, 6
	cw := GenerateWindow(Hamming, win)
	for _, align := range []WindowAlign{AlignLeft, AlignCenter} {
		p, err := New(Config{FrameSize: frame, WindowLength: win, WindowAlign: align, CustomWindow: cw})
		if err != nil {
			t.Fatalf("align=%v: %v", align, err)
		}
		w := p.Window()
		if len(w) != frame {
			t.Fatalf("align=%v Window len = %d, want %d", align, len(w), frame)
		}
		off := 0
		if align == AlignCenter {
			off = (frame - win) / 2
		}
		for i := range off {
			if w[i] != 0 {
				t.Fatalf("align=%v leading pad w[%d] = %g, want 0", align, i, w[i])
			}
		}
		for i := range cw {
			if w[off+i] != cw[i] {
				t.Fatalf("align=%v placed w[%d] = %g, want %g", align, off+i, w[off+i], cw[i])
			}
		}
		for i := off + win; i < frame; i++ {
			if w[i] != 0 {
				t.Fatalf("align=%v trailing pad w[%d] = %g, want 0", align, i, w[i])
			}
		}
	}
}

// TestWindowLengthEquivalentToPaddedCustom is the option A versus option B
// equivalence: a WindowLength < FrameSize config produces a window (and, through
// it, per-frame spectra) bit-identical to a config that hand-pads the same short
// window into a full-frame CustomWindow. This is the guarantee that lets mel
// develop against a hand-padded CustomWindow and switch to the fields later.
func TestWindowLengthEquivalentToPaddedCustom(t *testing.T) {
	const frame, hop, win = 1024, 256, 768
	sig := testSignal(4 * frame)
	for _, align := range []WindowAlign{AlignCenter, AlignLeft} {
		short := GenerateWindow(Hann, win)
		padded := make([]float32, frame)
		off := 0
		if align == AlignCenter {
			off = (frame - win) / 2
		}
		copy(padded[off:], short)

		aa, err := NewAnalyzer(Config{FrameSize: frame, HopSize: hop, WindowLength: win, WindowAlign: align})
		if err != nil {
			t.Fatalf("align=%v WindowLength config: %v", align, err)
		}
		ab, err := NewAnalyzer(Config{FrameSize: frame, HopSize: hop, CustomWindow: padded})
		if err != nil {
			t.Fatalf("align=%v padded custom config: %v", align, err)
		}

		wa, wb := aa.Window(), ab.Window()
		if len(wa) != frame || len(wb) != frame {
			t.Fatalf("align=%v window lengths %d, %d, want %d", align, len(wa), len(wb), frame)
		}
		// Positive control: an all-zero buildWindow would satisfy every equality
		// below vacuously. The Hann peak lands at the frame center for both aligns
		// (win 768 in 1024), so this is nonzero unless the window is broken.
		if wa[frame/2] == 0 {
			t.Fatalf("align=%v window is 0 at center; equivalence check would be vacuous", align)
		}
		for i := range wa {
			if wa[i] != wb[i] {
				t.Fatalf("align=%v window[%d]: WindowLength %g != padded custom %g", align, i, wa[i], wb[i])
			}
		}

		fa := collectFrames(aa, sig, 100) // odd chunk
		fb := collectFrames(ab, sig, 100)
		if len(fa) == 0 {
			t.Fatalf("align=%v produced no frames", align)
		}
		if len(fa) != len(fb) {
			t.Fatalf("align=%v frame counts %d != %d", align, len(fa), len(fb))
		}
		for f := range fa {
			for k := range fa[f] {
				if fa[f][k] != fb[f][k] { // bit-exact: identical window, identical transform
					t.Fatalf("align=%v frame %d bin %d: %v != %v", align, f, k, fa[f][k], fb[f][k])
				}
			}
		}
	}
}

// TestWindowLengthFullFrameUnchanged pins backward compatibility: an explicit
// WindowLength equal to FrameSize (and the 0 default) yields exactly the same
// window bytes as leaving the field unset.
func TestWindowLengthFullFrameUnchanged(t *testing.T) {
	const frame = 512
	def, err := New(Config{FrameSize: frame})
	if err != nil {
		t.Fatal(err)
	}
	// Positive control: the comparisons below are vacuous if buildWindow returns
	// all zeros. The default Hann peaks at 1 at the frame center.
	if len(def.Window()) != frame || def.Window()[frame/2] == 0 {
		t.Fatal("baseline window is degenerate; comparison would be vacuous")
	}
	explicit, err := New(Config{FrameSize: frame, WindowLength: frame, WindowAlign: AlignCenter})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(def.Window(), explicit.Window()) {
		t.Fatal("explicit WindowLength=FrameSize changed the window bytes")
	}
	zero, err := New(Config{FrameSize: frame, WindowLength: 0})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(def.Window(), zero.Window()) {
		t.Fatal("WindowLength 0 default changed the window bytes")
	}
}

func TestWindowAlignString(t *testing.T) {
	cases := map[WindowAlign]string{
		AlignCenter: "center",
		AlignLeft:   "left",
	}
	for a, want := range cases {
		if got := a.String(); got != want {
			t.Errorf("WindowAlign(%d).String() = %q, want %q", int(a), got, want)
		}
	}
	if got := WindowAlign(99).String(); got != "WindowAlign(99)" {
		t.Errorf("unknown align String() = %q, want %q", got, "WindowAlign(99)")
	}
}
