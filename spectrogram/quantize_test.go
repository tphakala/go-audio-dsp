package spectrogram

import "testing"

func TestQuantizeColumn(t *testing.T) {
	// Range [-100, 0] dB mapped to [0, 255]. Values are chosen away from exact
	// half-LSB boundaries, which are inherently ambiguous in float.
	col := []float32{-100, 0, -75, -200, 50, -25}
	dst := make([]uint8, len(col))
	n := QuantizeColumn(dst, col, -100, 0)
	if n != len(col) {
		t.Fatalf("wrote %d, want %d", n, len(col))
	}
	// -75 -> 25% -> 63.75 -> 64; -25 -> 75% -> 191.25 -> 191; -200 clamps to 0;
	// +50 clamps to 255.
	want := []uint8{0, 255, 64, 0, 255, 191}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("dst[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
}

func TestQuantizeColumnEdgeCases(t *testing.T) {
	// Invalid range writes nothing.
	dst := make([]uint8, 3)
	if got := QuantizeColumn(dst, []float32{1, 2, 3}, 0, 0); got != 0 {
		t.Errorf("degenerate range wrote %d, want 0", got)
	}
	if got := QuantizeColumn(dst, []float32{1, 2, 3}, 5, 1); got != 0 {
		t.Errorf("inverted range wrote %d, want 0", got)
	}
	// Mismatched lengths write the shorter count.
	short := make([]uint8, 2)
	if got := QuantizeColumn(short, []float32{1, 2, 3, 4}, 0, 10); got != 2 {
		t.Errorf("wrote %d, want 2", got)
	}
	if got := QuantizeColumn(dst, nil, 0, 10); got != 0 {
		t.Errorf("empty column wrote %d, want 0", got)
	}
}
