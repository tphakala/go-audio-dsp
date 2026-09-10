//go:build !race

// The race detector adds allocations to instrumented code, which makes
// testing.AllocsPerRun report non-zero counts. These zero-allocation assertions
// are therefore only meaningful without -race.

package pcm

import "testing"

func TestZeroAllocConversions(t *testing.T) {
	i16 := make([]int16, 1024)
	f := make([]float32, 1024)
	b := make([]byte, 2048)
	for i := range i16 {
		i16[i] = int16(i - 512)
	}
	// Confirm the byte conversions actually succeed before measuring allocations:
	// an always-erroring version would allocate nothing and pass this falsely.
	if _, err := BytesToFloat32(f, b); err != nil {
		t.Fatalf("BytesToFloat32: %v", err)
	}
	if _, err := Float32ToBytes(b, f); err != nil {
		t.Fatalf("Float32ToBytes: %v", err)
	}
	checks := []struct {
		name string
		fn   func()
	}{
		{"Int16ToFloat32", func() { Int16ToFloat32(f, i16) }},
		{"Float32ToInt16", func() { Float32ToInt16(i16, f) }},
		{"BytesToFloat32", func() { _, _ = BytesToFloat32(f, b) }},
		{"Float32ToBytes", func() { _, _ = Float32ToBytes(b, f) }},
		{"ScaleInt16", func() { ScaleInt16(i16, 2.0) }},
	}
	for _, c := range checks {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
	// InPlaceInt16 allocates nothing on little-endian hosts.
	if nativeLittleEndian {
		a := testing.AllocsPerRun(50, func() { InPlaceInt16(b, func([]int16) {}) })
		if a != 0 {
			t.Errorf("InPlaceInt16 allocated %v times, want 0", a)
		}
	}
}

// misalignSink is written by the zero-alloc probe below so the compiler cannot
// drop the InPlaceInt16 call (and the bytesAsInt16 reinterpret it makes) as dead
// code, which would make the allocation count trivially zero.
var misalignSink int16

// TestInPlaceInt16MisalignedZeroAlloc confirms the zero-copy fast path stays
// allocation-free even when the byte slice starts at an odd address (the
// unaligned int16 reinterpret). It complements the aligned case in
// TestZeroAllocConversions. Only meaningful on little-endian hosts, where
// InPlaceInt16 takes the bytesAsInt16 path; a big-endian host uses a scratch
// buffer and is expected to allocate.
func TestInPlaceInt16MisalignedZeroAlloc(t *testing.T) {
	if !nativeLittleEndian {
		t.Skip("InPlaceInt16 only avoids allocation on little-endian hosts")
	}
	b := oddStartBuf(t, 2048)
	if a := testing.AllocsPerRun(50, func() {
		InPlaceInt16(b, func(s []int16) {
			if len(s) > 0 {
				misalignSink = s[0]
			}
		})
	}); a != 0 {
		t.Errorf("InPlaceInt16 on a misaligned buffer allocated %v times, want 0", a)
	}
}
