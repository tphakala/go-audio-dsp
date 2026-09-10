package loudnorm

import (
	"errors"
	"slices"
	"testing"
)

// bytesLE encodes int16 samples as interleaved little-endian bytes.
func bytesLE(s []int16) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		u := uint16(v)
		b[2*i] = byte(u)
		b[2*i+1] = byte(u >> 8)
	}
	return b
}

// int16FromBytesLE decodes interleaved little-endian bytes back to int16.
func int16FromBytesLE(b []byte) []int16 {
	s := make([]int16, len(b)/2)
	for i := range s {
		s[i] = int16(uint16(b[2*i]) | uint16(b[2*i+1])<<8)
	}
	return s
}

// TestMeasureBytesMatchesInt16 checks the []byte measurement equals the []int16
// measurement of the same samples.
func TestMeasureBytesMatchesInt16(t *testing.T) {
	s := sineInt16(-20, 1000, 1.0, 48000)
	b := bytesLE(s)
	m1, err := MeasureInt16(s, 48000, 1)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := MeasureBytes(b, 48000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if m1 != m2 {
		t.Fatalf("MeasureBytes = %+v, MeasureInt16 = %+v", m2, m1)
	}
}

// TestNormalizeBytesMatchesInt16 checks the []byte normalization returns the
// same Result and writes the same samples as the []int16 normalization.
func TestNormalizeBytesMatchesInt16(t *testing.T) {
	opts := DefaultOptions()
	src := sineInt16(-20, 1000, 1.0, 48000)

	viaInt16 := slices.Clone(src)
	r1, err := NormalizeInt16(viaInt16, opts)
	if err != nil {
		t.Fatal(err)
	}
	if r1.GainDB == 0 {
		t.Fatal("test needs a non-zero gain to exercise the write-back")
	}

	b := bytesLE(src)
	r2, err := NormalizeBytes(b, opts)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Fatalf("NormalizeBytes Result = %+v, NormalizeInt16 = %+v", r2, r1)
	}
	got := int16FromBytesLE(b)
	for i := range viaInt16 {
		if got[i] != viaInt16[i] {
			t.Fatalf("sample %d: bytes path %d != int16 path %d", i, got[i], viaInt16[i])
		}
	}
}

func TestBytesOddLength(t *testing.T) {
	if _, err := MeasureBytes(make([]byte, 5), 48000, 1); !errors.Is(err, ErrOddByteLength) {
		t.Errorf("MeasureBytes odd: err = %v, want ErrOddByteLength", err)
	}
	if _, err := NormalizeBytes(make([]byte, 5), DefaultOptions()); !errors.Is(err, ErrOddByteLength) {
		t.Errorf("NormalizeBytes odd: err = %v, want ErrOddByteLength", err)
	}
}

func TestBytesValidationPropagates(t *testing.T) {
	b := bytesLE([]int16{0, 0, 0, 0})
	if _, err := MeasureBytes(b, 4, 1); !errors.Is(err, ErrSampleRateTooLow) {
		t.Errorf("MeasureBytes low rate: err = %v, want ErrSampleRateTooLow", err)
	}
	if _, err := MeasureBytes(b, 48000, 0); !errors.Is(err, ErrInvalidChannels) {
		t.Errorf("MeasureBytes zero channels: err = %v, want ErrInvalidChannels", err)
	}
	o := DefaultOptions()
	o.TruePeakDBTP = 1
	if _, err := NormalizeBytes(b, o); !errors.Is(err, ErrCeilingInvalid) {
		t.Errorf("NormalizeBytes bad ceiling: err = %v, want ErrCeilingInvalid", err)
	}
}

// TestBytesEmptyMatchesInt16 pins that empty input behaves exactly like the
// int16 entry points: it still validates the parameters and reports silence,
// rather than silently returning a zero value with no error.
func TestBytesEmptyMatchesInt16(t *testing.T) {
	mB, eB := MeasureBytes(nil, 48000, 1)
	mI, eI := MeasureInt16(nil, 48000, 1)
	if mB != mI || (eB == nil) != (eI == nil) {
		t.Fatalf("MeasureBytes(nil) = %+v,%v; MeasureInt16(nil) = %+v,%v", mB, eB, mI, eI)
	}
	// Invalid parameters on empty input must still error (previously skipped).
	if _, err := MeasureBytes(nil, 4, 1); !errors.Is(err, ErrSampleRateTooLow) {
		t.Errorf("MeasureBytes(nil, badRate) err = %v, want ErrSampleRateTooLow", err)
	}
	badOpts := Options{SampleRate: 4, Channels: 1, TargetLUFS: -23, TruePeakDBTP: -1}
	if _, err := NormalizeBytes(nil, badOpts); !errors.Is(err, ErrSampleRateTooLow) {
		t.Errorf("NormalizeBytes(nil, badRate) err = %v, want ErrSampleRateTooLow", err)
	}
}
