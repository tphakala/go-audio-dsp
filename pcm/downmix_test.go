package pcm

import (
	"errors"
	"math"
	"testing"
)

// ulpDiff returns the distance in float32 representable steps between a and b,
// counting through zero when the signs differ.
func ulpDiff(a, b float32) int64 {
	ord := func(f float32) int64 {
		u := int64(math.Float32bits(f))
		if u&0x80000000 != 0 {
			return -(u & 0x7fffffff)
		}
		return u
	}
	d := ord(a) - ord(b)
	if d < 0 {
		d = -d
	}
	return d
}

// interleavedSamples returns frames*channels int16 samples that cover both
// signs and the int16 endpoints.
func interleavedSamples(frames, channels int) []int16 {
	s := testSamples(frames * channels)
	if len(s) >= 2 {
		s[0] = math.MinInt16
		s[1] = math.MaxInt16
	}
	return s
}

// downmixRef is the float64 reference mean of interleaved int16 frames.
func downmixRef(s []int16, channels int) []float64 {
	frames := len(s) / channels
	out := make([]float64, frames)
	for i := range frames {
		var sum float64
		for c := range channels {
			sum += float64(s[i*channels+c])
		}
		out[i] = sum / float64(channels) / 32768
	}
	return out
}

func TestDownmixBytesStereoValues(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		const x = 12345
		s := []int16{x, x, x, -x, x, 0, 0, x, math.MaxInt16, math.MaxInt16, math.MinInt16, math.MinInt16}
		want := []float32{
			float32(x) / 32768, // L=x, R=x gives x bit-exact
			0,                  // L=x, R=-x cancels
			float32(x) / 65536, // L-only gives x/2
			float32(x) / 65536, // R-only gives x/2
			float32(math.MaxInt16) / 32768,
			-1,
		}
		dst := make([]float32, len(want))
		n, err := DownmixBytesToFloat32(dst, misaligned(leBytes(s)), 2)
		if err != nil || n != len(want) {
			t.Fatalf("n, err = %d, %v; want %d, nil", n, err, len(want))
		}
		for i := range want {
			if dst[i] != want[i] {
				t.Errorf("frame %d = %v, want %v", i, dst[i], want[i])
			}
		}
	})
}

func TestDownmixBytesMonoEqualsBytesToFloat32(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		s := interleavedSamples(1001, 1)
		src := leBytes(s)
		want := make([]float32, len(s))
		if _, err := BytesToFloat32(want, src); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(s))
		n, err := DownmixBytesToFloat32(got, src, 1)
		if err != nil || n != len(s) {
			t.Fatalf("n, err = %d, %v; want %d, nil", n, err, len(s))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("sample %d = %v, want %v (bit-exact)", i, got[i], want[i])
			}
		}
	})
}

func TestDownmixBytesAgainstReference(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		for _, ch := range []int{1, 2, 3, 4, 5, 6, 8} {
			const frames = 513
			s := interleavedSamples(frames, ch)
			ref := downmixRef(s, ch)
			got := make([]float32, frames)
			n, err := DownmixBytesToFloat32(got, misaligned(leBytes(s)), ch)
			if err != nil || n != frames {
				t.Fatalf("ch=%d n, err = %d, %v; want %d, nil", ch, n, err, frames)
			}
			for i := range ref {
				if d := ulpDiff(got[i], float32(ref[i])); d > 1 {
					t.Fatalf("ch=%d frame %d = %v, want %v (%d ulp apart)", ch, i, got[i], float32(ref[i]), d)
				}
			}
		}
	})
}

// TestDownmixBytesBothPathsAgree pins the zero-copy path against the byte-wise
// one on a source that starts at an odd address.
func TestDownmixBytesBothPathsAgree(t *testing.T) {
	if !canReinterpretInt16 {
		t.Skip("no zero-copy path on this host to compare against")
	}
	for _, ch := range []int{1, 2, 3, 6} {
		s := interleavedSamples(777, ch)
		src := oddStartBuf(t, 2*len(s))
		int16ToBytesLE(src, s)
		fast := make([]float32, 777)
		slow := make([]float32, 777)
		if _, err := DownmixBytesToFloat32(fast, src, ch); err != nil {
			t.Fatal(err)
		}
		if n := downmixBytesLE(slow, src, ch); n != 777 {
			t.Fatalf("ch=%d byte-wise path converted %d frames, want 777", ch, n)
		}
		for i := range fast {
			if math.Float32bits(fast[i]) != math.Float32bits(slow[i]) {
				t.Fatalf("ch=%d frame %d: fast %v, byte-wise %v", ch, i, fast[i], slow[i])
			}
		}
	}
}

func TestDownmixBytesShortDst(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		s := interleavedSamples(10, 3)
		ref := downmixRef(s, 3)
		dst := make([]float32, 4)
		n, err := DownmixBytesToFloat32(dst, leBytes(s), 3)
		if err != nil || n != 4 {
			t.Fatalf("n, err = %d, %v; want 4, nil", n, err)
		}
		for i := range dst {
			if ulpDiff(dst[i], float32(ref[i])) > 1 {
				t.Errorf("frame %d = %v, want %v", i, dst[i], ref[i])
			}
		}
		if n, err := DownmixBytesToFloat32(nil, leBytes(s), 3); err != nil || n != 0 {
			t.Fatalf("empty dst: n, err = %d, %v; want 0, nil", n, err)
		}
		if n, err := DownmixBytesToFloat32(dst, nil, 3); err != nil || n != 0 {
			t.Fatalf("empty src: n, err = %d, %v; want 0, nil", n, err)
		}
	})
}

func TestDownmixBytesErrors(t *testing.T) {
	bothPaths(t, func(t *testing.T) {
		t.Helper()
		dst := make([]float32, 8)
		cases := []struct {
			name     string
			src      []byte
			channels int
			want     error
		}{
			{"odd byte count", make([]byte, 5), 1, ErrOddByteLength},
			{"odd byte count stereo", make([]byte, 7), 2, ErrOddByteLength},
			{"partial frame", make([]byte, 6), 2, ErrPartialFrame},
			{"partial frame surround", make([]byte, 2*7), 6, ErrPartialFrame},
			{"zero channels", make([]byte, 4), 0, ErrInvalidConfig},
			{"negative channels", make([]byte, 4), -1, ErrInvalidConfig},
		}
		for _, c := range cases {
			n, err := DownmixBytesToFloat32(dst, c.src, c.channels)
			if !errors.Is(err, c.want) || n != 0 {
				t.Errorf("%s: n, err = %d, %v; want 0, %v", c.name, n, err, c.want)
			}
		}
	})
}

func TestDownmixFloat32Values(t *testing.T) {
	const x = 0.3
	src := []float32{x, x, x, -x, x, 0, 0, x}
	want := []float32{x, 0, x / 2, x / 2}
	dst := make([]float32, len(want))
	n, err := DownmixFloat32(dst, src, 2)
	if err != nil || n != len(want) {
		t.Fatalf("n, err = %d, %v; want %d, nil", n, err, len(want))
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("frame %d = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestDownmixFloat32MonoIsCopy(t *testing.T) {
	src := []float32{0.25, -0.5, 1, -1, 0, 1e-30}
	dst := make([]float32, len(src))
	n, err := DownmixFloat32(dst, src, 1)
	if err != nil || n != len(src) {
		t.Fatalf("n, err = %d, %v; want %d, nil", n, err, len(src))
	}
	for i := range src {
		if math.Float32bits(dst[i]) != math.Float32bits(src[i]) {
			t.Errorf("sample %d = %v, want %v", i, dst[i], src[i])
		}
	}
}

func TestDownmixFloat32AgainstReference(t *testing.T) {
	for _, ch := range []int{2, 3, 4, 5, 6, 8} {
		const frames = 257
		src := make([]float32, frames*ch)
		for i := range src {
			src[i] = float32(math.Sin(float64(i)*0.37)) * 0.9
		}
		dst := make([]float32, frames)
		n, err := DownmixFloat32(dst, src, ch)
		if err != nil || n != frames {
			t.Fatalf("ch=%d n, err = %d, %v; want %d, nil", ch, n, err, frames)
		}
		for i := range frames {
			var sum float64
			for c := range ch {
				sum += float64(src[i*ch+c])
			}
			want := float32(sum / float64(ch))
			if d := ulpDiff(dst[i], want); d > 1 {
				t.Fatalf("ch=%d frame %d = %v, want %v (%d ulp apart)", ch, i, dst[i], want, d)
			}
		}
	}
}

func TestDownmixFloat32ShortDst(t *testing.T) {
	src := []float32{1, 3, 5, 7, 9, 11}
	dst := make([]float32, 2)
	n, err := DownmixFloat32(dst, src, 2)
	if err != nil || n != 2 || dst[0] != 2 || dst[1] != 6 {
		t.Fatalf("n, err, dst = %d, %v, %v; want 2, nil, [2 6]", n, err, dst)
	}
}

func TestDownmixFloat32Errors(t *testing.T) {
	dst := make([]float32, 8)
	cases := []struct {
		name     string
		src      []float32
		channels int
		want     error
	}{
		{"partial frame", make([]float32, 3), 2, ErrPartialFrame},
		{"partial frame surround", make([]float32, 7), 6, ErrPartialFrame},
		{"zero channels", make([]float32, 4), 0, ErrInvalidConfig},
		{"negative channels", make([]float32, 4), -1, ErrInvalidConfig},
	}
	for _, c := range cases {
		n, err := DownmixFloat32(dst, c.src, c.channels)
		if !errors.Is(err, c.want) || n != 0 {
			t.Errorf("%s: n, err = %d, %v; want 0, %v", c.name, n, err, c.want)
		}
	}
}

// BenchmarkDownmixBytesToFloat32 measures a 15 s clip at 48 kHz, the size of a
// BirdNET-Go export, for stereo and 5.1 input. B/op must stay 0.
func BenchmarkDownmixBytesToFloat32(b *testing.B) {
	const frames = 15 * 48000
	for _, c := range []struct {
		name     string
		channels int
	}{{"2ch", 2}, {"6ch", 6}} {
		src := leBytes(interleavedSamples(frames, c.channels))
		dst := make([]float32, frames)
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			b.ReportAllocs()
			for b.Loop() {
				if _, err := DownmixBytesToFloat32(dst, src, c.channels); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
