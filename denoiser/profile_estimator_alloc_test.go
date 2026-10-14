//go:build !race

package denoiser

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/tphakala/go-audio-dsp/pcm"
)

// allocDelta runs f and returns the heap bytes and mallocs it performed.
func allocDelta(f func()) (bytes, mallocs uint64) {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc, after.Mallocs - before.Mallocs
}

// longSource yields a clip chunk by chunk without ever materialising it:
// loud noise with a quiet half second every 2 s, each successive one quieter,
// so every quiet window is a new best and the previous one is evicted.
type longSource struct {
	rng  *rand.Rand
	next int // next sample index
}

func (s *longSource) fill(dst []float32) {
	const sr = 48000
	for i := range dst {
		sec := float64(s.next) / sr
		amp := 0.05
		if math.Mod(sec, 2) < 0.5 {
			amp = 0.01 / (1 + sec/2) // successively quieter
		}
		dst[i] = float32(s.rng.NormFloat64() * amp)
		s.next++
	}
}

func TestNoiseProfileEstimatorMemoryBounded(t *testing.T) {
	const (
		sr      = 48000
		chunk   = 4800
		seconds = 300
		maxByte = 4 << 20
		maxMall = 200
	)
	d := mustDenoiser(t, Config{SampleRate: sr})
	for _, kind := range []string{"float32", "int16"} {
		t.Run(kind, func(t *testing.T) {
			src := &longSource{rng: rand.New(rand.NewPCG(1, 2))}
			buf := make([]float32, chunk)
			ibuf := make([]int16, chunk)
			var prof *NoiseProfile
			var perr error
			bytes, mallocs := allocDelta(func() {
				e := d.NewNoiseProfileEstimator()
				for range seconds * sr / chunk {
					src.fill(buf)
					if kind == "int16" {
						pcm.Float32ToInt16(ibuf, buf)
						e.AddInt16(ibuf)
					} else {
						e.AddFloat32(buf)
					}
				}
				prof, perr = e.Profile()
			})
			if perr != nil {
				t.Fatal(perr)
			}
			if prof.Info().Source != ProfileAuto {
				t.Fatalf("source = %v", prof.Info().Source)
			}
			t.Logf("%s: %d bytes, %d mallocs for %d s (clip as float32 would be %d bytes)",
				kind, bytes, mallocs, seconds, seconds*sr*4)
			if bytes > maxByte {
				t.Errorf("allocated %d bytes, want <= %d", bytes, maxByte)
			}
			if mallocs > maxMall {
				t.Errorf("%d mallocs, want <= %d (per-call allocation?)", mallocs, maxMall)
			}
		})
	}
}

// TestNoiseProfileEstimatorSnapshotAllocatedOnce feeds 150 well-separated,
// successively quieter windows: each evicts the previous best, and the snapshot
// buffers must be reused rather than reallocated per eviction.
func TestNoiseProfileEstimatorSnapshotAllocatedOnce(t *testing.T) {
	const sr = 48000
	d := mustDenoiser(t, Config{SampleRate: sr})
	x := make([]float32, 150*2*sr)
	for k := range 150 {
		base := k * 2 * sr
		noiseSpan(x, base, base+2*sr, 0.05, uint64(k))
		noiseSpan(x, base, base+sr/2, 0.01/float64(k+1), uint64(1000+k))
	}
	e := d.NewNoiseProfileEstimator()
	bytes, mallocs := allocDelta(func() { feedFloat(e, x, fixed(4800)) })
	t.Logf("%d bytes, %d mallocs over 150 evictions", bytes, mallocs)
	// Fixed buffers: two snapshots of region floats each, plus the energy array growth.
	if limit := uint64(2*e.region*4 + 2<<20); bytes > limit {
		t.Errorf("allocated %d bytes, want <= %d", bytes, limit)
	}
	if mallocs > 32 {
		t.Errorf("%d mallocs, want <= 32 (snapshot reallocated per eviction?)", mallocs)
	}
	if _, err := e.Profile(); err != nil {
		t.Fatal(err)
	}
}

func TestNoiseProfileEstimatorProfileAllocsBounded(t *testing.T) {
	c := findCase(t, "planted-quiet-window")
	d := mustDenoiser(t, c.cfg)
	e := d.NewNoiseProfileEstimator()
	e.AddFloat32(c.x)
	if _, err := e.Profile(); err != nil { // warm the scratch
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := e.Profile(); err != nil {
			t.Fatal(err)
		}
	})
	// NoiseProfile, its spectrum and the MeanPowerInto scratch, nothing per block.
	if allocs > 12 {
		t.Errorf("Profile allocs/op = %v, want a handful", allocs)
	}
}
