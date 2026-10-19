package stft

import "testing"

// BenchmarkEdgeAnalyzer runs a 15 s, 24 kHz in-memory clip through a 2048-point
// EdgeAnalyzer at hop 175. "warm" repeats the same geometry on one analyzer;
// "newhop" alternates hops 175 and 174, and after warm-up both geometries' windows
// are in the shared cache, so it measures lookups and should report no rebuilds;
// "cold" builds a new analyzer over an empty private cache for every run, the
// first-use cost of a configuration. rebuilds/op counts shape evaluations per
// operation.
func BenchmarkEdgeAnalyzer(b *testing.B) {
	const n, clip = 2048, 15 * 24000
	sig := testSignal(clip)
	shapes := []struct {
		name  string
		shape WindowShape
	}{
		{"xl-hann", WindowShape{Family: ShapeHann}},
		{"xl-dolph106.5", WindowShape{Family: ShapeDolphChebyshev, AttenuationDB: 106.5}},
	}
	for _, s := range shapes {
		b.Run(s.name+"/cold", func(b *testing.B) {
			var sink float64
			onFrame := func(power []float32) { sink += float64(power[1]) }
			b.ReportAllocs()
			var builds uint64
			for b.Loop() {
				a, err := newEdgeAnalyzer(EdgeConfig{FrameSize: n, Shape: s.shape}, newEdgeWindowCache(edgeWindowCacheBytes))
				if err != nil {
					b.Fatal(err)
				}
				if err := a.Reset(175); err != nil {
					b.Fatal(err)
				}
				a.Feed(sig, onFrame)
				a.Finish(onFrame)
				builds += a.windowBuilds
			}
			b.ReportMetric(float64(builds)/float64(b.N), "rebuilds/op")
			_ = sink
		})
		for _, mode := range []string{"warm", "newhop"} {
			b.Run(s.name+"/"+mode, func(b *testing.B) {
				a, err := NewEdgeAnalyzer(EdgeConfig{FrameSize: n, Shape: s.shape})
				if err != nil {
					b.Fatal(err)
				}
				var sink float64
				onFrame := func(power []float32) { sink += float64(power[1]) }
				hops := []int{175}
				if mode == "newhop" {
					hops = []int{175, 174}
				}
				run := func(i int) {
					if err := a.Reset(hops[i%len(hops)]); err != nil {
						b.Fatal(err)
					}
					a.Feed(sig, onFrame)
					a.Finish(onFrame)
				}
				for i := range len(hops) {
					run(i) // warm up every geometry, filling the shared cache
				}
				frames := 0
				count := func([]float32) { frames++ }
				if err := a.Reset(175); err != nil {
					b.Fatal(err)
				}
				a.Feed(sig, count)
				a.Finish(count)
				if want := int(a.NumFrames(clip)); frames != want {
					b.Fatalf("emitted %d frames, want %d", frames, want)
				}
				b.ReportAllocs()
				before := a.windowBuilds
				i := 0
				for b.Loop() {
					run(i)
					i++
				}
				b.ReportMetric(float64(a.windowBuilds-before)/float64(b.N), "rebuilds/op")
				_ = sink
				if a.windowBuilds != before {
					b.Fatalf("%s run rebuilt %d windows", mode, a.windowBuilds-before)
				}
			})
		}
	}
}
