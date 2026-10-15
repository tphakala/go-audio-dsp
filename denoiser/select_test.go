package denoiser

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestSelectNthMatchesSort checks selectNth against a full sort for inputs that
// stress a quickselect: random, many repeats, constant, sorted, reversed, organ
// pipe and a few elements, at every index of the small ones and at the percentile
// index the estimator uses for the large ones.
func TestSelectNthMatchesSort(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 5))
	gens := map[string]func(n int) []float32{
		"random":   func(n int) []float32 { return fill(n, func(int) float32 { return r.Float32() }) },
		"repeats":  func(n int) []float32 { return fill(n, func(int) float32 { return float32(r.IntN(4)) }) },
		"constant": func(n int) []float32 { return fill(n, func(int) float32 { return 1 }) },
		"sorted":   func(n int) []float32 { return fill(n, func(i int) float32 { return float32(i) }) },
		"reversed": func(n int) []float32 { return fill(n, func(i int) float32 { return float32(n - i) }) },
		"organ":    func(n int) []float32 { return fill(n, func(i int) float32 { return float32(min(i, n-i)) }) },
	}
	for name, gen := range gens {
		for _, n := range []int{1, 2, 3, 4, 5, 8, 17, 64, 1001, 20000} {
			a := gen(n)
			want := slices.Clone(a)
			slices.Sort(want)
			ks := []int{int(quietPercentile * float64(n-1)), 0, n - 1, n / 2}
			if n <= 17 {
				ks = ks[:0]
				for k := range n {
					ks = append(ks, k)
				}
			}
			for _, k := range ks {
				if got := selectNth(slices.Clone(a), k); got != want[k] {
					t.Fatalf("%s n=%d k=%d: got %v, want %v", name, n, k, got, want[k])
				}
			}
		}
	}
}

func fill(n int, f func(i int) float32) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = f(i)
	}
	return a
}
