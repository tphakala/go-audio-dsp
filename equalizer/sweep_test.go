package equalizer

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestLogSweepWithMarksErrors(t *testing.T) {
	for _, c := range []struct {
		name   string
		lo, hi float64
		n      int
	}{
		{"fLow 0", 0, 100, 10},
		{"fLow negative", -1, 100, 10},
		{"fHigh <= fLow", 100, 100, 10},
		{"n < 2", 20, 20000, 1},
		{"fHigh +Inf", 20, math.Inf(1), 10},
		{"fHigh +Inf, two points", 20, math.Inf(1), 2},
		{"one ulp wide", 20, math.Nextafter(20, 21), 5},
		{"1e-11 wide", 1000, 1000 + 1e-11, 256},
		{"float64 limit", math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64, 3},
	} {
		if _, err := LogSweepWithMarks(c.lo, c.hi, c.n, []float64{50}); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("LogSweepWithMarks(%s) err = %v, want ErrInvalidConfig", c.name, err)
		}
	}
	want, err := LogSweep(20, 20000, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range [][]float64{nil, {}, {1, 1e9, math.NaN(), math.Inf(1), math.Inf(-1)}} {
		got, err := LogSweepWithMarks(20, 20000, 100, m)
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("LogSweepWithMarks with no in-range marks %v differs from LogSweep (err %v)", m, err)
		}
	}
}

// markEditor mutates a mark set one step at a time and reports which in-range
// values the step added or removed.
type markEditor struct {
	r      *rand.Rand
	lo, hi float64
	grid   []float64
	marks  []float64
}

func (m *markEditor) inRange(f float64) bool { return m.lo <= f && f <= m.hi }

func (m *markEditor) step() (added, removed []float64) {
	switch m.r.IntN(6) {
	case 0:
		f := math.Exp(math.Log(m.lo) + m.r.Float64()*(math.Log(m.hi)-math.Log(m.lo)))
		m.marks, added = append(m.marks, f), []float64{f}
	case 1:
		f := m.grid[m.r.IntN(len(m.grid))] // a grid point: must not duplicate
		m.marks, added = append(m.marks, f), []float64{f}
	case 2:
		f := []float64{1, 1e9, math.NaN(), math.Inf(1), math.Inf(-1), m.lo, m.hi}[m.r.IntN(7)]
		m.marks = append(m.marks, f)
		if m.inRange(f) {
			added = []float64{f}
		}
	case 3:
		if len(m.marks) > 0 {
			i := m.r.IntN(len(m.marks))
			removed = []float64{m.marks[i]}
			m.marks = slices.Delete(m.marks, i, i+1)
		}
	case 4:
		if len(m.marks) > 0 {
			m.marks = append(m.marks, m.marks[m.r.IntN(len(m.marks))]) // duplicate
		}
	case 5:
		m.r.Shuffle(len(m.marks), func(i, j int) { m.marks[i], m.marks[j] = m.marks[j], m.marks[i] })
	}
	return added, removed
}

// expected is the grid plus the in-range marks, sorted and deduplicated.
func (m *markEditor) expected() []float64 {
	want := slices.Clone(m.grid)
	for _, f := range m.marks {
		if m.inRange(f) {
			want = append(want, f)
		}
	}
	slices.Sort(want)
	return slices.Compact(want)
}

// TestLogSweepWithMarksSequences edits a mark set step by step and checks the
// sweep after every edit: the grid never depends on the marks, a mark edit only
// adds or removes its own point, and the order of the marks is irrelevant.
func TestLogSweepWithMarksSequences(t *testing.T) {
	const lo, hi, n = 20.0, 20000.0, 64
	grid, err := LogSweep(lo, hi, n)
	if err != nil {
		t.Fatal(err)
	}
	ed := &markEditor{r: rand.New(rand.NewPCG(11, 12)), lo: lo, hi: hi, grid: grid, marks: []float64{}}
	var prev []float64
	for step := range 200 {
		added, removed := ed.step()
		got, err := LogSweepWithMarks(lo, hi, n, ed.marks)
		if err != nil {
			t.Fatal(err)
		}
		for i, f := range got {
			if math.IsNaN(f) || math.IsInf(f, 0) || !ed.inRange(f) || (i > 0 && !(f > got[i-1])) {
				t.Fatalf("step %d: point %d = %v is not finite, in range and strictly increasing", step, i, f)
			}
		}
		if !slices.Equal(got, ed.expected()) {
			t.Fatalf("step %d: sweep is not exactly the grid plus the in-range marks", step)
		}
		shuffled := slices.Clone(ed.marks)
		ed.r.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		if again, _ := LogSweepWithMarks(lo, hi, n, shuffled); !slices.Equal(got, again) {
			t.Fatalf("step %d: result depends on mark order", step)
		}
		if prev != nil {
			checkTransition(t, step, prev, got, grid, added, removed)
		}
		prev = got
	}
}

// checkTransition asserts that between two sweeps only the edited marks moved:
// every old point survives unless its mark was removed and it is not a grid
// point, and every new point is old or an added mark.
func checkTransition(t *testing.T, step int, prev, got, grid, added, removed []float64) {
	t.Helper()
	for _, f := range prev {
		if !slices.Contains(got, f) && (!slices.Contains(removed, f) || slices.Contains(grid, f)) {
			t.Fatalf("step %d: point %v vanished without its mark being removed", step, f)
		}
	}
	for _, f := range got {
		if !slices.Contains(prev, f) && !slices.Contains(added, f) {
			t.Fatalf("step %d: point %v appeared that is neither old nor an added mark", step, f)
		}
	}
}

// TestMarksSweepResolvesNarrowFeatures pins why marks exist: on a plain 256
// point sweep a narrow feature falls between points, while the sweep with the
// chain's marks lands on it.
func TestMarksSweepResolvesNarrowFeatures(t *testing.T) {
	const fs, n = 48000, 256
	plain, err := LogSweep(20, 20000, n)
	if err != nil {
		t.Fatal(err)
	}
	extreme := func(c *FilterChain, freqs []float64, pick func(ResponsePoint) float64, wantMax bool) float64 {
		best := math.Inf(1)
		if wantMax {
			best = math.Inf(-1)
		}
		for _, p := range c.Response(freqs) {
			if v := pick(p); (wantMax && v > best) || (!wantMax && v < best) {
				best = v
			}
		}
		return best
	}
	gain := func(p ResponsePoint) float64 { return p.GainDB }
	phase := func(p ResponsePoint) float64 { return math.Abs(p.PhaseRad) }

	for _, f0 := range []float64{1000, 3137, 9000} {
		build := func(b Band) (*FilterChain, []float64) {
			c := NewFilterChain()
			if err := c.AddFilter(mustFilter(t, b, fs)); err != nil {
				t.Fatal(err)
			}
			marked, err := LogSweepWithMarks(20, 20000, n, c.Marks())
			if err != nil {
				t.Fatal(err)
			}
			return c, marked
		}

		c, marked := build(Band{Type: BandReject, Frequency: f0, WidthHz: f0 / 30})
		if v := extreme(c, plain, gain, false); v < -12 {
			t.Errorf("notch %g Hz: plain sweep min = %.2f dB, want > -12 (the miss this test guards)", f0, v)
		}
		if v := extreme(c, marked, gain, false); v != SectionFloorDB {
			t.Errorf("notch %g Hz: marked sweep min = %v, want %v", f0, v, SectionFloorDB)
		}

		c, marked = build(Band{Type: Peaking, Frequency: f0, WidthHz: f0 / 30, GainDB: 12})
		if v := extreme(c, plain, gain, true); v > 11.5 {
			t.Errorf("peaking %g Hz: plain sweep max = %.2f dB, want < 11.5", f0, v)
		}
		if v := extreme(c, marked, gain, true); math.Abs(v-12) > 1e-9 {
			t.Errorf("peaking %g Hz: marked sweep max = %.12f dB, want 12", f0, v)
		}

		c, marked = build(Band{Type: BandPass, Frequency: f0, WidthHz: f0 / 30})
		if v := extreme(c, plain, gain, true); v > -1 {
			t.Errorf("band-pass %g Hz: plain sweep max = %.2f dB, want < -1", f0, v)
		}
		if v := extreme(c, marked, gain, true); math.Abs(v) > 1e-9 {
			t.Errorf("band-pass %g Hz: marked sweep max = %.12f dB, want 0", f0, v)
		}

		for _, ft := range []FilterType{LowPass, HighPass} {
			c, marked = build(Band{Type: ft, Frequency: f0, Q: 100})
			if v := extreme(c, plain, gain, true); v > 34 {
				t.Errorf("%v %g Hz Q 100: plain sweep max = %.2f dB, want < 34", ft, f0, v)
			}
			if v := extreme(c, marked, gain, true); math.Abs(v-40) > 1e-9 {
				t.Errorf("%v %g Hz Q 100: marked sweep max = %.12f dB, want 40", ft, f0, v)
			}
		}
	}

	c := NewFilterChain()
	if err := c.AddFilter(mustFilter(t, Band{Type: AllPass, Frequency: 3137, Q: 30}, fs)); err != nil {
		t.Fatal(err)
	}
	marked, err := LogSweepWithMarks(20, 20000, n, c.Marks())
	if err != nil {
		t.Fatal(err)
	}
	if v := extreme(c, plain, phase, true); v > 2.5 {
		t.Errorf("all-pass: plain sweep max |phase| = %.3f rad, want < 2.5", v)
	}
	if v := extreme(c, marked, phase, true); v < math.Pi-1e-9 {
		t.Errorf("all-pass: marked sweep max |phase| = %.12f rad, want pi", v)
	}

	bw, err := NewButterworthBandPass(990, 1010, 2, fs)
	if err != nil {
		t.Fatal(err)
	}
	bc := NewFilterChain()
	if err := bc.AddFilter(bw); err != nil {
		t.Fatal(err)
	}
	centre := math.Sqrt(990 * 1010)
	marked, err = LogSweepWithMarks(20, 20000, n, bc.Marks())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(marked, centre) {
		t.Fatalf("marked sweep lacks the Butterworth centre %v", centre)
	}
	if g := bc.Response([]float64{centre})[0].GainDB; math.Abs(g) > 1e-9 {
		t.Errorf("Butterworth band-pass at its geometric centre = %.12f dB, want 0", g)
	}
	if v := extreme(bc, plain, gain, true); v > -3 {
		t.Errorf("Butterworth band-pass: plain sweep max = %.2f dB, want < -3 (the miss)", v)
	}
}
