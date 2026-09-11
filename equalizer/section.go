package equalizer

// section is one biquad stage: fixed coefficients plus the Direct Form I state
// (the last two inputs and outputs), all in float64. A section processes one
// stream at a time; Reset clears the state.
type section struct {
	c              coeffs
	x1, x2, y1, y2 float64
}

// reset clears the stream state so the next run starts a fresh stream.
func (s *section) reset() { s.x1, s.x2, s.y1, s.y2 = 0, 0, 0, 0 }

// run filters buf in place through this section (Direct Form I). Input is read
// and output written as float32, while the recurrence accumulates in float64.
// The cascade runs one section over the whole buffer before the next, so between
// sections a sample is the float32 output of the previous stage; this makes the
// per-section arithmetic independent of chunk boundaries.
func (s *section) run(buf []float32) {
	c := s.c
	x1, x2, y1, y2 := s.x1, s.x2, s.y1, s.y2
	for i, v := range buf {
		x := float64(v)
		y := c.b0*x + c.b1*x1 + c.b2*x2 - c.a1*y1 - c.a2*y2
		x2, x1 = x1, x
		y2, y1 = y1, y
		buf[i] = float32(y)
	}
	s.x1, s.x2, s.y1, s.y2 = x1, x2, y1, y2
}

// run64 filters buf in place through this section in full float64. Unlike run,
// input and output stay float64 across the whole buffer, so a cascade of run64
// calls keeps float64 precision between sections instead of truncating to
// float32 at each stage. This is a numerically different filter from run (it
// carries more precision through the recurrence); the two share the same float64
// state, so a stream should use one or the other and reset between them.
func (s *section) run64(buf []float64) {
	c := s.c
	x1, x2, y1, y2 := s.x1, s.x2, s.y1, s.y2
	for i, x := range buf {
		y := c.b0*x + c.b1*x1 + c.b2*x2 - c.a1*y1 - c.a2*y2
		x2, x1 = x1, x
		y2, y1 = y1, y
		buf[i] = y
	}
	s.x1, s.x2, s.y1, s.y2 = x1, x2, y1, y2
}
