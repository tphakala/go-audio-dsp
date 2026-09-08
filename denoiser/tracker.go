package denoiser

import "math"

// MCRA constants (Cohen & Berdugo 2002), see the design spec section 5.3.
const (
	mcraAlphaS = 0.8  // power smoothing
	mcraDelta  = 5.0  // presence threshold on S / S_min
	mcraAlphaD = 0.95 // noise update smoothing
)

// mcra is a minima-controlled recursive-averaging noise tracker: per bin it
// smooths the frame power, tracks its minimum over a sliding window (the
// two-buffer trick), and updates the noise estimate only in frames where the
// smoothed power is within mcraDelta of that minimum (signal absent).
//
// The presence decision is hard (p = I, the paper's a_p is 0 here) rather
// than smoothed with a_p = 0.2: with the soft update, the first frame of a
// +30 dB onset leaks (1-0.8)*(1-a_d) = 1% of the onset power into the noise
// estimate, a +10 dB jump in that bin. Bird calls are exactly such onsets, so
// the gate closes immediately instead.
type mcra struct {
	s, smin, stmp []float32 // smoothed power, window minimum, running minimum
	noise         []float32 // the estimate the denoiser reads
	window, count int
	started       bool
}

func newMCRA(bins, window int) *mcra {
	m := &mcra{
		s:      make([]float32, bins),
		smin:   make([]float32, bins),
		stmp:   make([]float32, bins),
		noise:  make([]float32, bins),
		window: max(window, 1),
	}
	m.reset()
	return m
}

// reset clears the tracker; the estimate reads epsPower until the first
// update (so frames before the tracker starts pass through unattenuated).
func (m *mcra) reset() {
	m.started = false
	m.count = 0
	for k := range m.noise {
		m.noise[k] = epsPower
	}
}

// update feeds one frame's power spectrum.
func (m *mcra) update(power []float32) {
	if !m.started {
		// Seed the smoothed power and the noise estimate from the first frame,
		// but seed the minima at +Inf so they snap to the *smoothed* power on the
		// next frame rather than latching onto this single raw periodogram value.
		// A raw first frame is a high-variance draw: an unlucky low outlier would
		// otherwise seed smin so low that the presence gate never opens, freezing
		// that bin's estimate for the ~2 windows the two-buffer minimum needs to
		// forget it.
		for k, v := range power {
			if !(v < math.MaxFloat32) { // NaN or +Inf: keep the state finite
				v = epsPower
			}
			m.s[k] = v
			m.smin[k] = math.MaxFloat32
			m.stmp[k] = math.MaxFloat32
			m.noise[k] = max(v, epsPower)
		}
		m.started = true
		m.count = 1
		return
	}
	for k, pw := range power {
		if !(pw < math.MaxFloat32) { // NaN or +Inf: skip the bin, keep the state finite
			continue
		}
		s := mcraAlphaS*m.s[k] + (1-mcraAlphaS)*pw
		m.s[k] = s
		m.smin[k] = min(m.smin[k], s)
		m.stmp[k] = min(m.stmp[k], s)
		if s <= mcraDelta*m.smin[k] { // signal absent: recursive average
			m.noise[k] = max(mcraAlphaD*m.noise[k]+(1-mcraAlphaD)*pw, epsPower)
		}
	}
	m.count++
	if m.count >= m.window {
		// Window boundary (Cohen's S_min = min(S_tmp, S); S_tmp = S): the
		// boundary frame seeds the new window as well as closing the old one.
		copy(m.smin, m.stmp)
		copy(m.stmp, m.s)
		m.count = 0
	}
}

// trackWindowFrames converts the tracker window from seconds to frames.
func trackWindowFrames(sec float32, sampleRate, hop int) int {
	return max(1, int(math.Round(float64(sec)*float64(sampleRate)/float64(hop))))
}
