package loudnorm

import (
	"math"

	simdf32 "github.com/tphakala/simd/f32"
)

// Calibration and gating constants from ITU-R BS.1770-4 / EBU R 128.
const (
	loudnessOffset = -0.691 // LUFS calibration offset
	// AbsoluteGateLUFS is the BS.1770-4 absolute gate: blocks at or below it
	// never count toward the integrated loudness, so a clip that stays under it
	// measures -Inf.
	AbsoluteGateLUFS = -70.0
	relativeGateLU   = -10.0 // relative gate, below the absolute-gated mean
	subBlockSeconds  = 0.1   // sub-block length (100 ms hop, 75% overlap); 4 per 400 ms block
	subBlocksPerGate = 4     // 400 ms / 100 ms

	// MinSampleRate is the lowest supported rate. Below ~3364 Hz the K-weighting
	// high-shelf center (1681.97 Hz) exceeds Nyquist and the bilinear transform
	// produces unstable coefficients; 8000 Hz (the lowest standard audio rate)
	// keeps the filter well-conditioned and the 100 ms sub-block non-empty.
	MinSampleRate = 8000
)

// Gating is performed in the energy domain (comparing weighted energies, not
// dB) to avoid a per-block log; only the final integrated value is converted
// to dB.
var (
	// absGateEnergy is the absolute -70 LUFS gate as a weighted-energy threshold:
	// a block passes when z > absGateEnergy, equivalent to
	// loudnessOffset + 10*log10(z) > AbsoluteGateLUFS.
	absGateEnergy = float32(math.Pow(10, (AbsoluteGateLUFS-loudnessOffset)/10))
	// relGateEnergyFactor is the energy ratio for the -10 LU relative gate.
	relGateEnergyFactor = float32(math.Pow(10, relativeGateLU/10))
)

// kCascadeSections is the K-weighting cascade length: the high shelf and the
// RLB high pass of BS.1770-4.
const kCascadeSections = 2

// kweightCoeffs lays the two K-weighting stages out as simd BiquadCascade
// coefficients, [b0 b1 b2 a1 a2] per section. They are computed in float64 (see
// kWeightingStages) and stored as float32; the recurrence runs in float32, which
// stays within ~0.001 LU of the float64 result even for low-frequency,
// long-duration signals.
func kweightCoeffs(s1, s2 biquad) [kCascadeSections * 5]float32 {
	var k [kCascadeSections * 5]float32
	for i, c := range [kCascadeSections]biquad{s1, s2} {
		k[i*5], k[i*5+1], k[i*5+2], k[i*5+3], k[i*5+4] =
			float32(c.b0), float32(c.b1), float32(c.b2), float32(c.a1), float32(c.a2)
	}
	return k
}

// Meter accumulates K-weighted energy from interleaved PCM and computes the
// BS.1770-4 gated integrated loudness. Feed all samples with AddFloat64/Float32/
// Int16 (one or many calls), then read IntegratedLoudness. A Meter is not safe
// for concurrent use. The per-sample numeric path is float32; the gate
// summation and the returned loudness and true-peak values are float64.
type Meter struct {
	sampleRate   int
	channels     int
	weights      []float32
	subBlockSize int // samples per channel in a 100 ms sub-block

	kCoeffs [kCascadeSections * 5]float32   // K-weighting cascade coefficients, shared by every channel
	kState  [][kCascadeSections * 4]float32 // per channel: [x1 x2 y1 y2] of each section

	accum    [][]float32 // per-channel K-weighted samples for the current sub-block
	accumLen int         // frames filled in the current sub-block

	// subEnergy[c] holds the sum of squares of each completed 100 ms sub-block
	// for channel c.
	subEnergy [][]float32

	tp []truePeakChannel // per-channel true-peak estimator (raw, not K-weighted)

	zBuf  []float32 // reusable per-block weighted-energy scratch for gating
	tpSig []float32 // reusable [history | chunk] signal buffer for true-peak convolution
}

// subBlockSamples returns the number of samples per channel in a 100 ms gating
// sub-block at the given sample rate. It is < 1 only for absurdly low rates,
// which the public API rejects.
func subBlockSamples(sampleRate int) int {
	return int(math.Round(subBlockSeconds * float64(sampleRate)))
}

// NewMeter creates a meter for the given sample rate (Hz) and interleaved
// channel count. It panics if channels is not positive or sampleRate is below
// MinSampleRate (8000 Hz); callers using the validated top-level API never hit
// that.
func NewMeter(sampleRate, channels int) *Meter {
	if channels <= 0 {
		panic("loudnorm: NewMeter requires positive channels")
	}
	if sampleRate < MinSampleRate {
		panic("loudnorm: NewMeter requires sample rate >= 8000 Hz")
	}
	s1, s2 := kWeightingStages(float64(sampleRate))
	m := &Meter{
		sampleRate:   sampleRate,
		channels:     channels,
		weights:      channelWeights(channels),
		subBlockSize: subBlockSamples(sampleRate),
		kCoeffs:      kweightCoeffs(s1, s2),
		kState:       make([][kCascadeSections * 4]float32, channels),
		accum:        make([][]float32, channels),
		subEnergy:    make([][]float32, channels),
		tp:           make([]truePeakChannel, channels),
		tpSig:        make([]float32, tapsPerPhase-1+tpChunk),
	}
	for c := range channels {
		m.accum[c] = make([]float32, m.subBlockSize)
	}
	return m
}

// Reset returns the meter to its just-constructed state so it can be reused for
// another clip without reallocating. Filter state, sub-block accumulation, true
// peak, and gating buffers are cleared while their backing arrays are retained,
// so a reused meter is allocation-free in steady state. Sample rate and channel
// count are unchanged.
func (m *Meter) Reset() {
	for c := range m.channels {
		m.kState[c] = [kCascadeSections * 4]float32{}
		m.subEnergy[c] = m.subEnergy[c][:0]
		m.tp[c] = truePeakChannel{}
	}
	m.accumLen = 0
}

// AddFloat64 feeds interleaved float64 samples (range nominally [-1, 1]). The
// length must be a multiple of the channel count; any trailing partial frame is
// ignored. Samples are narrowed to float32 internally (lossless for audio in
// [-1, 1]).
func (m *Meter) AddFloat64(interleaved []float64) { addSamples(m, interleaved, 1) }

// AddFloat32 feeds interleaved float32 samples, the native BirdNET-Go format.
func (m *Meter) AddFloat32(interleaved []float32) { addSamples(m, interleaved, 1) }

// AddInt16 feeds interleaved 16-bit PCM, converting inline with the 1/32768
// convention (exact in float32, a power of two).
func (m *Meter) AddInt16(interleaved []int16) { addSamples(m, interleaved, 1.0/32768.0) }

// sample is the set of PCM sample types the meter accepts. Each underlying type
// gets its own instantiation, so the conversions below compile to straight-line
// code rather than dictionary calls.
type sample interface{ ~int16 | ~float32 | ~float64 }

// addSamples is the one feed loop behind the three Add* entry points: scale is
// 1/32768 for int16 and 1 (an exact float32 multiply) otherwise. Frames are
// consumed in runs that end at sub-block boundaries, and each channel's K-weighting
// cascade runs over its run with the filter state held in locals (see kweightRun).
func addSamples[T sample](m *Meter, interleaved []T, scale float32) {
	stride := m.channels
	frames := len(interleaved) / stride
	if frames == 0 {
		return
	}
	for i := 0; i < frames; {
		n := min(frames-i, m.subBlockSize-m.accumLen)
		for c := range stride {
			kweightRun(m.kCoeffs[:], m.kState[c][:], m.accum[c][m.accumLen:m.accumLen+n], interleaved[i*stride+c:], stride, scale)
		}
		i += n
		m.accumLen += n
		if m.accumLen == m.subBlockSize {
			for c := range stride {
				m.subEnergy[c] = append(m.subEnergy[c], simdf32.SumOfSquares(m.accum[c]))
			}
			m.accumLen = 0
		}
	}
	for c := range stride {
		for pos := 0; pos < frames; pos += tpChunk {
			n := min(tpChunk, frames-pos)
			dst := m.tpChunkStart(c, n)
			for i := range n {
				dst[i] = float32(interleaved[(pos+i)*stride+c]) * scale
			}
			m.tpChunkEnd(c, n)
		}
	}
}

// kweightRun converts len(dst) samples of one channel, read from src at the given
// interleave stride, to scaled float32 and filters them in place through the two
// K-weighting stages with the carried state. The simd cascade rounds every product
// and sum to float32 separately, so the result does not depend on how a stream is
// split into calls or on the architecture.
func kweightRun[T sample](coeffs, state, dst []float32, src []T, stride int, scale float32) {
	for j := range dst {
		dst[j] = float32(src[j*stride]) * scale
	}
	simdf32.BiquadCascade(dst, dst, coeffs, state)
}

// tpChunkStart copies channel c's carried-over history into the shared true-peak
// signal buffer and returns the slice to fill with the next n raw samples.
func (m *Meter) tpChunkStart(c, n int) []float32 {
	copy(m.tpSig[:tapsPerPhase-1], m.tp[c].hist[:])
	return m.tpSig[tapsPerPhase-1 : tapsPerPhase-1+n]
}

// tpChunkEnd runs the fused oversampling peak detection over the filled buffer.
func (m *Meter) tpChunkEnd(c, n int) {
	m.tp[c].batch(m.tpSig[:tapsPerPhase-1+n])
}

// blockEnergies returns the per-gating-block weighted mean-square energy z_j
// (sum_i G_i * meanSquare_i) over all overlapping 400 ms blocks.
func (m *Meter) blockEnergies() []float32 {
	numSub := len(m.subEnergy[0])
	numBlocks := numSub - (subBlocksPerGate - 1)
	if numBlocks <= 0 {
		return nil
	}
	blockSamples := float32(subBlocksPerGate * m.subBlockSize)
	if cap(m.zBuf) < numBlocks {
		m.zBuf = make([]float32, numBlocks)
	}
	z := m.zBuf[:numBlocks]
	for j := range numBlocks {
		var sum float32
		for c := range m.channels {
			e := m.subEnergy[c][j] + m.subEnergy[c][j+1] + m.subEnergy[c][j+2] + m.subEnergy[c][j+3]
			sum += m.weights[c] * (e / blockSamples)
		}
		z[j] = sum
	}
	return z
}

// IntegratedLoudness returns the BS.1770-4 gated integrated loudness in LUFS.
// It returns negative infinity when the input is shorter than one 400 ms block
// or entirely below the absolute gate (silence). Per-block energies are
// computed in float32; the gate sums accumulate in float64 so precision holds
// on very long inputs, with the final dB conversion in float64.
func (m *Meter) IntegratedLoudness() float64 { return m.loudnessAtGain(0) }

// loudnessAtGain returns the gated integrated loudness the measured signal would
// have after a gain of gainDB, from the stored block energies and without reading
// any audio. The K-weighting filter is linear, so each block energy scales by
// 10^(gainDB/10); the absolute gate is a fixed level and the relative gate follows the scaled mean, which is what makes loudness
// non-linear in gain: a boost lifts blocks over the absolute gate, an attenuation
// drops them under it. At 0 dB it is IntegratedLoudness exactly.
func (m *Meter) loudnessAtGain(gainDB float64) float64 {
	z := m.blockEnergies()
	if len(z) == 0 {
		return math.Inf(-1)
	}
	scale := 1.0
	if gainDB != 0 {
		scale = math.Pow(10, gainDB/10)
	}
	absGate := float64(absGateEnergy)

	// Absolute gate at -70 LUFS (z > absGateEnergy).
	var sumAbs float64
	var cntAbs int
	for _, zj := range z {
		if e := float64(zj) * scale; e > absGate {
			sumAbs += e
			cntAbs++
		}
	}
	if cntAbs == 0 {
		return math.Inf(-1)
	}

	// Relative gate: 10 LU below the absolute-gated mean (z > 0.1 * meanAbs).
	relGate := (sumAbs / float64(cntAbs)) * float64(relGateEnergyFactor)
	var sumRel float64
	var cntRel int
	for _, zj := range z {
		if e := float64(zj) * scale; e > absGate && e > relGate {
			sumRel += e
			cntRel++
		}
	}
	if cntRel == 0 {
		return math.Inf(-1)
	}
	return loudnessOffset + 10*math.Log10(sumRel/float64(cntRel))
}

// TruePeakDBTP returns the maximum true peak across all channels in dBTP (full
// scale = 0 dBTP). It returns negative infinity for digital silence.
func (m *Meter) TruePeakDBTP() float64 {
	maxDB := math.Inf(-1)
	for c := range m.tp {
		if d := m.tp[c].dbtp(); d > maxDB {
			maxDB = d
		}
	}
	return maxDB
}

// channelWeights returns the BS.1770-4 channel weighting coefficients G_i for a
// standard interleaved layout. Mono, stereo, 3.0 and 4.0 use unity weights; 5.0
// and 5.1 apply the +1.5 dB surround weighting and exclude LFE.
func channelWeights(channels int) []float32 {
	w := make([]float32, channels)
	for i := range w {
		w[i] = 1.0
	}
	switch channels {
	case 5: // L R C Ls Rs
		w[3], w[4] = 1.41, 1.41
	case 6: // 5.1: L R C LFE Ls Rs
		w[3] = 0.0
		w[4], w[5] = 1.41, 1.41
	}
	return w
}
