// Package stft is a reusable short-time Fourier analysis layer over mono float32
// audio: it owns framing, analysis windows, the padding/centering convention, and
// the per-frame real-input transform, built on github.com/tphakala/simd.
//
// Two entry points share one Config:
//
//   - Plan (New) is the whole-clip transform. Spectrum writes one complex
//     half-spectrum per frame; PowerInto writes the frame-contiguous |X|^2 power,
//     ready to feed a mel-filterbank projection. Both take an explicit PadMode so
//     the framing convention is a call-site decision, not hidden state.
//   - Analyzer (NewAnalyzer) is the streaming transform. Feed pushes arbitrary
//     chunks and runs a callback for each frame that completes, handing it the
//     frame's complex spectrum (which the callback may modify in place) and its
//     power. Inverse transforms a spectrum back to time samples, so a synthesis
//     consumer (the denoiser) can filter in the frequency domain and resynthesize.
//     Streaming is always NoPad; feed leading zeros for a centered or delayed
//     start.
//
// GenerateWindow and WOLANorm are exposed for consumers that build their own
// windows or need the overlap-add normalization for resynthesis.
//
// A Plan and an Analyzer each hold per-transform scratch through their simd plan,
// so neither is safe for concurrent use; build one per goroutine. Distinct Plans
// and Analyzers share no state. The transforms run through simd's AVX/NEON paths
// where available and portable Go elsewhere; there is no CGo.
package stft
