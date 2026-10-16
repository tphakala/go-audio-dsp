// Package stft is a reusable short-time Fourier analysis layer over mono float32
// audio: it owns framing, analysis windows, the padding/centering convention, and
// the per-frame real-input transform, built on github.com/tphakala/simd.
//
// Four entry points:
//
//   - Plan (New) is the whole-clip transform. Spectrum writes one complex
//     half-spectrum per frame; PowerInto writes the frame-contiguous |X|^2 power,
//     ready to feed a mel-filterbank projection. Both take an explicit PadMode so
//     the framing convention is a call-site decision, not hidden state.
//   - Analyzer (NewAnalyzer) is the streaming transform. Feed pushes arbitrary
//     chunks and runs a callback for each frame that completes, handing it the
//     frame's complex spectrum (which the callback may modify in place) and its
//     power. Inverse transforms a spectrum back to time samples, so a synthesis
//     consumer can filter in the frequency domain and resynthesize.
//     Streaming is always NoPad; feed leading zeros for a centered or delayed
//     start.
//   - Synthesizer (NewSynthesizer) is the overlap-add counterpart of an
//     Analyzer: Add inverse-transforms a (possibly modified) spectrum and
//     accumulates the windowed frame, and Finish releases each finished hop
//     block divided by the WOLA normalization, so an unmodified pass
//     reconstructs the input.
//   - EdgeAnalyzer (NewEdgeAnalyzer) is a streaming power analyzer with the frame
//     schedule and edge handling of the sox 14.4.2 spectrogram effect (named as
//     the behavioural reference only): lead zeros so frames are centred on hop
//     blocks, drain zeros added by Finish, an edge-adapted symmetric window on every
//     frame that includes padding, and a per-frame amplitude factor so a full-scale
//     sinusoid centred on a bin away from DC and Nyquist reads close to power 1.
//     WindowShape selects the Hann or
//     Dolph-Chebyshev family (centre-normalized, which sets FullWindowSum). It
//     caches the edge windows across Reset calls: a repeat run on the same geometry
//     builds none, and the cache holds at most 32 windows per kind, so memory is
//     bounded by the frame size alone, not the input length.
//
// The analysis window may be shorter than the transform: Config.WindowLength in
// [1, FrameSize] with Config.WindowAlign selects a window zero-padded into the
// FrameSize frame (librosa win_length < n_fft), while the transform size, NumBins
// and framing grid stay FrameSize. The default (WindowLength 0) is a full-frame
// window, so existing configs are unchanged.
//
// GenerateWindow and WOLANorm are exposed for consumers that build their own
// windows or need the overlap-add normalization for resynthesis. DolphChebyshev
// (periodic, the default form) and DolphChebyshevSymmetric generate the
// parameterized Dolph-Chebyshev window, which feeds Config.CustomWindow.
//
// A Plan, an Analyzer and an EdgeAnalyzer each hold per-transform scratch through
// their simd plan, so none is safe for concurrent use; build one per goroutine.
// Distinct Plans, Analyzers and EdgeAnalyzers share no state; a Synthesizer shares
// its Analyzer's window and inverse-transform scratch, so the pair belongs to one
// goroutine. The transforms run through simd's AVX/NEON paths where available and
// portable Go elsewhere; there is no CGo.
package stft
