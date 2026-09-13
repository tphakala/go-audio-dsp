// Package spectrogram turns mono float32 audio into the numeric magnitude,
// power, or dB matrix behind a spectrogram, layered on the shared short-time
// transform in package stft. It provides two producers over one scale
// implementation: a whole-clip Spectrogram (New, ComputeInto) for rendering a
// clip to an image, and a streaming ColumnSource (NewColumnSource, Feed) that
// emits one finished column per completed frame for a live waterfall. Because
// both run the same stft.Analyzer framing and the same scale stage on the same
// samples, a streamed column equals the whole-clip column at the same frame
// offset bit for bit, for every scale (DB included). Separately, and like
// package stft, the values themselves are tolerance-stable rather than
// bit-stable across build targets and SIMD tiers; it is the batch-versus-stream
// agreement within one build that is exact.
//
// The framing is NoPad, matching stft.Analyzer: frame f spans input samples
// [f*HopSize, f*HopSize+FrameSize). A consumer that wants the image aligned to
// the very edges of the clip (centered framing) feeds its own leading zeros,
// the same convention stft documents. Presentation stays with the consumer:
// this package returns numbers (linear frequency bins plus a BinHz helper), and
// colormaps, axes, legends, log-frequency remapping, and PNG encoding live in
// the renderer.
//
// Every value type is single channel. Construction may allocate; the per-frame
// paths (ComputeInto into a caller-sized Matrix, and ColumnSource.Feed) are
// allocation-free in the steady state and route the scale arithmetic through
// github.com/tphakala/simd, so the package also builds and runs under
// SIMD_DISABLE=all. No type here is safe for concurrent use; build one per
// stream.
package spectrogram
