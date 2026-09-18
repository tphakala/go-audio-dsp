// Package mel turns mono float32 audio into per-frame (log-)mel spectrogram
// columns, layered on the shared short-time transform in package stft. It is the
// third analysis block on that core, after spectrogram, and follows the same
// shape: two producers over one projection stage, a whole-clip Extractor (New,
// Compute, ComputeInto) and a streaming ColumnSource (NewColumnSource, Feed) that
// emits one finished column per completed frame. Because both run the same
// stft.Analyzer framing and the same mel projection on the same samples, a
// streamed column equals the whole-clip column at the same frame offset bit for
// bit under NoPad. Like package stft, the values themselves are tolerance-stable
// rather than bit-stable across build targets and SIMD tiers; it is the
// batch-versus-stream agreement within one build that is exact.
//
// A column is the STFT power (or magnitude) spectrum projected through a mel
// filterbank and, optionally, compressed by a floored log. The filterbank is
// generated on the Slaney (librosa default) or HTK scale, with or without Slaney
// area normalization, or supplied precomputed through Config.Filterbank
// (FilterbankFromRows) for a model whose bank comes from a table. The whole-clip
// path takes an stft.PadMode per call, so a caller can match librosa center=True
// (PadZero or PadReflect) or center=False (NoPad); the streaming path is NoPad, and
// a consumer that wants a centered start feeds its own leading zeros.
//
// The package is deliberately pure DSP: it has no model presets and no per-model
// normalization, pitch features, frame stacking, or tensor reshaping, which belong
// to the consumer. Every value type is single channel. Construction may allocate;
// the per-frame paths (ComputeInto into a caller-sized Matrix, and
// ColumnSource.Feed) are allocation-free in the steady state and route the
// arithmetic through github.com/tphakala/simd, so the package also builds and runs
// under SIMD_DISABLE=all. No type here is safe for concurrent use; build one per
// stream. A *Filterbank is immutable and may be shared.
package mel
