// Package equalizer is a cascade of RBJ biquad filters that runs as a streaming
// dsp.Processor over mono float32 audio.
//
// It implements the eight filter types of the Robert Bristow-Johnson audio EQ
// cookbook: low-pass, high-pass, all-pass, band-pass (constant 0 dB peak),
// band-reject (notch), low-shelf, high-shelf and peaking. A Config lists the
// bands; New builds one biquad section per band (more if a band sets Passes) and
// validates every band, rejecting parameters that would produce non-finite or
// unstable coefficients. Bands are fixed at construction.
//
// An Equalizer has zero latency: each output sample aligns with its input
// sample, so MaxOutputLen(n) is n, Latency is 0, and it carries no tail and does
// not implement dsp.Flusher. It does hold filter state between calls; Reset
// clears it to start a new stream. ProcessInto filters in place or into a
// separate buffer. An Equalizer is not safe for concurrent use; run one instance
// per stream (process stereo as two equalizers, one per channel).
//
// Coefficients and the per-sample recurrence are float64 while input and output
// are float32. The double-precision state is deliberate: at low corner
// frequencies relative to the sample rate the filter poles approach the unit
// circle, where float32 state loses the precision to stay stable. The bandwidth
// of low-pass, high-pass, all-pass and the shelves is set by Q; band-pass,
// band-reject and peaking take a bandwidth in Hz. A band whose width would put
// the lower band edge at or below 0 Hz is rejected rather than clamped.
//
// Response and LogSweep report the steady-state magnitude and phase for drawing
// a response curve; they allocate and are meant for visualization, not the audio
// path. The biquad recurrence is serial (each output depends on the previous
// one), so it has no SIMD form; it runs in scalar float64. There is no CGo.
package equalizer
