// Package denoiser is a pure-Go, streaming spectral audio denoiser.
//
// It removes stationary and slowly varying background noise (wind, rain,
// traffic hum, recorder hiss) from mono float32 audio using short-time
// spectral gain: each frame is transformed with a Hann-windowed real FFT, a
// per-bin gain derived from the estimated noise power (MMSE log-spectral
// amplitude by default, Wiener and power subtraction as alternatives) is
// applied, and the frame is resynthesized by overlap-add. Output is aligned
// sample-for-sample with input.
//
// The per-bin gain is driven by an estimate of the noise power, which comes
// from one of three sources: a profile measured from a caller-selected
// noise-only excerpt (NoiseProfileFromSamples), a profile measured
// automatically from the quietest part of a clip (EstimateNoiseProfile), or an
// adaptive minima-controlled tracker that runs when no profile is set. The
// Light, Medium and Heavy strengths map onto the gain floor and aggressiveness,
// and Params exposes every knob: the estimator, the residual gain floor,
// decision-directed smoothing and optional smoothing across frequency bins.
//
// A Denoiser streams: call Process (or ProcessInto) with consecutive chunks of
// any size and Flush (or FlushInto) at the end. Denoise, DenoiseWithNoise and
// DenoiseWithProfile wrap that for a clip already in memory. One Denoiser
// serves one stream at a time and is not safe for concurrent use; process
// stereo as two Denoisers, one per channel.
//
// This package is the default spectral (STFT) denoise method and the home of
// the shared vocabulary for a pluggable set of methods. Additional method
// families (an ML denoiser, a noise gate, a wavelet method) live as sibling
// denoiser/<method> sub-packages, each satisfying only the shared dsp.Processor
// contract (plus dsp.Flusher when it holds a tail). Method-specific concepts
// stay in their own package and never enter a shared interface; cross-cutting
// optional behaviour is a capability interface such as NoiseLearner, discovered
// by type assertion; and Strength is the coarse aggressiveness dial shared
// across methods.
//
// The transform and the elementwise kernels come from
// github.com/tphakala/simd, so the hot loops take its AVX/NEON paths where
// available and fall back to portable Go elsewhere. There is no CGo.
package denoiser
