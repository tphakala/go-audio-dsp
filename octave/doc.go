// Package octave is a streaming 1/3-octave band level meter: a bank of band-pass
// filters over the ISO 266 / IEC 61260-1 base-10 band series that reports the
// mean square of each band once per integration period.
//
// It is a meter, not a dsp.Processor. A Processor turns audio into audio of the
// same length; a Meter turns audio into one row of band energies per period, so
// its output length is unrelated to its input length. Like loudnorm it therefore
// has its own streaming contract: Process takes mono float32 chunks of any size,
// calls emit once for every completed period, carries the partial period across
// calls, and returns the same values however the input is chunked. Reset clears
// the filters and the partial period. A Meter is not safe for concurrent use.
//
// # Bands
//
// Bands are the base-10 series with exact centres 1000 * 10^(n/10) Hz and the
// preferred nominal labels (25, 31.5, 40, 50, 63, ..., 1000, 1250, ..., 20000).
// A band's edges are Exact * 10^(-0.05) and Exact * 10^(+0.05). Config selects
// the nominal range (default 25 Hz to 20 kHz) and drops any band whose upper edge
// is not below NyquistMargin (default 0.95) of the Nyquist frequency, so Bands is
// the list a consumer builds its labels from: 30 bands at 48 kHz and above, 29 at
// 44.1 kHz. A configuration that leaves no band is an error wrapping
// ErrInvalidConfig.
//
// # Filters and level
//
// Each band is an equalizer BandPass (the RBJ cookbook constant 0 dB peak
// band-pass, with its bandwidth prewarp) whose width equals the band's own octave
// width, so the response keeps the designed bandwidth near Nyquist instead of
// narrowing as the centre approaches it; the filter code is the equalizer's, not
// a copy. State and sums are float64; samples come in as float32, and a
// non-finite sample is treated as 0 so it cannot poison the state. Process
// accumulates the squares of each band's output sample by sample, so no per-band
// audio is stored and memory does not depend on the period length.
//
// The reference is a full scale amplitude of 1.0: a full scale sine at a band's
// centre reads a mean square of 0.5 (-3.01 dB, see DB) in that band.
//
// Config.Order cascades that many identical sections per band. Each section is
// designed with its -3 dB points at the band edges, so Order n puts the edges at
// -3n dB and narrows the band, which moves the level of broadband noise:
//
//	Order  edge level  white noise vs the rectangular band (1 kHz, 96 kHz)
//	1      -3.01 dB    +1.93 dB (a second order band-pass has pi/2 times its
//	                   3 dB bandwidth as noise bandwidth)
//	2      -6.02 dB    -1.05 dB
//	3      -9.03 dB    -2.30 dB
//	4      -12.04 dB   -3.09 dB
//
// A steady sine at a band's centre reads the same at every Order. Near Nyquist
// the bilinear transform warps the response: a band whose upper edge is above
// 0.3 of Nyquist reads its edges within a few dB of the figures above, and the
// error grows toward Nyquist (about 3 dB at the upper edge of the 20 kHz band at
// 48 kHz), several times less than a band-pass without the bandwidth prewarp. A
// profile that meets an IEC 61260-1 class is not implemented.
//
// The first period includes the filters' start-up transient; there is no warm-up
// to discard. A-, C- and Z-weighting and F- and S-time-weighting are not
// implemented, and the meter is mono (run one per channel).
package octave
