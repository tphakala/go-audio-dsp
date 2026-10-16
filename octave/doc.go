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
// Config.Filter selects the band-pass design. State and sums are float64;
// samples come in as float32, and a non-finite sample is treated as 0 so it
// cannot poison the state. Process accumulates the squares of each band's output
// sample by sample, so no per-band audio is stored and memory does not depend on
// the period length. The reference is a full scale amplitude of 1.0: a full
// scale sine at a band's centre reads a mean square of 0.5 (-3.01 dB, see DB) in
// that band, in both designs.
//
// FilterBiquad, the default and the design of earlier releases, makes each band
// an equalizer BandPass (the RBJ cookbook constant 0 dB peak band-pass, with its
// bandwidth prewarp) whose width equals the band's own octave width, so the
// response keeps the designed bandwidth near Nyquist instead of narrowing as the
// centre approaches it; the filter code is the equalizer's, not a copy.
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
// 48 kHz), several times less than a band-pass without the bandwidth prewarp.
//
// FilterButterworth is an opt-in profile for a flatter and steeper response.
// Each band is an eight-pole Butterworth band-pass of four biquad sections
// (equalizer.NewButterworthBandPass) designed for the actual sample rate: its
// -3 dB points sit on Band.Lower and Band.Upper to within 0.002 dB, with the
// edge frequencies prewarped for the bilinear transform, so the band keeps its
// width at 44.1, 48 and 192 kHz and near Nyquist alike, and its gain is 0 dB at
// the band centre. Order must be 0 (the order is fixed) and NyquistMargin at
// most 0.95. Compared with the default at Order 1 it reads broadband noise about
// 1.8 dB lower:
//
//	Filter       edge level  white noise vs the rectangular band
//	Biquad (1)   -3.01 dB    +1.93 dB
//	Butterworth  -3.01 dB    +0.11 dB (+0.085 dB in the top band at 48 kHz)
//
// and a tone between two band centres reads differently, so levels recorded with
// one design are not comparable with the other. It costs about three times the
// CPU of the default (four sections per band instead of one).
//
// The relative attenuation of FilterButterworth is within the class 1 template of
// IEC 61260:1995 (identical national adoption IS 6964:2001, Table 1 and Annex B)
// for every band the meter includes, at any sample rate, evaluated from 0 Hz to
// the Nyquist frequency; the class 2 template, which is wider, holds too. This
// is a statement about that one template, not a claim of conformance to
// IEC 61260-1:2014: the 2014 acceptance limits were not checked, nor were
// effective bandwidth, linearity, time invariance or any other requirement of
// either edition. The input has no energy above Nyquist, so for a band near it
// the template's stop band breakpoints that fall above Nyquist do not apply. The
// NyquistMargin limit exists because the order used fits the template only while
// a band's upper edge is below about 0.968 of Nyquist.
//
// A-, C- and Z-weighting and F- and S-time-weighting are not implemented, and the
// meter is mono (run one per channel). The first period includes the filters'
// start-up transient; there is no warm-up to discard. The transient can last
// longer with FilterButterworth in the low bands.
package octave
