# octave

Streaming 1/3-octave band level meter for Go: a bank of band-pass filters over
the ISO 266 / IEC 61260-1 band series that reports the mean square of each band
once per integration period. Pure Go, no cgo, zero allocations in steady state,
any sample rate (ultrasonic included). Part of
[`go-audio-dsp`](../README.md).

It is a meter, not a `dsp.Processor`: the output is band energy per period, not
audio of the same length.

## Usage

```go
m, err := octave.New(octave.Config{
	SampleRate:    48000,
	PeriodSamples: 48000, // one second
})
if err != nil {
	// errors.Is(err, dsp.ErrInvalidConfig)
}

bands := m.Bands() // 30 bands at 48 kHz: Nominal, Exact, Lower, Upper
m.Process(chunk, func(meanSquare []float64) {
	for i, ms := range meanSquare {
		fmt.Printf("%g Hz: %.1f dB\n", bands[i].Nominal, octave.DB(ms))
	}
})
```

`Process` takes mono `float32` chunks of any size and calls the callback once for
every completed period; the `meanSquare` slice is reused and valid only during
the call. The values do not depend on how the input is chunked. `Reset` starts a
new stream.

To use the Butterworth profile, set `Filter: octave.FilterButterworth` (leave
`Order` at 0 and `NyquistMargin` at 0.95 or below):

```go
m, err := octave.New(octave.Config{
	SampleRate:    48000,
	PeriodSamples: 48000,
	Filter:        octave.FilterButterworth,
})
```

## Levels

The reference is a full scale amplitude of 1.0 (a full scale sine has mean
square 0.5, which `DB` reports as -3.01 dB). `DB` is always finite: it floors at
-200 dB.

| Config.Order | edge level | white noise vs the rectangular band (1 kHz, 96 kHz) |
|---|---|---|
| 1 (default) | -3.01 dB | +1.93 dB |
| 2 | -6.02 dB | -1.05 dB |
| 3 | -9.03 dB | -2.30 dB |
| 4 | -12.04 dB | -3.09 dB |

Order 1 is a single second order band-pass per band. A steady sine at a band's
centre reads the same at every Order. Near Nyquist the bilinear transform warps
the response; the bandwidth prewarp keeps the error several times smaller than a
plain RBJ band-pass (see the package documentation for the numbers).

| Config.Filter | edge level | white noise vs the rectangular band | CPU |
|---|---|---|---|
| `FilterBiquad` (default), Order 1 | -3.01 dB | +1.93 dB | 1x |
| `FilterButterworth` | -3.01 dB | +0.11 dB (+0.085 dB in the top band at 48 kHz) | about 3x |

`FilterButterworth` is an eight-pole Butterworth band-pass per band whose -3 dB
points are exactly on the band edges at the actual sample rate, with 0 dB at the
centre. Its relative attenuation is within the class 1 template of IEC 61260:1995
(identical national adoption IS 6964:2001, Table 1 and Annex B) for every band
the meter includes, at any sample rate, from 0 Hz to Nyquist. This is not a claim
of IEC 61260-1:2014 conformance: the 2014 acceptance limits were not checked, nor
were effective bandwidth, linearity, time invariance or any other requirement of
either edition. Levels from the two designs are not comparable: broadband noise
reads about 1.8 dB lower with the Butterworth profile.

## Not implemented

A/C/Z weighting, F/S time weighting, multichannel input, fractional octaves
other than 1/3, and verification against the IEC 61260-1:2014 acceptance limits
(the Butterworth profile is checked against the 1995 attenuation template only).

## Design

By default each band is an [`equalizer`](../equalizer/) `BandPass` with
`Passes = Order`; with `FilterButterworth` it is
`equalizer.NewButterworthBandPass` over the band's edges. Either way the filter
code is shared with the equalizer, not copied. State and sums are
float64; squares are accumulated sample by sample, which is what makes the result
independent of chunk boundaries.
