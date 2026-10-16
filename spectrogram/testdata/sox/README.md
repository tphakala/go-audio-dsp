# sox reference renders

Reference images for the `spectrogram.Fit` tests. Each PNG is the raw output (`-r`, no axes or legend) of the sox `spectrogram` effect, so its pixels are palette indices that the tests compare with the decibel values `Fit` returns.

Rendered with sox 14.4.2, the Homebrew keg version 14.4.2_6 (`sox --version` prints an empty version suffix, so the keg version is the reference).

## Input

`clip.wav` is 1 s of mono 16-bit PCM at 24 kHz (48044 bytes, a 44-byte header). It is written by `soxClip` in `spectrogram/sox_clip_test.go`, with no input from outside the repository:

- noise from a fixed-seed xorshift64 integer generator, amplitude 0.01;
- a linear chirp from 500 Hz to 11000 Hz, amplitude 0.3;
- a 3000 Hz tone, amplitude 0.1;
- a 7777 Hz tone, amplitude 0.003, near the floor of a 100 dB range;
- digital silence from 0.45 s to 0.60 s, so the clip end still has content;
- a click of amplitude 0.9 at one third of the clip;
- rounding to int16 with saturation.

`TestSoxClipMatchesGenerator` checks the committed file against the generator (one LSB of slack, because `math.Sin` may differ in its last bit between targets).

## Commands

Run in this directory. The first 18 lines write the committed images. The last seven write images that only `TestSoxLive` makes (in a temporary directory) and compares with `Fit`.

```
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 100 -w hann -o hann-sm.png -r
sox clip.wav -n spectrogram -x 514 -y 257 -d 15 -z 100 -w hann -o hann-md.png -r
sox clip.wav -n spectrogram -x 1026 -y 513 -d 15 -z 100 -w hann -o hann-lg.png -r
sox clip.wav -n spectrogram -x 2050 -y 1025 -d 15 -z 100 -w hann -o hann-xl.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 20 -w dolph -o dolph-sm-z20.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 80 -w dolph -o dolph-sm-z80.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 100 -w dolph -o dolph-sm-z100.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 120 -w dolph -o dolph-sm-z120.png -r
sox clip.wav -n spectrogram -x 1026 -y 513 -d 15 -z 80 -w dolph -o dolph-lg-z80.png -r
sox clip.wav -n spectrogram -x 1026 -y 513 -d 15 -z 100 -w dolph -o dolph-lg-z100.png -r
sox clip.wav -n spectrogram -x 1026 -y 513 -d 15 -z 120 -w dolph -o dolph-lg-z120.png -r
sox clip.wav -n spectrogram -x 100 -y 129 -d 6000s -z 100 -w hann -o hann-sm-trunc.png -r
sox clip.wav -n trim 0 175s spectrogram -x 1026 -y 513 -d 15 -z 100 -w hann -o short175-lg-hann.png -r
sox clip.wav -n trim 0 200s spectrogram -x 1026 -y 513 -d 15 -z 100 -w hann -o short200-lg-hann.png -r
sox clip.wav -n trim 0 1000s spectrogram -x 1026 -y 513 -d 15 -z 100 -w hann -o short1000-lg-hann.png -r
sox clip.wav -n trim 0 175s spectrogram -x 2050 -y 1025 -d 15 -z 100 -w dolph -o short175-xl-dolph.png -r
sox clip.wav -n trim 0 200s spectrogram -x 2050 -y 1025 -d 15 -z 100 -w dolph -o short200-xl-dolph.png -r
sox clip.wav -n trim 0 1000s spectrogram -x 2050 -y 1025 -d 15 -z 100 -w dolph -o short1000-xl-dolph.png -r
sox clip.wav -n spectrogram -x 514 -y 257 -d 15 -z 100 -w dolph -o live-dolph-md-z100.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 40 -w dolph -o live-dolph-sm-z40.png -r
sox clip.wav -n spectrogram -x 258 -y 129 -d 15 -z 60 -w dolph -o live-dolph-sm-z60.png -r
sox clip.wav -n spectrogram -x 2050 -y 1025 -d 15 -z 120 -w dolph -o live-dolph-xl-z120.png -r
sox clip.wav -n spectrogram -x 1026 -y 513 -d 15 -z 20 -w dolph -o live-dolph-lg-z20.png -r
sox clip.wav -n spectrogram -x 10000 -y 129 -d 1 -z 100 -w hann -o live-hann-sm-cap.png -r
sox clip.wav -n spectrogram -x 514 -y 257 -d 2 -z 100 -w hann -o live-hann-md-d2.png -r
```

`-y` is the number of bins, so the transform size is `2*(y-1)`: 256, 512, 1024 and 2048 for `sm`, `md`, `lg` and `xl`. `-x` is the column limit and `-d` the span the picture covers. A span of `6000s` is 6000 samples (0.25 s). The `short` images cut the clip to 175, 200 or 1000 samples with `trim`, each shorter than one frame.

`TestSoxReadmeListsCommands` checks that every command above is the one the tests run.

## Regenerating

With sox on the path:

```
SOX_REGEN=1 go test -run TestRegenerateSoxReferences ./spectrogram
```

writes `clip.wav` and the committed images. Without sox the tests that need the binary (`TestSoxLive`, `TestSoxLiveEmpty`, `TestSoxLiveGain`) skip, and the committed images carry the parity assertions.
