package audiotest

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// FFmpegPath returns the ffmpeg binary, skipping the test when it is absent.
func FFmpegPath(t *testing.T) string {
	t.Helper()
	p, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not found in PATH; skipping afftdn oracle test")
	}
	return p
}

// F32LEBytes encodes x as little-endian float32.
func F32LEBytes(x []float32) []byte {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

// F32LESamples decodes little-endian float32 bytes.
func F32LESamples(b []byte) []float32 {
	x := make([]float32, len(b)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return x
}

// RunAfftdn runs x (mono, sr Hz) through ffmpeg's afftdn with noise reduction
// nr dB and noise floor nf dBFS, returning a buffer the length of x. The output
// is not delay-aligned; see Shifted and BestLag.
func RunAfftdn(t *testing.T, x []float32, sr, nr, nf int) []float32 {
	t.Helper()
	bin := FFmpegPath(t)
	dir := t.TempDir()
	in := filepath.Join(dir, "in.f32le")
	out := filepath.Join(dir, "out.f32le")
	if err := os.WriteFile(in, F32LEBytes(x), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "-hide_banner", "-nostats", "-y",
		"-f", "f32le", "-ar", strconv.Itoa(sr), "-ac", "1", "-i", in,
		"-af", "afftdn=nr="+strconv.Itoa(nr)+":nf="+strconv.Itoa(nf),
		"-f", "f32le", out)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("ffmpeg failed: %v\n%s", err, stderr.String())
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	y := F32LESamples(b)
	if len(y) < len(x) {
		y = append(y, make([]float32, len(x)-len(y))...)
	}
	return y[:len(x)]
}

// DecodeToF32Mono decodes a corpus clip to mono float32 at sr Hz with ffmpeg,
// which the afftdn reference already requires; this avoids a WAV parser and any
// go-wav dependency. ffmpeg accepts any container, encoding, channel count or
// source rate.
func DecodeToF32Mono(t *testing.T, bin, path string, sr int) []float32 {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "-hide_banner", "-nostats", "-v", "error",
		"-i", path, "-f", "f32le", "-ac", "1", "-ar", strconv.Itoa(sr), "-")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("ffmpeg decode %s failed: %v\n%s", filepath.Base(path), err, stderr.String())
	}
	return F32LESamples(stdout.Bytes())
}
