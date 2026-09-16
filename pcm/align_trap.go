//go:build mips || mipsle || mips64 || mips64le

package pcm

// unalignedAccessOK is false on the MIPS family, where an unaligned (odd-address)
// 16-bit load can trap in hardware rather than execute, or fall back to slow
// kernel trap-and-emulate. Of these, only the little-endian variants (mipsle,
// mips64le) can reach the zero-copy reinterpret (the big-endian ones already take
// the byte-swap path via nativeLittleEndian); gating all four keeps the constant
// an accurate statement about the architecture. See align_ok.go.
const unalignedAccessOK = false
