//go:build !mips && !mipsle && !mips64 && !mips64le

package pcm

// unalignedAccessOK reports whether the target is treated as permitting unaligned
// multi-byte access for the zero-copy []byte <-> []int16 reinterpret (see
// canReinterpretInt16). It is true on every GOARCH this library targets (amd64,
// arm64, 386); only the MIPS family, which faults on an odd-address load in
// hardware, is set false and routed to the byte-wise fallback. Other little-endian
// arches whose unaligned behavior is not guaranteed (riscv64, older 32-bit arm)
// keep the fast path; they are not targets of this library, and where Go runs them
// the access is hardware- or kernel-emulated rather than trapped, so results stay
// correct and only throughput suffers. The byte-wise fallback is correct on any
// host regardless.
const unalignedAccessOK = true
