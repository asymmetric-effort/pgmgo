package blas

import "runtime"

// Threshold constants for deciding when to use optimized BLAS kernels.
const (
	// Level1Threshold is the minimum vector length for Level-1 BLAS dispatch.
	Level1Threshold = 64

	// Level2Threshold is the minimum matrix dimension for Level-2 BLAS dispatch.
	Level2Threshold = 32

	// Level3Threshold is the minimum matrix dimension for Level-3 BLAS dispatch.
	Level3Threshold = 64
)

// hasAVX2 indicates whether the CPU supports AVX2 instructions.
// In pure Go we detect this via runtime architecture; actual AVX2 usage
// will come in a future assembly pass.
var hasAVX2 bool

func init() {
	// On amd64 we assume AVX2 is available on modern hardware.
	// This flag is informational for future ASM kernels; the pure-Go
	// optimised paths run regardless.
	hasAVX2 = runtime.GOARCH == "amd64"
}

// HasAVX2 reports whether the current CPU supports AVX2.
func HasAVX2() bool {
	return hasAVX2
}

// UseBLAS returns true if an optimized BLAS kernel should be used for the
// given BLAS level (1, 2, or 3) and problem size.
//
// For Level 1, size is the vector length.
// For Level 2, size is min(m, n) of the matrix.
// For Level 3, size is min(m, n, k) of the matrix multiply.
func UseBLAS(level, size int) bool {
	switch level {
	case 1:
		return size >= Level1Threshold
	case 2:
		return size >= Level2Threshold
	case 3:
		return size >= Level3Threshold
	default:
		return false
	}
}
