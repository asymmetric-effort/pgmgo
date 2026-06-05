package blas

// Block sizes for cache-blocked matrix multiplication.
// These are tunable; 256 is a reasonable default for L2 cache.
const (
	mc = 256 // block rows of A / rows of C
	kc = 256 // block cols of A / rows of B (contraction dimension)
	nc = 256 // block cols of B / cols of C
)

// Micro-kernel dimensions for the inner loop.
const (
	mr = 4 // micro-panel rows
	nr = 4 // micro-panel cols
)

// Dgemm computes a general matrix-matrix product with cache blocking.
//
//	C = alpha * op(A) * op(B) + beta * C
//
// where op(X) = X if transX is false, or X^T if transX is true.
//
// A is m-by-k (after transpose), B is k-by-n (after transpose), C is m-by-n.
// Matrices are stored in row-major order with leading dimensions lda, ldb, ldc.
func Dgemm(transA, transB bool, m, n, k int,
	alpha float64, a []float64, lda int,
	b []float64, ldb int,
	beta float64, c []float64, ldc int) {

	if m <= 0 || n <= 0 || k <= 0 {
		return
	}

	// Scale C by beta.
	if beta == 0 {
		for i := 0; i < m; i++ {
			row := c[i*ldc : i*ldc+n]
			for j := range row {
				row[j] = 0
			}
		}
	} else if beta != 1 {
		for i := 0; i < m; i++ {
			row := c[i*ldc : i*ldc+n]
			for j := range row {
				row[j] *= beta
			}
		}
	}

	if alpha == 0 {
		return
	}

	// For small problems, use a simple triple loop to avoid packing overhead.
	if m <= mr && n <= nr {
		dgemmSmall(transA, transB, m, n, k, alpha, a, lda, b, ldb, c, ldc)
		return
	}

	// Allocate packing buffers.
	packA := make([]float64, mc*kc)
	packB := make([]float64, kc*nc)

	// Cache-blocked GEBP (GEneral Block Panel) algorithm.
	for jc := 0; jc < n; jc += nc {
		jb := min(nc, n-jc)
		for pc := 0; pc < k; pc += kc {
			pb := min(kc, k-pc)

			// Pack B panel: B[pc:pc+pb, jc:jc+jb] -> packB (pb x jb, row-major).
			packBPanel(transB, pb, jb, b, ldb, pc, jc, packB)

			for ic := 0; ic < m; ic += mc {
				ib := min(mc, m-ic)

				// Pack A panel: A[ic:ic+ib, pc:pc+pb] -> packA (ib x pb, row-major).
				packAPanel(transA, ib, pb, a, lda, ic, pc, packA)

				// Micro-kernel: C[ic:ic+ib, jc:jc+jb] += alpha * packA * packB
				gebpKernel(ib, jb, pb, alpha, packA, packB, c, ldc, ic, jc)
			}
		}
	}
}

// dgemmSmall handles very small matrix multiplies without packing overhead.
func dgemmSmall(transA, transB bool, m, n, k int,
	alpha float64, a []float64, lda int, b []float64, ldb int,
	c []float64, ldc int) {

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for l := 0; l < k; l++ {
				aVal := getElem(transA, a, lda, i, l)
				bVal := getElem(transB, b, ldb, l, j)
				sum += aVal * bVal
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

// getElem returns A[i][j] or A[j][i] depending on the transpose flag.
func getElem(trans bool, a []float64, lda, i, j int) float64 {
	if trans {
		return a[j*lda+i]
	}
	return a[i*lda+j]
}

// packAPanel packs a sub-panel of A into a contiguous buffer in row-major order.
// The panel covers rows [ic, ic+ib) and columns [pc, pc+pb) of op(A).
func packAPanel(transA bool, ib, pb int, a []float64, lda, ic, pc int, pack []float64) {
	idx := 0
	if !transA {
		for i := 0; i < ib; i++ {
			rowOff := (ic + i) * lda
			for l := 0; l < pb; l++ {
				pack[idx] = a[rowOff+pc+l]
				idx++
			}
		}
	} else {
		// A is transposed: A^T[i,l] = A[l,i].
		for i := 0; i < ib; i++ {
			for l := 0; l < pb; l++ {
				pack[idx] = a[(pc+l)*lda+ic+i]
				idx++
			}
		}
	}
}

// packBPanel packs a sub-panel of B into a contiguous buffer in row-major order.
// The panel covers rows [pc, pc+pb) and columns [jc, jc+jb) of op(B).
func packBPanel(transB bool, pb, jb int, b []float64, ldb, pc, jc int, pack []float64) {
	idx := 0
	if !transB {
		for l := 0; l < pb; l++ {
			rowOff := (pc + l) * ldb
			for j := 0; j < jb; j++ {
				pack[idx] = b[rowOff+jc+j]
				idx++
			}
		}
	} else {
		// B is transposed: B^T[l,j] = B[j,l].
		for l := 0; l < pb; l++ {
			for j := 0; j < jb; j++ {
				pack[idx] = b[(jc+j)*ldb+pc+l]
				idx++
			}
		}
	}
}

// gebpKernel multiplies packed A (ib x pb) by packed B (pb x jb) and
// accumulates into C[ic:ic+ib, jc:jc+jb] with scaling alpha.
// Uses a 4x4 micro-kernel with loop unrolling.
func gebpKernel(ib, jb, pb int, alpha float64, packA, packB []float64,
	c []float64, ldc, ic, jc int) {

	// Process full 4x4 micro-tiles.
	ibFull := ib - ib%mr
	jbFull := jb - jb%nr

	for i := 0; i < ibFull; i += mr {
		for j := 0; j < jbFull; j += nr {
			microKernel4x4(pb, alpha,
				packA[i*pb:], packB, j,
				c, ldc, ic+i, jc+j, jb)
		}
		// Right fringe (columns beyond jbFull).
		for j := jbFull; j < jb; j++ {
			for ii := 0; ii < mr; ii++ {
				var sum float64
				aOff := (i + ii) * pb
				for l := 0; l < pb; l++ {
					sum += packA[aOff+l] * packB[l*jb+j]
				}
				c[(ic+i+ii)*ldc+jc+j] += alpha * sum
			}
		}
	}
	// Bottom fringe (rows beyond ibFull).
	for i := ibFull; i < ib; i++ {
		aOff := i * pb
		for j := 0; j < jb; j++ {
			var sum float64
			for l := 0; l < pb; l++ {
				sum += packA[aOff+l] * packB[l*jb+j]
			}
			c[(ic+i)*ldc+jc+j] += alpha * sum
		}
	}
}

// microKernel4x4 computes a 4x4 tile of C += alpha * A_panel * B_panel.
// packA is laid out with rows of the A micro-panel at stride pb.
// packB is the full packed B panel with stride jb.
func microKernel4x4(pb int, alpha float64,
	packA []float64, packB []float64, bColOff int,
	c []float64, ldc, ci, cj, jb int) {

	// Accumulate into local registers to avoid repeated memory access.
	var c00, c01, c02, c03 float64
	var c10, c11, c12, c13 float64
	var c20, c21, c22, c23 float64
	var c30, c31, c32, c33 float64

	a0Off := 0 * pb
	a1Off := 1 * pb
	a2Off := 2 * pb
	a3Off := 3 * pb

	// Main loop with manual unrolling.
	pbUnroll := pb - pb%4
	for l := 0; l < pbUnroll; l += 4 {
		bOff0 := (l + 0) * jb
		bOff1 := (l + 1) * jb
		bOff2 := (l + 2) * jb
		bOff3 := (l + 3) * jb

		// Iteration l+0
		a0 := packA[a0Off+l]
		a1 := packA[a1Off+l]
		a2 := packA[a2Off+l]
		a3 := packA[a3Off+l]
		b0 := packB[bOff0+bColOff]
		b1 := packB[bOff0+bColOff+1]
		b2 := packB[bOff0+bColOff+2]
		b3 := packB[bOff0+bColOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// Iteration l+1
		a0 = packA[a0Off+l+1]
		a1 = packA[a1Off+l+1]
		a2 = packA[a2Off+l+1]
		a3 = packA[a3Off+l+1]
		b0 = packB[bOff1+bColOff]
		b1 = packB[bOff1+bColOff+1]
		b2 = packB[bOff1+bColOff+2]
		b3 = packB[bOff1+bColOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// Iteration l+2
		a0 = packA[a0Off+l+2]
		a1 = packA[a1Off+l+2]
		a2 = packA[a2Off+l+2]
		a3 = packA[a3Off+l+2]
		b0 = packB[bOff2+bColOff]
		b1 = packB[bOff2+bColOff+1]
		b2 = packB[bOff2+bColOff+2]
		b3 = packB[bOff2+bColOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3

		// Iteration l+3
		a0 = packA[a0Off+l+3]
		a1 = packA[a1Off+l+3]
		a2 = packA[a2Off+l+3]
		a3 = packA[a3Off+l+3]
		b0 = packB[bOff3+bColOff]
		b1 = packB[bOff3+bColOff+1]
		b2 = packB[bOff3+bColOff+2]
		b3 = packB[bOff3+bColOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}

	// Cleanup loop for remaining iterations.
	for l := pbUnroll; l < pb; l++ {
		bOff := l * jb
		a0 := packA[a0Off+l]
		a1 := packA[a1Off+l]
		a2 := packA[a2Off+l]
		a3 := packA[a3Off+l]
		b0 := packB[bOff+bColOff]
		b1 := packB[bOff+bColOff+1]
		b2 := packB[bOff+bColOff+2]
		b3 := packB[bOff+bColOff+3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}

	// Write back to C.
	c[(ci+0)*ldc+cj+0] += alpha * c00
	c[(ci+0)*ldc+cj+1] += alpha * c01
	c[(ci+0)*ldc+cj+2] += alpha * c02
	c[(ci+0)*ldc+cj+3] += alpha * c03
	c[(ci+1)*ldc+cj+0] += alpha * c10
	c[(ci+1)*ldc+cj+1] += alpha * c11
	c[(ci+1)*ldc+cj+2] += alpha * c12
	c[(ci+1)*ldc+cj+3] += alpha * c13
	c[(ci+2)*ldc+cj+0] += alpha * c20
	c[(ci+2)*ldc+cj+1] += alpha * c21
	c[(ci+2)*ldc+cj+2] += alpha * c22
	c[(ci+2)*ldc+cj+3] += alpha * c23
	c[(ci+3)*ldc+cj+0] += alpha * c30
	c[(ci+3)*ldc+cj+1] += alpha * c31
	c[(ci+3)*ldc+cj+2] += alpha * c32
	c[(ci+3)*ldc+cj+3] += alpha * c33
}
