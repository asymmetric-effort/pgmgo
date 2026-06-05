package blas

// Dgemv computes a general matrix-vector product.
//
//	If trans == false:  y = alpha * A * x + beta * y
//	If trans == true:   y = alpha * A^T * x + beta * y
//
// A is m-by-n stored in row-major order with leading dimension lda.
// x has length n (or m if transposed), y has length m (or n if transposed).
func Dgemv(trans bool, m, n int, alpha float64, a []float64, lda int,
	x []float64, incx int, beta float64, y []float64, incy int) {

	if m <= 0 || n <= 0 {
		return
	}

	// Determine lengths of x and y.
	var lenX, lenY int
	if !trans {
		lenX = n
		lenY = m
	} else {
		lenX = m
		lenY = n
	}
	_ = lenX // used for clarity

	// Scale y by beta.
	if beta == 0 {
		iy := 0
		for i := 0; i < lenY; i++ {
			y[iy] = 0
			iy += incy
		}
	} else if beta != 1 {
		iy := 0
		for i := 0; i < lenY; i++ {
			y[iy] *= beta
			iy += incy
		}
	}

	if alpha == 0 {
		return
	}

	if !trans {
		// y = alpha * A * x + y (after beta scaling).
		// Row-oriented: for each row i, compute dot(A[i,:], x).
		iy := 0
		for i := 0; i < m; i++ {
			var sum float64
			ix := 0
			rowOff := i * lda
			for j := 0; j < n; j++ {
				sum += a[rowOff+j] * x[ix]
				ix += incx
			}
			y[iy] += alpha * sum
			iy += incy
		}
	} else {
		// y = alpha * A^T * x + y (after beta scaling).
		// Column-oriented: for each row i of A (column of A^T),
		// update y += alpha * x[i] * A[i,:].
		ix := 0
		for i := 0; i < m; i++ {
			if x[ix] != 0 {
				temp := alpha * x[ix]
				iy := 0
				rowOff := i * lda
				for j := 0; j < n; j++ {
					y[iy] += temp * a[rowOff+j]
					iy += incy
				}
			}
			ix += incx
		}
	}
}

// Dtrsv solves a triangular system of equations: A*x = b  or  A^T*x = b,
// overwriting x with the solution.
//
// Parameters:
//
//	uplo: 'U' for upper triangular, 'L' for lower triangular
//	trans: 'N' for no transpose, 'T' for transpose
//	diag: 'U' for unit diagonal, 'N' for non-unit diagonal
//	n: order of the matrix A
//	a: the triangular matrix in row-major order, leading dimension lda
//	lda: leading dimension of a
//	x: on entry the right-hand side b, on exit the solution
//	incx: stride for x
func Dtrsv(uplo, trans, diag byte, n int, a []float64, lda int, x []float64, incx int) {
	if n <= 0 {
		return
	}

	noUnit := diag == 'N'

	if trans == 'N' || trans == 'n' {
		if uplo == 'U' || uplo == 'u' {
			// Upper triangular, no transpose: back substitution.
			ix := (n - 1) * incx
			for i := n - 1; i >= 0; i-- {
				var sum float64
				jx := ix + incx
				for j := i + 1; j < n; j++ {
					sum += a[i*lda+j] * x[jx]
					jx += incx
				}
				x[ix] -= sum
				if noUnit {
					x[ix] /= a[i*lda+i]
				}
				ix -= incx
			}
		} else {
			// Lower triangular, no transpose: forward substitution.
			ix := 0
			for i := 0; i < n; i++ {
				var sum float64
				jx := 0
				for j := 0; j < i; j++ {
					sum += a[i*lda+j] * x[jx]
					jx += incx
				}
				x[ix] -= sum
				if noUnit {
					x[ix] /= a[i*lda+i]
				}
				ix += incx
			}
		}
	} else {
		// Transposed.
		if uplo == 'U' || uplo == 'u' {
			// Upper triangular, transpose: forward substitution.
			ix := 0
			for i := 0; i < n; i++ {
				var sum float64
				jx := 0
				for j := 0; j < i; j++ {
					sum += a[j*lda+i] * x[jx]
					jx += incx
				}
				x[ix] -= sum
				if noUnit {
					x[ix] /= a[i*lda+i]
				}
				ix += incx
			}
		} else {
			// Lower triangular, transpose: back substitution.
			ix := (n - 1) * incx
			for i := n - 1; i >= 0; i-- {
				var sum float64
				jx := ix + incx
				for j := i + 1; j < n; j++ {
					sum += a[j*lda+i] * x[jx]
					jx += incx
				}
				x[ix] -= sum
				if noUnit {
					x[ix] /= a[i*lda+i]
				}
				ix -= incx
			}
		}
	}
}
