package numgo

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/numgo/internal/blas"
)

// Dot computes the dot product of two arrays.
//   - 1D-1D: inner product (scalar result wrapped in 0-D array)
//   - 2D-2D: matrix multiplication
//   - 2D-1D: matrix-vector product
func Dot(a, b *NDArray) (*NDArray, error) {
	if a.Ndim() == 1 && b.Ndim() == 1 {
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("numgo.Dot: 1D shapes mismatch: %d vs %d", a.shape[0], b.shape[0])
		}
		n := a.shape[0]
		var sum float64
		if blas.UseBLAS(1, n) {
			sum = blas.Ddot(n, a.data, 1, b.data, 1)
		} else {
			for i := 0; i < n; i++ {
				sum += a.data[i] * b.data[i]
			}
		}
		return NewNDArray([]int{1}, []float64{sum}), nil
	}
	if a.Ndim() == 2 && b.Ndim() == 2 {
		return Matmul(a, b)
	}
	if a.Ndim() == 2 && b.Ndim() == 1 {
		m, k := a.shape[0], a.shape[1]
		if k != b.shape[0] {
			return nil, fmt.Errorf("numgo.Dot: shapes (%d,%d) and (%d,) not aligned", m, k, b.shape[0])
		}
		data := make([]float64, m)
		for i := 0; i < m; i++ {
			sum := 0.0
			for j := 0; j < k; j++ {
				sum += a.Get(i, j) * b.data[j]
			}
			data[i] = sum
		}
		return NewNDArray([]int{m}, data), nil
	}
	return nil, fmt.Errorf("numgo.Dot: unsupported dimensions %dD and %dD", a.Ndim(), b.Ndim())
}

// Matmul performs matrix multiplication of two 2D arrays.
func Matmul(a, b *NDArray) (*NDArray, error) {
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("numgo.Matmul: inputs must be 2D, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		return nil, fmt.Errorf("numgo.Matmul: inner dimensions mismatch: %d vs %d", k1, k2)
	}
	data := make([]float64, m*n)
	minDim := m
	if n < minDim {
		minDim = n
	}
	if k1 < minDim {
		minDim = k1
	}
	if blas.UseBLAS(3, minDim) {
		blas.Dgemm(false, false, m, n, k1, 1.0, a.data, k1, b.data, n, 0.0, data, n)
	} else {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := 0.0
				for l := 0; l < k1; l++ {
					sum += a.Get(i, l) * b.Get(l, j)
				}
				data[i*n+j] = sum
			}
		}
	}
	return NewNDArray([]int{m, n}, data), nil
}

// Inner computes the inner product of two arrays.
// For 1D arrays, this is the dot product.
// For higher dimensions, it sums over the last axis of a and the second-to-last of b.
func Inner(a, b *NDArray) (*NDArray, error) {
	if a.Ndim() == 1 && b.Ndim() == 1 {
		return Dot(a, b)
	}
	return nil, fmt.Errorf("numgo.Inner: only 1D-1D is supported, got %dD and %dD", a.Ndim(), b.Ndim())
}

// Outer computes the outer product of two 1D arrays.
func Outer(a, b *NDArray) *NDArray {
	m := a.Size()
	n := b.Size()
	data := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = a.data[i] * b.data[j]
		}
	}
	return NewNDArray([]int{m, n}, data)
}

// Tensordot computes the tensor contraction of a and b over the last `axes` axes
// of a and the first `axes` axes of b.
func Tensordot(a, b *NDArray, axes int) (*NDArray, error) {
	if axes < 0 {
		return nil, fmt.Errorf("numgo.Tensordot: axes must be non-negative")
	}
	if axes > a.Ndim() || axes > b.Ndim() {
		return nil, fmt.Errorf("numgo.Tensordot: axes=%d exceeds array dimensions", axes)
	}
	// Check that contracted dimensions match.
	for i := 0; i < axes; i++ {
		da := a.shape[a.Ndim()-axes+i]
		db := b.shape[i]
		if da != db {
			return nil, fmt.Errorf("numgo.Tensordot: contracted dimension mismatch at axis %d: %d vs %d", i, da, db)
		}
	}
	// Output shape: a's free axes ++ b's free axes.
	outShape := make([]int, 0, a.Ndim()+b.Ndim()-2*axes)
	for i := 0; i < a.Ndim()-axes; i++ {
		outShape = append(outShape, a.shape[i])
	}
	for i := axes; i < b.Ndim(); i++ {
		outShape = append(outShape, b.shape[i])
	}
	if len(outShape) == 0 {
		// Scalar result.
		sum := 0.0
		for i := 0; i < a.Size(); i++ {
			sum += a.data[i] * b.data[i]
		}
		return NewNDArray([]int{1}, []float64{sum}), nil
	}

	// Compute contracted dimension size.
	contractedSize := 1
	for i := 0; i < axes; i++ {
		contractedSize *= a.shape[a.Ndim()-axes+i]
	}

	// Reshape a to (freeA, contracted) and b to (contracted, freeB).
	freeASize := a.Size() / contractedSize
	freeBSize := b.Size() / contractedSize

	result := make([]float64, freeASize*freeBSize)
	for i := 0; i < freeASize; i++ {
		for j := 0; j < freeBSize; j++ {
			sum := 0.0
			for k := 0; k < contractedSize; k++ {
				sum += a.data[i*contractedSize+k] * b.data[k*freeBSize+j]
			}
			result[i*freeBSize+j] = sum
		}
	}
	return NewNDArray(outShape, result), nil
}

// Solve solves the linear system Ax = b via Gaussian elimination with partial pivoting.
// a must be a square 2D matrix, b must be a 1D or 2D array.
//
// For large systems (n >= Level2Threshold), uses LU factorization with BLAS
// triangular solve (Dtrsv) for better cache performance.
func Solve(a, b *NDArray) (*NDArray, error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("numgo.Solve: a must be a square 2D matrix")
	}
	n := a.shape[0]

	if b.Ndim() == 1 {
		if b.shape[0] != n {
			return nil, fmt.Errorf("numgo.Solve: dimension mismatch: a is %dx%d, b has %d elements", n, n, b.shape[0])
		}
		if blas.UseBLAS(2, n) {
			return solveBLAS(a, b)
		}
		return solveNaive(a, b)
	}

	if b.Ndim() == 2 {
		if b.shape[0] != n {
			return nil, fmt.Errorf("numgo.Solve: dimension mismatch")
		}
		nrhs := b.shape[1]
		resultData := make([]float64, n*nrhs)
		for col := 0; col < nrhs; col++ {
			bCol := make([]float64, n)
			for i := 0; i < n; i++ {
				bCol[i] = b.Get(i, col)
			}
			bArr := NewNDArray([]int{n}, bCol)
			xArr, err := Solve(a, bArr)
			if err != nil {
				return nil, err
			}
			for i := 0; i < n; i++ {
				resultData[i*nrhs+col] = xArr.data[i]
			}
		}
		return NewNDArray([]int{n, nrhs}, resultData), nil
	}

	return nil, fmt.Errorf("numgo.Solve: b must be 1D or 2D")
}

// solveNaive solves Ax = b using Gaussian elimination (the original path).
func solveNaive(a, b *NDArray) (*NDArray, error) {
	n := a.shape[0]
	// Augmented matrix [A|b].
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		for j := 0; j < n; j++ {
			aug[i][j] = a.Get(i, j)
		}
		aug[i][n] = b.data[i]
	}
	// Forward elimination with partial pivoting.
	for col := 0; col < n; col++ {
		maxVal := math.Abs(aug[col][col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			return nil, fmt.Errorf("numgo.Solve: singular matrix")
		}
		aug[col], aug[maxRow] = aug[maxRow], aug[col]
		for row := col + 1; row < n; row++ {
			factor := aug[row][col] / aug[col][col]
			for j := col; j <= n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}
	// Back substitution.
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
		x[i] /= aug[i][i]
	}
	return NewNDArray([]int{n}, x), nil
}

// solveBLAS solves Ax = b using LU factorization with BLAS-accelerated
// triangular solve for better cache performance on larger systems.
func solveBLAS(a, b *NDArray) (*NDArray, error) {
	n := a.shape[0]

	// Copy A into a flat row-major LU buffer and build pivot array.
	lu := make([]float64, n*n)
	copy(lu, a.data)
	piv := make([]int, n)
	for i := range piv {
		piv[i] = i
	}

	// LU factorization with partial pivoting (in-place).
	for col := 0; col < n; col++ {
		// Find pivot.
		maxVal := math.Abs(lu[col*n+col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if v := math.Abs(lu[row*n+col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			return nil, fmt.Errorf("numgo.Solve: singular matrix")
		}
		if maxRow != col {
			// Swap rows in LU and record pivot.
			for j := 0; j < n; j++ {
				lu[col*n+j], lu[maxRow*n+j] = lu[maxRow*n+j], lu[col*n+j]
			}
			piv[col], piv[maxRow] = piv[maxRow], piv[col]
		}
		// Compute multipliers and update trailing submatrix.
		pivot := lu[col*n+col]
		for row := col + 1; row < n; row++ {
			lu[row*n+col] /= pivot
			factor := lu[row*n+col]
			for j := col + 1; j < n; j++ {
				lu[row*n+j] -= factor * lu[col*n+j]
			}
		}
	}

	// Apply permutation to b: pb = P * b.
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = b.data[piv[i]]
	}

	// Solve L*y = pb using Dtrsv (unit diagonal lower triangular).
	blas.Dtrsv('L', 'N', 'U', n, lu, n, x, 1)

	// Solve U*x = y using Dtrsv (non-unit diagonal upper triangular).
	blas.Dtrsv('U', 'N', 'N', n, lu, n, x, 1)

	return NewNDArray([]int{n}, x), nil
}

// Inv computes the inverse of a square matrix via Gauss-Jordan elimination.
func Inv(a *NDArray) (*NDArray, error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("numgo.Inv: input must be a square 2D matrix")
	}
	n := a.shape[0]

	// Build augmented matrix [A|I].
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = a.Get(i, j)
		}
		aug[i][n+i] = 1.0
	}

	// Gauss-Jordan elimination with partial pivoting.
	for col := 0; col < n; col++ {
		maxVal := math.Abs(aug[col][col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			return nil, fmt.Errorf("numgo.Inv: singular matrix")
		}
		aug[col], aug[maxRow] = aug[maxRow], aug[col]

		pivot := aug[col][col]
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= pivot
		}

		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Extract inverse.
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = aug[i][n+j]
		}
	}
	return NewNDArray([]int{n, n}, data), nil
}

// Det computes the determinant of a square matrix via LU decomposition (Gaussian elimination).
func Det(a *NDArray) (float64, error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return 0, fmt.Errorf("numgo.Det: input must be a square 2D matrix")
	}
	n := a.shape[0]
	if n == 0 {
		return 1, nil
	}

	// Copy matrix.
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			m[i][j] = a.Get(i, j)
		}
	}

	det := 1.0
	for col := 0; col < n; col++ {
		// Partial pivoting.
		maxVal := math.Abs(m[col][col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if math.Abs(m[row][col]) > maxVal {
				maxVal = math.Abs(m[row][col])
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			return 0, nil
		}
		if maxRow != col {
			m[col], m[maxRow] = m[maxRow], m[col]
			det *= -1
		}
		det *= m[col][col]
		for row := col + 1; row < n; row++ {
			factor := m[row][col] / m[col][col]
			for j := col + 1; j < n; j++ {
				m[row][j] -= factor * m[col][j]
			}
		}
	}
	return det, nil
}

// Eig computes the eigenvalues and right eigenvectors of a square matrix
// using the QR algorithm. Only supports real eigenvalues.
func Eig(a *NDArray) (values, vectors *NDArray, err error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, nil, fmt.Errorf("numgo.Eig: input must be a square 2D matrix")
	}
	n := a.shape[0]

	// QR algorithm: iterate A_k+1 = R * Q until convergence.
	ak := a.Copy()
	// Accumulate eigenvectors.
	vAcc := Eye(n)
	maxIter := 1000
	for iter := 0; iter < maxIter; iter++ {
		q, r, qrerr := QR(ak)
		if qrerr != nil {
			return nil, nil, qrerr
		}
		// A_{k+1} = R * Q
		newA, merr := Matmul(r, q)
		if merr != nil {
			return nil, nil, merr
		}
		// Accumulate: V = V * Q
		newV, verr := Matmul(vAcc, q)
		if verr != nil {
			return nil, nil, verr
		}
		vAcc = newV

		// Check convergence: sub-diagonal elements near zero.
		offDiag := 0.0
		for i := 1; i < n; i++ {
			offDiag += math.Abs(newA.Get(i, i-1))
		}
		ak = newA
		if offDiag < 1e-10 {
			break
		}
	}

	// Extract eigenvalues from diagonal.
	eigvals := make([]float64, n)
	for i := 0; i < n; i++ {
		eigvals[i] = ak.Get(i, i)
	}

	return NewNDArray([]int{n}, eigvals), vAcc, nil
}

// Eigh computes eigenvalues and eigenvectors of a symmetric matrix.
// Uses the same QR algorithm but assumes symmetry for better convergence.
func Eigh(a *NDArray) (values, vectors *NDArray, err error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, nil, fmt.Errorf("numgo.Eigh: input must be a square 2D matrix")
	}
	return Eig(a)
}

// Eigvals returns only the eigenvalues of a square matrix.
func Eigvals(a *NDArray) (*NDArray, error) {
	vals, _, err := Eig(a)
	return vals, err
}

// SVD computes the singular value decomposition A = U * diag(S) * Vt.
// Uses eigendecomposition of A^T A and A A^T.
func SVD(a *NDArray) (u, s, vt *NDArray, err error) {
	if a.Ndim() != 2 {
		return nil, nil, nil, fmt.Errorf("numgo.SVD: input must be 2D")
	}
	m, n := a.shape[0], a.shape[1]
	at := a.T()

	// Compute A^T * A (n x n).
	ata, err := Matmul(at, a)
	if err != nil {
		return nil, nil, nil, err
	}

	// Eigendecomposition of A^T A.
	eigvals, eigvecs, err := Eig(ata)
	if err != nil {
		return nil, nil, nil, err
	}

	// Sort eigenvalues in descending order.
	k := eigvals.shape[0]
	indices := make([]int, k)
	for i := range indices {
		indices[i] = i
	}
	evData := eigvals.Data()
	for i := 0; i < k; i++ {
		for j := i + 1; j < k; j++ {
			if evData[indices[j]] > evData[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	// Singular values are sqrt of eigenvalues.
	minDim := m
	if n < m {
		minDim = n
	}
	sData := make([]float64, minDim)
	for i := 0; i < minDim && i < k; i++ {
		val := evData[indices[i]]
		if val < 0 {
			val = 0
		}
		sData[i] = math.Sqrt(val)
	}
	s = NewNDArray([]int{minDim}, sData)

	// V (right singular vectors) from eigenvectors of A^T A, sorted.
	vtData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			vtData[i*n+j] = eigvecs.Get(j, indices[i])
		}
	}
	vt = NewNDArray([]int{n, n}, vtData)

	// U = A * V * S^{-1} for non-zero singular values.
	v := vt.T()
	uData := make([]float64, m*m)
	for i := 0; i < minDim; i++ {
		if sData[i] < 1e-14 {
			continue
		}
		// u_i = A * v_i / s_i
		for row := 0; row < m; row++ {
			sum := 0.0
			for col := 0; col < n; col++ {
				sum += a.Get(row, col) * v.Get(col, i)
			}
			uData[row*m+i] = sum / sData[i]
		}
	}
	// Fill remaining columns of U with orthogonal vectors if m > minDim.
	// For a basic implementation, we use Gram-Schmidt on remaining basis vectors.
	if m > minDim {
		for i := minDim; i < m; i++ {
			// Start with standard basis vector e_i.
			col := make([]float64, m)
			col[i] = 1.0
			// Orthogonalize against existing columns.
			for j := 0; j < i; j++ {
				dot := 0.0
				for r := 0; r < m; r++ {
					dot += col[r] * uData[r*m+j]
				}
				for r := 0; r < m; r++ {
					col[r] -= dot * uData[r*m+j]
				}
			}
			// Normalize.
			norm := 0.0
			for r := 0; r < m; r++ {
				norm += col[r] * col[r]
			}
			norm = math.Sqrt(norm)
			if norm > 1e-14 {
				for r := 0; r < m; r++ {
					uData[r*m+i] = col[r] / norm
				}
			}
		}
	}
	u = NewNDArray([]int{m, m}, uData)

	return u, s, vt, nil
}

// Cholesky computes the Cholesky decomposition of a symmetric positive-definite matrix.
// Returns the lower-triangular matrix L such that A = L * L^T.
func Cholesky(a *NDArray) (*NDArray, error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("numgo.Cholesky: input must be a square 2D matrix")
	}
	n := a.shape[0]
	l := Zeros(n, n)

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += l.Get(i, k) * l.Get(j, k)
			}
			if i == j {
				val := a.Get(i, i) - sum
				if val < 0 {
					return nil, fmt.Errorf("numgo.Cholesky: matrix is not positive definite")
				}
				l.Set(math.Sqrt(val), i, j)
			} else {
				ljj := l.Get(j, j)
				if math.Abs(ljj) < 1e-14 {
					return nil, fmt.Errorf("numgo.Cholesky: matrix is not positive definite")
				}
				l.Set((a.Get(i, j)-sum)/ljj, i, j)
			}
		}
	}
	return l, nil
}

// QR computes the QR factorization of a 2D matrix using the Gram-Schmidt process.
// Returns Q (orthogonal) and R (upper triangular) such that A = Q * R.
func QR(a *NDArray) (q, r *NDArray, err error) {
	if a.Ndim() != 2 {
		return nil, nil, fmt.Errorf("numgo.QR: input must be 2D")
	}
	m, n := a.shape[0], a.shape[1]

	qData := make([]float64, m*n)
	rData := make([]float64, n*n)

	// Copy columns of A.
	cols := make([][]float64, n)
	for j := 0; j < n; j++ {
		cols[j] = make([]float64, m)
		for i := 0; i < m; i++ {
			cols[j][i] = a.Get(i, j)
		}
	}

	// Modified Gram-Schmidt.
	qCols := make([][]float64, n)
	for j := 0; j < n; j++ {
		qCols[j] = make([]float64, m)
		copy(qCols[j], cols[j])

		for k := 0; k < j; k++ {
			// r[k][j] = qCols[k] . cols[j]
			dot := 0.0
			for i := 0; i < m; i++ {
				dot += qCols[k][i] * qCols[j][i]
			}
			rData[k*n+j] = dot
			for i := 0; i < m; i++ {
				qCols[j][i] -= dot * qCols[k][i]
			}
		}

		// r[j][j] = ||qCols[j]||
		norm := 0.0
		for i := 0; i < m; i++ {
			norm += qCols[j][i] * qCols[j][i]
		}
		norm = math.Sqrt(norm)
		rData[j*n+j] = norm

		if norm > 1e-14 {
			for i := 0; i < m; i++ {
				qCols[j][i] /= norm
			}
		}
	}

	// Build Q matrix.
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			qData[i*n+j] = qCols[j][i]
		}
	}

	q = NewNDArray([]int{m, n}, qData)
	r = NewNDArray([]int{n, n}, rData)
	return q, r, nil
}

// Lstsq finds the least squares solution to Ax = b via the normal equations (A^T A) x = A^T b.
func Lstsq(a, b *NDArray) (*NDArray, error) {
	if a.Ndim() != 2 {
		return nil, fmt.Errorf("numgo.Lstsq: a must be 2D")
	}
	at := a.T()
	ata, err := Matmul(at, a)
	if err != nil {
		return nil, err
	}
	atb, err2 := Dot(at, b)
	if err2 != nil {
		return nil, err2
	}
	return Solve(ata, atb)
}

// Norm computes a vector or matrix norm.
//
//	ord=1: sum of absolute values (or max column sum for matrices)
//	ord=2: Euclidean norm (or spectral norm for matrices)
//	ord=-1: for vectors, min of absolute values
//
// axis=-1 means compute over the flattened array.
func Norm(a *NDArray, ord int, axis int) (*NDArray, error) {
	if axis == -1 {
		// Flatten and compute vector norm.
		flat := a.Flatten()
		var val float64
		switch ord {
		case 1:
			for _, v := range flat.data {
				val += math.Abs(v)
			}
		case 2:
			for _, v := range flat.data {
				val += v * v
			}
			val = math.Sqrt(val)
		case 0:
			// Number of non-zero elements.
			for _, v := range flat.data {
				if v != 0 {
					val++
				}
			}
		default:
			p := float64(ord)
			for _, v := range flat.data {
				val += math.Pow(math.Abs(v), p)
			}
			val = math.Pow(val, 1.0/p)
		}
		return NewNDArray([]int{1}, []float64{val}), nil
	}
	return nil, fmt.Errorf("numgo.Norm: axis-specific norms not yet implemented")
}

// Cond computes the condition number of a matrix (ratio of largest to smallest singular value).
func Cond(a *NDArray) (float64, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	sData := s.Data()
	if len(sData) == 0 {
		return math.Inf(1), nil
	}
	maxS := sData[0]
	minS := sData[0]
	for _, v := range sData {
		if v > maxS {
			maxS = v
		}
		if v < minS {
			minS = v
		}
	}
	if minS < 1e-14 {
		return math.Inf(1), nil
	}
	return maxS / minS, nil
}

// MatrixRank computes the rank of a matrix by counting non-negligible singular values.
func MatrixRank(a *NDArray) (int, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	tol := 1e-10 * float64(max(a.shape[0], a.shape[1]))
	sData := s.Data()
	// Scale tolerance by largest singular value.
	if len(sData) > 0 && sData[0] > 0 {
		tol *= sData[0]
	}
	rank := 0
	for _, v := range sData {
		if v > tol {
			rank++
		}
	}
	return rank, nil
}

// MatrixPower computes A^n for a square matrix via repeated multiplication.
// n=0 returns identity, negative n uses the inverse.
func MatrixPower(a *NDArray, n int) (*NDArray, error) {
	if a.Ndim() != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("numgo.MatrixPower: input must be a square 2D matrix")
	}
	sz := a.shape[0]

	if n == 0 {
		return Eye(sz), nil
	}

	base := a
	if n < 0 {
		inv, err := Inv(a)
		if err != nil {
			return nil, err
		}
		base = inv
		n = -n
	}

	result := Eye(sz)
	temp := base.Copy()
	for n > 0 {
		if n%2 == 1 {
			r, err := Matmul(result, temp)
			if err != nil {
				return nil, err
			}
			result = r
		}
		if n > 1 {
			t, err := Matmul(temp, temp)
			if err != nil {
				return nil, err
			}
			temp = t
		}
		n /= 2
	}
	return result, nil
}

// Pinv computes the Moore-Penrose pseudo-inverse via SVD.
func Pinv(a *NDArray) (*NDArray, error) {
	u, s, vt, err := SVD(a)
	if err != nil {
		return nil, err
	}
	m, n := a.shape[0], a.shape[1]
	sData := s.Data()

	// Build S^+ (pseudo-inverse of singular values).
	tol := 1e-10 * float64(max(m, n))
	if len(sData) > 0 && sData[0] > 0 {
		tol *= sData[0]
	}

	// pinv = V * S^+ * U^T
	// V = Vt^T, dimensions: n x n
	// S^+: diagonal, dimensions min(m,n)
	// U^T: m x m
	v := vt.T()
	ut := u.T()

	minDim := len(sData)

	// Compute V * diag(1/s_i) * U^T
	resultData := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			sum := 0.0
			for k := 0; k < minDim; k++ {
				if sData[k] > tol {
					sum += v.Get(i, k) * (1.0 / sData[k]) * ut.Get(k, j)
				}
			}
			resultData[i*m+j] = sum
		}
	}
	return NewNDArray([]int{n, m}, resultData), nil
}

// Slogdet computes the sign and natural logarithm of the determinant of a square matrix.
func Slogdet(a *NDArray) (sign float64, logdet float64, err error) {
	det, err := Det(a)
	if err != nil {
		return 0, 0, err
	}
	if det == 0 {
		return 0, math.Inf(-1), nil
	}
	if det > 0 {
		return 1, math.Log(det), nil
	}
	return -1, math.Log(-det), nil
}

// Trace returns the sum of the diagonal elements of a 2D matrix.
func Trace(a *NDArray) (float64, error) {
	if a.Ndim() != 2 {
		return 0, fmt.Errorf("numgo.Trace: input must be 2D")
	}
	m, n := a.shape[0], a.shape[1]
	sum := 0.0
	minDim := m
	if n < m {
		minDim = n
	}
	for i := 0; i < minDim; i++ {
		sum += a.Get(i, i)
	}
	return sum, nil
}

// Cross computes the cross product of two 3-element vectors.
func Cross(a, b *NDArray) (*NDArray, error) {
	if a.Size() != 3 || b.Size() != 3 {
		return nil, fmt.Errorf("numgo.Cross: both arrays must have exactly 3 elements")
	}
	ad := a.data
	bd := b.data
	return NewNDArray([]int{3}, []float64{
		ad[1]*bd[2] - ad[2]*bd[1],
		ad[2]*bd[0] - ad[0]*bd[2],
		ad[0]*bd[1] - ad[1]*bd[0],
	}), nil
}
