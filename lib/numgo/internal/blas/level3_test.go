//go:build unit

package blas

import (
	"math"
	"testing"
)

// naiveMatmul computes C = alpha*A*B + beta*C using a naive triple loop.
func naiveMatmul(m, n, k int, alpha float64, a []float64, lda int,
	b []float64, ldb int, beta float64, c []float64, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i*ldc+j] *= beta
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

func TestDgemm_Basic2x2(t *testing.T) {
	// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
	// C = A*B = [[19,22],[43,50]]
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	c := make([]float64, 4)
	Dgemm(false, false, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm 2x2 [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_Identity(t *testing.T) {
	n := 4
	eye := make([]float64, n*n)
	for i := 0; i < n; i++ {
		eye[i*n+i] = 1
	}
	a := make([]float64, n*n)
	for i := range a {
		a[i] = float64(i + 1)
	}
	c := make([]float64, n*n)
	Dgemm(false, false, n, n, n, 1, a, n, eye, n, 0, c, n)
	for i := range a {
		if !approxEq(c[i], a[i], tol) {
			t.Errorf("Dgemm identity [%d] = %v, want %v", i, c[i], a[i])
		}
	}
}

func TestDgemm_AlphaBeta(t *testing.T) {
	// C = 2*A*B + 3*C
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	c := []float64{1, 1, 1, 1}
	Dgemm(false, false, 2, 2, 2, 2, a, 2, b, 2, 3, c, 2)
	// 2*[19,22,43,50] + 3*[1,1,1,1] = [41, 47, 89, 103]
	want := []float64{41, 47, 89, 103}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm alpha/beta [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_TransA(t *testing.T) {
	// A stored as [[1,3],[2,4]] (transposed), effective A = [[1,2],[3,4]]
	// B = [[5,6],[7,8]]
	a := []float64{1, 3, 2, 4}
	b := []float64{5, 6, 7, 8}
	c := make([]float64, 4)
	Dgemm(true, false, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm transA [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_TransB(t *testing.T) {
	// A = [[1,2],[3,4]], B stored as [[5,7],[6,8]] (transposed), effective B = [[5,6],[7,8]]
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 7, 6, 8}
	c := make([]float64, 4)
	Dgemm(false, true, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm transB [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_TransAB(t *testing.T) {
	// Both transposed.
	a := []float64{1, 3, 2, 4}
	b := []float64{5, 7, 6, 8}
	c := make([]float64, 4)
	Dgemm(true, true, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm transAB [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_Rectangular(t *testing.T) {
	// A (2x3) * B (3x2) = C (2x2)
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 10, 11, 12}
	c := make([]float64, 4)
	Dgemm(false, false, 2, 2, 3, 1, a, 3, b, 2, 0, c, 2)
	// [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm rect [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_BetaZero(t *testing.T) {
	c := []float64{100, 200, 300, 400}
	a := []float64{1, 0, 0, 1}
	b := []float64{5, 6, 7, 8}
	Dgemm(false, false, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	want := []float64{5, 6, 7, 8}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm beta=0 [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_AlphaZero(t *testing.T) {
	c := []float64{1, 2, 3, 4}
	a := []float64{5, 6, 7, 8}
	b := []float64{9, 10, 11, 12}
	Dgemm(false, false, 2, 2, 2, 0, a, 2, b, 2, 2, c, 2)
	want := []float64{2, 4, 6, 8}
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm alpha=0 [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_ZeroDimension(t *testing.T) {
	c := []float64{42}
	Dgemm(false, false, 0, 1, 1, 1, nil, 1, nil, 1, 0, c, 1)
	if c[0] != 42 {
		t.Errorf("Dgemm zero dim modified c: %v", c)
	}
}

// TestDgemm_Medium tests with a matrix large enough to exercise cache blocking.
func TestDgemm_Medium(t *testing.T) {
	n := 33 // Slightly larger than Level3Threshold to test fringe handling.
	a := make([]float64, n*n)
	b := make([]float64, n*n)
	cBlas := make([]float64, n*n)
	cNaive := make([]float64, n*n)

	for i := range a {
		a[i] = float64(i%7) + 1
		b[i] = float64(i%5) + 1
	}

	Dgemm(false, false, n, n, n, 1, a, n, b, n, 0, cBlas, n)
	naiveMatmul(n, n, n, 1, a, n, b, n, 0, cNaive, n)

	for i := 0; i < n*n; i++ {
		if math.Abs(cBlas[i]-cNaive[i]) > 1e-8 {
			t.Errorf("Dgemm medium [%d] = %v, want %v", i, cBlas[i], cNaive[i])
			break
		}
	}
}

// TestDgemm_Large exercises all cache blocking and micro-kernel paths.
func TestDgemm_Large(t *testing.T) {
	n := 100
	a := make([]float64, n*n)
	b := make([]float64, n*n)
	cBlas := make([]float64, n*n)
	cNaive := make([]float64, n*n)

	for i := range a {
		a[i] = float64(i%11) - 5
		b[i] = float64(i%13) - 6
	}

	Dgemm(false, false, n, n, n, 2.5, a, n, b, n, 0, cBlas, n)
	naiveMatmul(n, n, n, 2.5, a, n, b, n, 0, cNaive, n)

	for i := 0; i < n*n; i++ {
		if math.Abs(cBlas[i]-cNaive[i]) > 1e-6 {
			t.Errorf("Dgemm large [%d] = %v, want %v", i, cBlas[i], cNaive[i])
			break
		}
	}
}

// TestDgemm_LargeTransA tests transposed A with cache blocking.
func TestDgemm_LargeTransA(t *testing.T) {
	m, n, k := 50, 60, 70
	a := make([]float64, k*m) // stored as k x m (transposed)
	b := make([]float64, k*n)
	cBlas := make([]float64, m*n)
	cNaive := make([]float64, m*n)

	for i := range a {
		a[i] = float64(i%9) - 4
	}
	for i := range b {
		b[i] = float64(i%7) - 3
	}

	Dgemm(true, false, m, n, k, 1, a, m, b, n, 0, cBlas, n)

	// Naive: explicit transpose.
	aT := make([]float64, m*k)
	for i := 0; i < k; i++ {
		for j := 0; j < m; j++ {
			aT[j*k+i] = a[i*m+j]
		}
	}
	naiveMatmul(m, n, k, 1, aT, k, b, n, 0, cNaive, n)

	for i := 0; i < m*n; i++ {
		if math.Abs(cBlas[i]-cNaive[i]) > 1e-6 {
			t.Errorf("Dgemm transA large [%d] = %v, want %v", i, cBlas[i], cNaive[i])
			break
		}
	}
}

// TestDgemm_LargeTransB tests transposed B with cache blocking.
func TestDgemm_LargeTransB(t *testing.T) {
	m, n, k := 50, 60, 70
	a := make([]float64, m*k)
	b := make([]float64, n*k) // stored as n x k (transposed)
	cBlas := make([]float64, m*n)
	cNaive := make([]float64, m*n)

	for i := range a {
		a[i] = float64(i%9) - 4
	}
	for i := range b {
		b[i] = float64(i%7) - 3
	}

	Dgemm(false, true, m, n, k, 1, a, k, b, k, 0, cBlas, n)

	// Naive: explicit transpose B.
	bT := make([]float64, k*n)
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			bT[j*n+i] = b[i*k+j]
		}
	}
	naiveMatmul(m, n, k, 1, a, k, bT, n, 0, cNaive, n)

	for i := 0; i < m*n; i++ {
		if math.Abs(cBlas[i]-cNaive[i]) > 1e-6 {
			t.Errorf("Dgemm transB large [%d] = %v, want %v", i, cBlas[i], cNaive[i])
			break
		}
	}
}

func TestDgemm_BetaScaling(t *testing.T) {
	n := 20
	a := make([]float64, n*n)
	b := make([]float64, n*n)
	c := make([]float64, n*n)
	for i := range a {
		a[i] = float64(i%5) + 1
		b[i] = float64(i%3) + 1
		c[i] = float64(i % 4)
	}
	cOrig := make([]float64, n*n)
	copy(cOrig, c)

	Dgemm(false, false, n, n, n, 1, a, n, b, n, 2.5, c, n)

	// Verify against naive with same beta.
	cRef := make([]float64, n*n)
	copy(cRef, cOrig)
	naiveMatmul(n, n, n, 1, a, n, b, n, 2.5, cRef, n)

	for i := 0; i < n*n; i++ {
		if math.Abs(c[i]-cRef[i]) > 1e-8 {
			t.Errorf("Dgemm beta scaling [%d] = %v, want %v", i, c[i], cRef[i])
			break
		}
	}
}

func TestDgemm_1x1(t *testing.T) {
	a := []float64{3}
	b := []float64{4}
	c := []float64{0}
	Dgemm(false, false, 1, 1, 1, 1, a, 1, b, 1, 0, c, 1)
	if !approxEq(c[0], 12, tol) {
		t.Errorf("Dgemm 1x1 = %v, want 12", c[0])
	}
}

func TestDgemm_SmallPath(t *testing.T) {
	// 3x3 to exercise dgemmSmall (m<=mr && n<=nr).
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := []float64{9, 8, 7, 6, 5, 4, 3, 2, 1}
	c := make([]float64, 9)
	Dgemm(false, false, 3, 3, 3, 1, a, 3, b, 3, 0, c, 3)
	want := make([]float64, 9)
	naiveMatmul(3, 3, 3, 1, a, 3, b, 3, 0, want, 3)
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm small [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}

func TestDgemm_NonSquareSmall(t *testing.T) {
	// 2x3 * 3x4 = 2x4
	m, n, k := 2, 4, 3
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	c := make([]float64, m*n)
	Dgemm(false, false, m, n, k, 1, a, k, b, n, 0, c, n)
	want := make([]float64, m*n)
	naiveMatmul(m, n, k, 1, a, k, b, n, 0, want, n)
	for i := range want {
		if !approxEq(c[i], want[i], tol) {
			t.Errorf("Dgemm non-square small [%d] = %v, want %v", i, c[i], want[i])
		}
	}
}
