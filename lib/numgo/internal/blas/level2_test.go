//go:build unit

package blas

import (
	"math"
	"testing"
)

// --- Dgemv ---

func TestDgemv_NoTrans_Basic(t *testing.T) {
	// A = [[1,2],[3,4]], x = [1,1], y = [0,0]
	// y = 1.0 * A * x + 0 * y = [3, 7]
	a := []float64{1, 2, 3, 4}
	x := []float64{1, 1}
	y := []float64{0, 0}
	Dgemv(false, 2, 2, 1, a, 2, x, 1, 0, y, 1)
	if !approxEq(y[0], 3, tol) || !approxEq(y[1], 7, tol) {
		t.Errorf("Dgemv no trans = %v, want [3, 7]", y)
	}
}

func TestDgemv_NoTrans_AlphaBeta(t *testing.T) {
	// A = [[1,2],[3,4]], x = [2,3], y = [1,1]
	// y = 2 * A * x + 3 * y = 2*[8,18] + 3*[1,1] = [19, 39]
	a := []float64{1, 2, 3, 4}
	x := []float64{2, 3}
	y := []float64{1, 1}
	Dgemv(false, 2, 2, 2, a, 2, x, 1, 3, y, 1)
	if !approxEq(y[0], 19, tol) || !approxEq(y[1], 39, tol) {
		t.Errorf("Dgemv alpha/beta = %v, want [19, 39]", y)
	}
}

func TestDgemv_Trans(t *testing.T) {
	// A = [[1,2],[3,4]], trans => A^T = [[1,3],[2,4]], x = [1,1]
	// y = 1 * A^T * x + 0 * y = [4, 6]
	a := []float64{1, 2, 3, 4}
	x := []float64{1, 1}
	y := []float64{0, 0}
	Dgemv(true, 2, 2, 1, a, 2, x, 1, 0, y, 1)
	if !approxEq(y[0], 4, tol) || !approxEq(y[1], 6, tol) {
		t.Errorf("Dgemv trans = %v, want [4, 6]", y)
	}
}

func TestDgemv_Rectangular(t *testing.T) {
	// A = [[1,2,3],[4,5,6]] (2x3), x = [1,1,1], y = [0,0]
	// y = A * x = [6, 15]
	a := []float64{1, 2, 3, 4, 5, 6}
	x := []float64{1, 1, 1}
	y := []float64{0, 0}
	Dgemv(false, 2, 3, 1, a, 3, x, 1, 0, y, 1)
	if !approxEq(y[0], 6, tol) || !approxEq(y[1], 15, tol) {
		t.Errorf("Dgemv rect = %v, want [6, 15]", y)
	}
}

func TestDgemv_TransRectangular(t *testing.T) {
	// A = [[1,2,3],[4,5,6]] (2x3), trans A^T is 3x2
	// x = [1,1] (length m=2), y = [0,0,0] (length n=3)
	// y = A^T * x = [5, 7, 9]
	a := []float64{1, 2, 3, 4, 5, 6}
	x := []float64{1, 1}
	y := []float64{0, 0, 0}
	Dgemv(true, 2, 3, 1, a, 3, x, 1, 0, y, 1)
	if !approxEq(y[0], 5, tol) || !approxEq(y[1], 7, tol) || !approxEq(y[2], 9, tol) {
		t.Errorf("Dgemv trans rect = %v, want [5, 7, 9]", y)
	}
}

func TestDgemv_BetaOne(t *testing.T) {
	a := []float64{1, 0, 0, 1}
	x := []float64{3, 4}
	y := []float64{1, 2}
	Dgemv(false, 2, 2, 1, a, 2, x, 1, 1, y, 1)
	if !approxEq(y[0], 4, tol) || !approxEq(y[1], 6, tol) {
		t.Errorf("Dgemv beta=1: %v, want [4,6]", y)
	}
}

func TestDgemv_AlphaZero(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	x := []float64{1, 1}
	y := []float64{5, 6}
	Dgemv(false, 2, 2, 0, a, 2, x, 1, 2, y, 1)
	// y = 0*A*x + 2*y = [10, 12]
	if !approxEq(y[0], 10, tol) || !approxEq(y[1], 12, tol) {
		t.Errorf("Dgemv alpha=0: %v, want [10,12]", y)
	}
}

func TestDgemv_ZeroDimension(t *testing.T) {
	y := []float64{5}
	Dgemv(false, 0, 1, 1, nil, 1, nil, 1, 0, y, 1)
	if y[0] != 5 {
		t.Errorf("Dgemv zero dim modified y: %v", y)
	}
}

func TestDgemv_Stride(t *testing.T) {
	// A = [[1,2],[3,4]], x = [1, _, 1] (incx=2), y = [0, _, 0] (incy=2)
	a := []float64{1, 2, 3, 4}
	x := []float64{1, 99, 1}
	y := []float64{0, 99, 0}
	Dgemv(false, 2, 2, 1, a, 2, x, 2, 0, y, 2)
	if !approxEq(y[0], 3, tol) || y[1] != 99 || !approxEq(y[2], 7, tol) {
		t.Errorf("Dgemv stride = %v, want [3, 99, 7]", y)
	}
}

func TestDgemv_TransWithZeroX(t *testing.T) {
	// Test trans path where x[ix] == 0 (skip branch).
	a := []float64{1, 2, 3, 4}
	x := []float64{0, 1}
	y := []float64{0, 0}
	Dgemv(true, 2, 2, 1, a, 2, x, 1, 0, y, 1)
	// A^T * [0,1] = [3, 4]
	if !approxEq(y[0], 3, tol) || !approxEq(y[1], 4, tol) {
		t.Errorf("Dgemv trans zero x = %v, want [3, 4]", y)
	}
}

// --- Dtrsv ---

func TestDtrsv_UpperNoTrans(t *testing.T) {
	// U = [[2, 1], [0, 3]], b = [5, 6]
	// 3*x2 = 6 => x2=2, 2*x1 + 1*2 = 5 => x1=1.5
	a := []float64{2, 1, 0, 3}
	x := []float64{5, 6}
	Dtrsv('U', 'N', 'N', 2, a, 2, x, 1)
	if !approxEq(x[0], 1.5, tol) || !approxEq(x[1], 2, tol) {
		t.Errorf("Dtrsv upper = %v, want [1.5, 2]", x)
	}
}

func TestDtrsv_LowerNoTrans(t *testing.T) {
	// L = [[2, 0], [1, 3]], b = [4, 7]
	// 2*x1 = 4 => x1=2, 1*2 + 3*x2 = 7 => x2=5/3
	a := []float64{2, 0, 1, 3}
	x := []float64{4, 7}
	Dtrsv('L', 'N', 'N', 2, a, 2, x, 1)
	if !approxEq(x[0], 2, tol) || !approxEq(x[1], 5.0/3.0, tol) {
		t.Errorf("Dtrsv lower = %v, want [2, 5/3]", x)
	}
}

func TestDtrsv_UpperTrans(t *testing.T) {
	// U = [[2, 1], [0, 3]], solve U^T x = b, b = [4, 7]
	// U^T = [[2, 0], [1, 3]]
	// 2*x1 = 4 => x1=2, 1*2 + 3*x2 = 7 => x2=5/3
	a := []float64{2, 1, 0, 3}
	x := []float64{4, 7}
	Dtrsv('U', 'T', 'N', 2, a, 2, x, 1)
	if !approxEq(x[0], 2, tol) || !approxEq(x[1], 5.0/3.0, tol) {
		t.Errorf("Dtrsv upper trans = %v, want [2, 5/3]", x)
	}
}

func TestDtrsv_LowerTrans(t *testing.T) {
	// L = [[2, 0], [1, 3]], solve L^T x = b, b = [5, 6]
	// L^T = [[2, 1], [0, 3]]
	// 3*x2 = 6 => x2=2, 2*x1 + 1*2 = 5 => x1=1.5
	a := []float64{2, 0, 1, 3}
	x := []float64{5, 6}
	Dtrsv('L', 'T', 'N', 2, a, 2, x, 1)
	if !approxEq(x[0], 1.5, tol) || !approxEq(x[1], 2, tol) {
		t.Errorf("Dtrsv lower trans = %v, want [1.5, 2]", x)
	}
}

func TestDtrsv_UnitDiag(t *testing.T) {
	// L = [[1, 0], [2, 1]] (unit diagonal), b = [3, 8]
	// x1 = 3, 2*3 + x2 = 8 => x2 = 2
	a := []float64{1, 0, 2, 1}
	x := []float64{3, 8}
	Dtrsv('L', 'N', 'U', 2, a, 2, x, 1)
	if !approxEq(x[0], 3, tol) || !approxEq(x[1], 2, tol) {
		t.Errorf("Dtrsv unit diag = %v, want [3, 2]", x)
	}
}

func TestDtrsv_ZeroN(t *testing.T) {
	x := []float64{1}
	Dtrsv('U', 'N', 'N', 0, nil, 1, x, 1)
	if x[0] != 1 {
		t.Errorf("Dtrsv n=0 modified x: %v", x)
	}
}

func TestDtrsv_Stride(t *testing.T) {
	// U = [[2, 1], [0, 3]], b stored with stride 2: [5, _, 6]
	a := []float64{2, 1, 0, 3}
	x := []float64{5, 99, 6}
	Dtrsv('U', 'N', 'N', 2, a, 2, x, 2)
	if !approxEq(x[0], 1.5, tol) || x[1] != 99 || !approxEq(x[2], 2, tol) {
		t.Errorf("Dtrsv stride = %v, want [1.5, 99, 2]", x)
	}
}

func TestDtrsv_3x3_Lower(t *testing.T) {
	// L = [[1,0,0],[2,3,0],[4,5,6]], b = [1, 8, 32]
	// x1 = 1, 2*1 + 3*x2 = 8 => x2=2, 4*1+5*2+6*x3 = 32 => x3=3
	a := []float64{1, 0, 0, 2, 3, 0, 4, 5, 6}
	x := []float64{1, 8, 32}
	Dtrsv('L', 'N', 'N', 3, a, 3, x, 1)
	if !approxEq(x[0], 1, tol) || !approxEq(x[1], 2, tol) || !approxEq(x[2], 3, tol) {
		t.Errorf("Dtrsv 3x3 lower = %v, want [1, 2, 3]", x)
	}
}

func TestDtrsv_3x3_Upper(t *testing.T) {
	// U = [[1,2,3],[0,4,5],[0,0,6]], b = [14, 23, 18]
	// 6*x3 = 18 => x3=3, 4*x2+5*3=23 => x2=2, 1*x1+2*2+3*3=14 => x1=1
	a := []float64{1, 2, 3, 0, 4, 5, 0, 0, 6}
	x := []float64{14, 23, 18}
	Dtrsv('U', 'N', 'N', 3, a, 3, x, 1)
	if !approxEq(x[0], 1, tol) || !approxEq(x[1], 2, tol) || !approxEq(x[2], 3, tol) {
		t.Errorf("Dtrsv 3x3 upper = %v, want [1, 2, 3]", x)
	}
}

func TestDtrsv_UpperUnitDiag(t *testing.T) {
	// U = [[1, 2], [0, 1]], unit diag, b = [7, 3]
	// x2 = 3, x1 + 2*3 = 7 => x1=1
	a := []float64{1, 2, 0, 1}
	x := []float64{7, 3}
	Dtrsv('U', 'N', 'U', 2, a, 2, x, 1)
	if !approxEq(x[0], 1, tol) || !approxEq(x[1], 3, tol) {
		t.Errorf("Dtrsv upper unit = %v, want [1, 3]", x)
	}
}

func TestDtrsv_LowerTransUnitDiag(t *testing.T) {
	// L = [[1, 0], [2, 1]], unit diag, solve L^T x = b, b = [5, 3]
	// L^T = [[1, 2], [0, 1]]
	// x2 = 3, x1 + 2*3 = 5 => x1 = -1
	a := []float64{1, 0, 2, 1}
	x := []float64{5, 3}
	Dtrsv('L', 'T', 'U', 2, a, 2, x, 1)
	if !approxEq(x[0], -1, tol) || !approxEq(x[1], 3, tol) {
		t.Errorf("Dtrsv lower trans unit = %v, want [-1, 3]", x)
	}
}

func TestDtrsv_UpperTransUnitDiag(t *testing.T) {
	// U = [[1, 3], [0, 1]], unit diag, solve U^T x = b, b = [2, 10]
	// U^T = [[1, 0], [3, 1]]
	// x1 = 2, 3*2 + x2 = 10 => x2 = 4
	a := []float64{1, 3, 0, 1}
	x := []float64{2, 10}
	Dtrsv('U', 'T', 'U', 2, a, 2, x, 1)
	if !approxEq(x[0], 2, tol) || !approxEq(x[1], 4, tol) {
		t.Errorf("Dtrsv upper trans unit = %v, want [2, 4]", x)
	}
}

// TestDtrsv_RoundTrip verifies Dtrsv by multiplying the solution back.
func TestDtrsv_RoundTrip(t *testing.T) {
	n := 5
	// Build a random-ish lower triangular matrix.
	a := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			a[i*n+j] = float64((i+1)*(j+1) + 1)
		}
	}
	// Original b.
	bOrig := make([]float64, n)
	for i := 0; i < n; i++ {
		bOrig[i] = float64(i*3 + 1)
	}
	x := make([]float64, n)
	copy(x, bOrig)
	Dtrsv('L', 'N', 'N', n, a, n, x, 1)

	// Verify: A * x should equal bOrig.
	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j <= i; j++ {
			sum += a[i*n+j] * x[j]
		}
		if !approxEq(sum, bOrig[i], 1e-8) {
			t.Errorf("round trip row %d: got %v, want %v", i, sum, bOrig[i])
		}
	}
}

func TestDgemv_LargeNoTrans(t *testing.T) {
	// 4x4 identity times [1,2,3,4] = [1,2,3,4]
	n := 4
	a := make([]float64, n*n)
	for i := 0; i < n; i++ {
		a[i*n+i] = 1
	}
	x := []float64{1, 2, 3, 4}
	y := make([]float64, n)
	Dgemv(false, n, n, 1, a, n, x, 1, 0, y, 1)
	for i := 0; i < n; i++ {
		if !approxEq(y[i], x[i], tol) {
			t.Errorf("Dgemv identity [%d] = %v, want %v", i, y[i], x[i])
		}
	}
}

// TestDgemv_VerifyAgainstNaive checks Dgemv against a naive implementation.
func TestDgemv_VerifyAgainstNaive(t *testing.T) {
	m, n := 7, 5
	a := make([]float64, m*n)
	for i := range a {
		a[i] = float64(i%7 + 1)
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
	}

	// Naive.
	want := make([]float64, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			want[i] += a[i*n+j] * x[j]
		}
		want[i] *= 2.5
		want[i] += 1.5 * float64(i+1)
	}

	y := make([]float64, m)
	for i := range y {
		y[i] = float64(i + 1)
	}
	Dgemv(false, m, n, 2.5, a, n, x, 1, 1.5, y, 1)

	for i := 0; i < m; i++ {
		if math.Abs(y[i]-want[i]) > 1e-10 {
			t.Errorf("Dgemv verify [%d] = %v, want %v", i, y[i], want[i])
		}
	}
}
