//go:build unit

package numgo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// 1. solveBLAS — 64x64 system (Level2Threshold=32, so n=64 triggers BLAS path)
// ---------------------------------------------------------------------------

func TestSolveBLAS_64x64(t *testing.T) {
	n := 64
	// Build a diagonally-dominant matrix so the system is well-conditioned.
	aData := make([]float64, n*n)
	bData := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				aData[i*n+j] = float64(n) + 1.0
			} else {
				aData[i*n+j] = 1.0 / float64(1+iabs(i-j))
			}
		}
		bData[i] = float64(i + 1)
	}
	A := NewNDArray([]int{n, n}, aData)
	b := NewNDArray([]int{n}, bData)

	x, err := Solve(A, b)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	// Verify Ax ≈ b.
	Ax, err := Dot(A, x)
	if err != nil {
		t.Fatalf("Dot failed: %v", err)
	}
	for i := 0; i < n; i++ {
		if diff := math.Abs(Ax.data[i] - bData[i]); diff > 1e-8 {
			t.Fatalf("residual too large at index %d: got %g, want %g (diff=%g)", i, Ax.data[i], bData[i], diff)
		}
	}
}

// Test Solve with 2D rhs through the BLAS path.
func TestSolveBLAS_2D_RHS(t *testing.T) {
	n := 64
	aData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				aData[i*n+j] = float64(n) + 1.0
			} else {
				aData[i*n+j] = 0.5 / float64(1+iabs(i-j))
			}
		}
	}
	A := NewNDArray([]int{n, n}, aData)

	// 2D RHS: 2 columns
	nrhs := 2
	bData := make([]float64, n*nrhs)
	for i := 0; i < n; i++ {
		bData[i*nrhs+0] = float64(i + 1)
		bData[i*nrhs+1] = float64(n - i)
	}
	B := NewNDArray([]int{n, nrhs}, bData)

	X, err := Solve(A, B)
	if err != nil {
		t.Fatalf("Solve 2D failed: %v", err)
	}
	if X.shape[0] != n || X.shape[1] != nrhs {
		t.Fatalf("unexpected shape: %v", X.shape)
	}

	// Verify A*X ≈ B for first column
	for col := 0; col < nrhs; col++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				sum += A.Get(i, j) * X.Get(j, col)
			}
			want := B.Get(i, col)
			if diff := math.Abs(sum - want); diff > 1e-8 {
				t.Fatalf("col %d row %d: residual %g", col, i, diff)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 2. Large Dot — 1D vectors with >64 elements (Level1Threshold=64)
// ---------------------------------------------------------------------------

func TestDotLargeBLAS(t *testing.T) {
	n := 128
	aData := make([]float64, n)
	bData := make([]float64, n)
	expected := 0.0
	for i := 0; i < n; i++ {
		aData[i] = float64(i + 1)
		bData[i] = float64(n - i)
		expected += aData[i] * bData[i]
	}
	a := NewNDArray([]int{n}, aData)
	b := NewNDArray([]int{n}, bData)

	result, err := Dot(a, b)
	if err != nil {
		t.Fatalf("Dot failed: %v", err)
	}
	if diff := math.Abs(result.data[0] - expected); diff > 1e-8 {
		t.Fatalf("Dot large: got %g, want %g", result.data[0], expected)
	}
}

// ---------------------------------------------------------------------------
// 3. Large Matmul — 65x65 matrices (Level3Threshold=64)
// ---------------------------------------------------------------------------

func TestMatmulLargeBLAS(t *testing.T) {
	n := 65
	aData := make([]float64, n*n)
	bData := make([]float64, n*n)
	for i := 0; i < n*n; i++ {
		aData[i] = float64(i%7) + 1
		bData[i] = float64(i%5) + 1
	}
	A := NewNDArray([]int{n, n}, aData)
	B := NewNDArray([]int{n, n}, bData)

	C, err := Matmul(A, B)
	if err != nil {
		t.Fatalf("Matmul failed: %v", err)
	}

	// Verify a few entries by naive computation.
	for _, rc := range [][2]int{{0, 0}, {1, 2}, {n - 1, n - 1}} {
		r, c := rc[0], rc[1]
		sum := 0.0
		for k := 0; k < n; k++ {
			sum += A.Get(r, k) * B.Get(k, c)
		}
		got := C.Get(r, c)
		if diff := math.Abs(got - sum); diff > 1e-8 {
			t.Fatalf("Matmul[%d,%d]: got %g, want %g", r, c, got, sum)
		}
	}
}

// ---------------------------------------------------------------------------
// 4. Eig — exercise convergence path and check eigenvalues
// ---------------------------------------------------------------------------

func TestEigSymmetric3x3(t *testing.T) {
	// Symmetric matrix with known eigenvalues: diag(1,2,3)
	// P*D*P^T where P is a simple rotation
	A := NewNDArray([]int{3, 3}, []float64{
		2, -1, 0,
		-1, 2, -1,
		0, -1, 2,
	})
	vals, vecs, err := Eig(A)
	if err != nil {
		t.Fatalf("Eig failed: %v", err)
	}
	if vals.Size() != 3 {
		t.Fatalf("expected 3 eigenvalues, got %d", vals.Size())
	}
	if vecs.shape[0] != 3 || vecs.shape[1] != 3 {
		t.Fatalf("unexpected eigenvector shape: %v", vecs.shape)
	}

	// Sort eigenvalues and check they are close to 2-sqrt(2), 2, 2+sqrt(2)
	evs := make([]float64, 3)
	copy(evs, vals.data)
	sortFloat64s(evs)
	expected := []float64{2 - math.Sqrt(2), 2, 2 + math.Sqrt(2)}
	for i, e := range expected {
		if diff := math.Abs(evs[i] - e); diff > 1e-6 {
			t.Errorf("eigenvalue %d: got %g, want %g", i, evs[i], e)
		}
	}
}

func TestEig1x1(t *testing.T) {
	A := NewNDArray([]int{1, 1}, []float64{42})
	vals, vecs, err := Eig(A)
	if err != nil {
		t.Fatalf("Eig failed: %v", err)
	}
	if math.Abs(vals.data[0]-42) > 1e-10 {
		t.Errorf("eigenvalue: got %g, want 42", vals.data[0])
	}
	if math.Abs(vecs.Get(0, 0)-1) > 1e-10 {
		t.Errorf("eigenvector: got %g, want 1", vecs.Get(0, 0))
	}
}

// ---------------------------------------------------------------------------
// 5. Lstsq — overdetermined system
// ---------------------------------------------------------------------------

func TestLstsqOverdetermined(t *testing.T) {
	// 4 equations, 2 unknowns: y = 2*x + 1
	A := NewNDArray([]int{4, 2}, []float64{
		1, 0,
		1, 1,
		1, 2,
		1, 3,
	})
	b := NewNDArray([]int{4}, []float64{1, 3, 5, 7})

	x, err := Lstsq(A, b)
	if err != nil {
		t.Fatalf("Lstsq failed: %v", err)
	}
	// x should be approximately [1, 2]
	if diff := math.Abs(x.data[0] - 1); diff > 1e-8 {
		t.Errorf("intercept: got %g, want 1", x.data[0])
	}
	if diff := math.Abs(x.data[1] - 2); diff > 1e-8 {
		t.Errorf("slope: got %g, want 2", x.data[1])
	}
}

func TestLstsqNoisy(t *testing.T) {
	// Noisy overdetermined system
	A := NewNDArray([]int{5, 2}, []float64{
		1, 1,
		1, 2,
		1, 3,
		1, 4,
		1, 5,
	})
	b := NewNDArray([]int{5}, []float64{2.1, 3.9, 6.2, 7.8, 10.1})

	x, err := Lstsq(A, b)
	if err != nil {
		t.Fatalf("Lstsq failed: %v", err)
	}
	// Rough check: slope should be near 2, intercept near 0
	if x.data[1] < 1 || x.data[1] > 3 {
		t.Errorf("slope out of range: %g", x.data[1])
	}
}

// ---------------------------------------------------------------------------
// 6. Cond — well-conditioned and ill-conditioned matrices
// ---------------------------------------------------------------------------

func TestCondIdentity4x4(t *testing.T) {
	I := Eye(4)
	c, err := Cond(I)
	if err != nil {
		t.Fatalf("Cond failed: %v", err)
	}
	if diff := math.Abs(c - 1.0); diff > 1e-6 {
		t.Errorf("Cond(I) = %g, want 1.0", c)
	}
}

func TestCondIllConditioned(t *testing.T) {
	// Near-singular matrix
	A := NewNDArray([]int{2, 2}, []float64{
		1, 1,
		1, 1.0001,
	})
	c, err := Cond(A)
	if err != nil {
		t.Fatalf("Cond failed: %v", err)
	}
	if c < 1000 {
		t.Errorf("expected large condition number, got %g", c)
	}
}

// ---------------------------------------------------------------------------
// 7. MatrixPower — negative power, power > 2
// ---------------------------------------------------------------------------

func TestMatrixPowerNegativeBLAS(t *testing.T) {
	A := NewNDArray([]int{2, 2}, []float64{
		1, 2,
		3, 4,
	})
	// A^-1
	Ainv, err := MatrixPower(A, -1)
	if err != nil {
		t.Fatalf("MatrixPower(-1) failed: %v", err)
	}
	// A * A^-1 should be identity
	prod, err := Matmul(A, Ainv)
	if err != nil {
		t.Fatalf("Matmul failed: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if diff := math.Abs(prod.Get(i, j) - expected); diff > 1e-10 {
				t.Errorf("A*A^-1[%d,%d] = %g, want %g", i, j, prod.Get(i, j), expected)
			}
		}
	}
}

func TestMatrixPowerLarge(t *testing.T) {
	// A^3 via binary exponentiation exercises the n>1 squaring path
	A := NewNDArray([]int{2, 2}, []float64{
		1, 1,
		0, 1,
	})
	A3, err := MatrixPower(A, 3)
	if err != nil {
		t.Fatalf("MatrixPower(3) failed: %v", err)
	}
	// [[1,1],[0,1]]^3 = [[1,3],[0,1]]
	if diff := math.Abs(A3.Get(0, 1) - 3); diff > 1e-10 {
		t.Errorf("A^3[0,1] = %g, want 3", A3.Get(0, 1))
	}
}

func TestMatrixPowerNeg2(t *testing.T) {
	A := NewNDArray([]int{2, 2}, []float64{
		2, 0,
		0, 3,
	})
	result, err := MatrixPower(A, -2)
	if err != nil {
		t.Fatalf("MatrixPower(-2) failed: %v", err)
	}
	// diag(2,3)^-2 = diag(0.25, 1/9)
	if diff := math.Abs(result.Get(0, 0) - 0.25); diff > 1e-10 {
		t.Errorf("got %g, want 0.25", result.Get(0, 0))
	}
	if diff := math.Abs(result.Get(1, 1) - 1.0/9.0); diff > 1e-10 {
		t.Errorf("got %g, want %g", result.Get(1, 1), 1.0/9.0)
	}
}

// ---------------------------------------------------------------------------
// 8. Hsplit — 2D case (exercises the axis=1 path at 66.7%)
// ---------------------------------------------------------------------------

func TestHsplit2D(t *testing.T) {
	A := NewNDArray([]int{2, 4}, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	parts, err := Hsplit(A, 2)
	if err != nil {
		t.Fatalf("Hsplit failed: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	// First part: [[1,2],[5,6]]
	if parts[0].Get(0, 0) != 1 || parts[0].Get(0, 1) != 2 {
		t.Errorf("first part row 0: %v", parts[0].data)
	}
	if parts[0].Get(1, 0) != 5 || parts[0].Get(1, 1) != 6 {
		t.Errorf("first part row 1: %v", parts[0].data)
	}
}

func TestHsplit2D_FourWay(t *testing.T) {
	A := NewNDArray([]int{3, 4}, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	parts, err := Hsplit(A, 4)
	if err != nil {
		t.Fatalf("Hsplit failed: %v", err)
	}
	if len(parts) != 4 {
		t.Fatalf("expected 4 parts, got %d", len(parts))
	}
	// Each part should be 3x1
	for i, p := range parts {
		if p.shape[0] != 3 || p.shape[1] != 1 {
			t.Errorf("part %d shape: %v", i, p.shape)
		}
	}
}

// ---------------------------------------------------------------------------
// 9. broadcastElementWise — different-shaped arrays via exported operations
// ---------------------------------------------------------------------------

func TestBroadcastElementWise_ScalarAndVector(t *testing.T) {
	// scalar (shape [1]) + vector (shape [3]) should broadcast
	a := NewNDArray([]int{1}, []float64{10})
	b := NewNDArray([]int{3}, []float64{1, 2, 3})
	result := Add(a, b)
	expected := []float64{11, 12, 13}
	for i, e := range expected {
		if result.data[i] != e {
			t.Errorf("index %d: got %g, want %g", i, result.data[i], e)
		}
	}
}

func TestBroadcastElementWise_2DWith1D(t *testing.T) {
	// (2,3) + (3,) => broadcasting
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := NewNDArray([]int{3}, []float64{10, 20, 30})
	result := Add(a, b)
	expected := []float64{11, 22, 33, 14, 25, 36}
	for i, e := range expected {
		if result.data[i] != e {
			t.Errorf("index %d: got %g, want %g", i, result.data[i], e)
		}
	}
}

func TestBroadcastElementWise_ColumnAndRow(t *testing.T) {
	// (3,1) * (1,4) => (3,4)
	a := NewNDArray([]int{3, 1}, []float64{1, 2, 3})
	b := NewNDArray([]int{1, 4}, []float64{10, 20, 30, 40})
	result := Mul(a, b)
	if result.shape[0] != 3 || result.shape[1] != 4 {
		t.Fatalf("unexpected shape: %v", result.shape)
	}
	if result.Get(2, 3) != 120 {
		t.Errorf("result[2,3] = %g, want 120", result.Get(2, 3))
	}
}

// ---------------------------------------------------------------------------
// 10. ArrayEquiv — various broadcasting equivalences
// ---------------------------------------------------------------------------

func TestArrayEquivScalarBroadcast(t *testing.T) {
	a := NewNDArray([]int{3}, []float64{5, 5, 5})
	b := NewNDArray([]int{1}, []float64{5})
	if !ArrayEquiv(a, b) {
		t.Error("expected ArrayEquiv to be true for [5,5,5] and [5]")
	}
}

func TestArrayEquivDifferent(t *testing.T) {
	a := NewNDArray([]int{3}, []float64{1, 2, 3})
	b := NewNDArray([]int{1}, []float64{5})
	if ArrayEquiv(a, b) {
		t.Error("expected ArrayEquiv to be false")
	}
}

func TestArrayEquivIncompatibleShapes(t *testing.T) {
	a := NewNDArray([]int{2}, []float64{1, 2})
	b := NewNDArray([]int{3}, []float64{1, 2, 3})
	if ArrayEquiv(a, b) {
		t.Error("expected ArrayEquiv to be false for incompatible shapes")
	}
}

func TestArrayEquiv2D(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 1, 2, 3})
	b := NewNDArray([]int{1, 3}, []float64{1, 2, 3})
	if !ArrayEquiv(a, b) {
		t.Error("expected ArrayEquiv true for (2,3) vs (1,3)")
	}
}

// ---------------------------------------------------------------------------
// 11. Solve large via BLAS with singular matrix (error path)
// ---------------------------------------------------------------------------

func TestSolveBLAS_Singular(t *testing.T) {
	n := 64
	// All-zero matrix is singular.
	aData := make([]float64, n*n)
	bData := make([]float64, n)
	bData[0] = 1
	A := NewNDArray([]int{n, n}, aData)
	b := NewNDArray([]int{n}, bData)

	_, err := Solve(A, b)
	if err == nil {
		t.Fatal("expected error for singular matrix")
	}
}

// ---------------------------------------------------------------------------
// 12. Large Matmul with non-square (cover rectangular BLAS path)
// ---------------------------------------------------------------------------

func TestMatmulLargeRectangular(t *testing.T) {
	m, k, n := 70, 65, 80
	aData := make([]float64, m*k)
	bData := make([]float64, k*n)
	for i := range aData {
		aData[i] = float64(i%11) + 1
	}
	for i := range bData {
		bData[i] = float64(i%7) + 1
	}
	A := NewNDArray([]int{m, k}, aData)
	B := NewNDArray([]int{k, n}, bData)

	C, err := Matmul(A, B)
	if err != nil {
		t.Fatalf("Matmul failed: %v", err)
	}

	// Spot-check a couple entries
	for _, rc := range [][2]int{{0, 0}, {m - 1, n - 1}} {
		r, c := rc[0], rc[1]
		sum := 0.0
		for l := 0; l < k; l++ {
			sum += A.Get(r, l) * B.Get(l, c)
		}
		if diff := math.Abs(C.Get(r, c) - sum); diff > 1e-6 {
			t.Errorf("C[%d,%d]: got %g, want %g", r, c, C.Get(r, c), sum)
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func iabs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func sortFloat64s(a []float64) {
	for i := 0; i < len(a); i++ {
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[i] {
				a[i], a[j] = a[j], a[i]
			}
		}
	}
}
