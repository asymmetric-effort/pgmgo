//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// LU Decomposition Tests
// ---------------------------------------------------------------------------

func TestLU(t *testing.T) {
	a := [][]float64{
		{2, 1, 1},
		{4, 3, 3},
		{8, 7, 9},
	}
	p, l, u, err := LU(a)
	if err != nil {
		t.Fatalf("LU: unexpected error: %v", err)
	}

	// Verify P*A = L*U.
	n := len(a)
	pa := matMul(p, a, n)
	lu := matMul(l, u, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(pa[i][j], lu[i][j], 1e-10) {
				t.Errorf("LU: P*A[%d][%d] = %v, L*U[%d][%d] = %v", i, j, pa[i][j], i, j, lu[i][j])
			}
		}
	}

	// Verify L is lower-triangular with ones on diagonal.
	for i := 0; i < n; i++ {
		if !approxEqual(l[i][i], 1, 1e-14) {
			t.Errorf("LU: L[%d][%d] = %v, want 1", i, i, l[i][i])
		}
		for j := i + 1; j < n; j++ {
			if l[i][j] != 0 {
				t.Errorf("LU: L[%d][%d] = %v, want 0", i, j, l[i][j])
			}
		}
	}

	// Verify U is upper-triangular.
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if !approxEqual(u[i][j], 0, 1e-10) {
				t.Errorf("LU: U[%d][%d] = %v, want 0", i, j, u[i][j])
			}
		}
	}
}

func TestLUFactor(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	lu, piv, err := LUFactor(a)
	if err != nil {
		t.Fatalf("LUFactor: unexpected error: %v", err)
	}
	if lu == nil || piv == nil {
		t.Fatal("LUFactor: nil result")
	}
}

func TestLUSolve(t *testing.T) {
	a := [][]float64{
		{2, 1, 1},
		{4, 3, 3},
		{8, 7, 9},
	}
	b := []float64{1, 1, 1}
	lu, piv, err := LUFactor(a)
	if err != nil {
		t.Fatalf("LUFactor: %v", err)
	}
	x, err := LUSolve(lu, piv, b)
	if err != nil {
		t.Fatalf("LUSolve: %v", err)
	}

	// Verify A*x = b.
	for i := 0; i < 3; i++ {
		sum := 0.0
		for j := 0; j < 3; j++ {
			sum += a[i][j] * x[j]
		}
		if !approxEqual(sum, b[i], 1e-10) {
			t.Errorf("LUSolve: A*x[%d] = %v, want %v", i, sum, b[i])
		}
	}
}

func TestLUSingular(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{2, 4},
	}
	_, _, err := LUFactor(a)
	if err == nil {
		t.Error("LUFactor: expected error for singular matrix")
	}
}

// ---------------------------------------------------------------------------
// Cholesky Tests
// ---------------------------------------------------------------------------

func TestChoFactor(t *testing.T) {
	// Symmetric positive-definite matrix.
	a := [][]float64{
		{4, 2},
		{2, 3},
	}
	l, err := ChoFactor(a)
	if err != nil {
		t.Fatalf("ChoFactor: %v", err)
	}

	// Verify L * L^T = A.
	n := 2
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += l[i][k] * l[j][k]
			}
			if !approxEqual(sum, a[i][j], 1e-10) {
				t.Errorf("ChoFactor: L*L^T[%d][%d] = %v, want %v", i, j, sum, a[i][j])
			}
		}
	}
}

func TestChoFactorNotPD(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{2, 1},
	}
	_, err := ChoFactor(a)
	if err == nil {
		t.Error("ChoFactor: expected error for non-positive-definite matrix")
	}
}

func TestChoSolve(t *testing.T) {
	a := [][]float64{
		{4, 2},
		{2, 3},
	}
	b := []float64{1, 2}
	cho, err := ChoFactor(a)
	if err != nil {
		t.Fatalf("ChoFactor: %v", err)
	}
	x, err := ChoSolve(cho, b)
	if err != nil {
		t.Fatalf("ChoSolve: %v", err)
	}

	// Verify A*x = b.
	for i := 0; i < 2; i++ {
		sum := 0.0
		for j := 0; j < 2; j++ {
			sum += a[i][j] * x[j]
		}
		if !approxEqual(sum, b[i], 1e-10) {
			t.Errorf("ChoSolve: A*x[%d] = %v, want %v", i, sum, b[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Schur Tests
// ---------------------------------------------------------------------------

func TestSchur(t *testing.T) {
	// Symmetric matrix.
	a := [][]float64{
		{2, 1},
		{1, 3},
	}
	tt, z, err := Schur(a)
	if err != nil {
		t.Fatalf("Schur: %v", err)
	}

	// Verify A = Z*T*Z^T.
	n := 2
	zt := make([][]float64, n)
	for i := 0; i < n; i++ {
		zt[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			zt[i][j] = z[j][i]
		}
	}
	ztt := matMul(z, tt, n)
	result := matMul(ztt, zt, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(result[i][j], a[i][j], 1e-6) {
				t.Errorf("Schur: Z*T*Z^T[%d][%d] = %v, want %v", i, j, result[i][j], a[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Hessenberg Tests
// ---------------------------------------------------------------------------

func TestHessenberg(t *testing.T) {
	a := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	h, q, err := Hessenberg(a)
	if err != nil {
		t.Fatalf("Hessenberg: %v", err)
	}

	// Verify A = Q*H*Q^T.
	n := 3
	qt := make([][]float64, n)
	for i := 0; i < n; i++ {
		qt[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			qt[i][j] = q[j][i]
		}
	}
	qh := matMul(q, h, n)
	result := matMul(qh, qt, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(result[i][j], a[i][j], 1e-10) {
				t.Errorf("Hessenberg: Q*H*Q^T[%d][%d] = %v, want %v", i, j, result[i][j], a[i][j])
			}
		}
	}

	// Verify H is upper Hessenberg (zeros below first subdiagonal).
	for i := 2; i < n; i++ {
		for j := 0; j < i-1; j++ {
			if !approxEqual(h[i][j], 0, 1e-10) {
				t.Errorf("Hessenberg: H[%d][%d] = %v, want 0", i, j, h[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Special Matrix Tests
// ---------------------------------------------------------------------------

func TestBlockDiag(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5}}
	result := BlockDiag(a, b)
	expected := [][]float64{
		{1, 2, 0},
		{3, 4, 0},
		{0, 0, 5},
	}
	for i := range expected {
		for j := range expected[i] {
			if result[i][j] != expected[i][j] {
				t.Errorf("BlockDiag[%d][%d] = %v, want %v", i, j, result[i][j], expected[i][j])
			}
		}
	}
}

func TestCompanion(t *testing.T) {
	// Polynomial: x^3 + 2x^2 + 3x + 4 => coeffs = [1, 2, 3, 4]
	c := Companion([]float64{1, 2, 3, 4})
	if len(c) != 3 || len(c[0]) != 3 {
		t.Fatalf("Companion: wrong size")
	}
	if c[0][0] != -2 || c[0][1] != -3 || c[0][2] != -4 {
		t.Errorf("Companion: first row = %v, want [-2, -3, -4]", c[0])
	}
	if c[1][0] != 1 || c[2][1] != 1 {
		t.Error("Companion: subdiagonal should be 1")
	}
}

func TestCirculant(t *testing.T) {
	c := Circulant([]float64{1, 2, 3})
	expected := [][]float64{
		{1, 2, 3},
		{3, 1, 2},
		{2, 3, 1},
	}
	for i := range expected {
		for j := range expected[i] {
			if c[i][j] != expected[i][j] {
				t.Errorf("Circulant[%d][%d] = %v, want %v", i, j, c[i][j], expected[i][j])
			}
		}
	}
}

func TestHadamard(t *testing.T) {
	h := Hadamard(4)
	if h == nil {
		t.Fatal("Hadamard(4): nil")
	}
	// Verify H * H^T = 4 * I.
	n := 4
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += h[i][k] * h[j][k]
			}
			expected := 0.0
			if i == j {
				expected = float64(n)
			}
			if !approxEqual(sum, expected, 1e-10) {
				t.Errorf("Hadamard: H*H^T[%d][%d] = %v, want %v", i, j, sum, expected)
			}
		}
	}
}

func TestHadamardInvalidSize(t *testing.T) {
	if Hadamard(3) != nil {
		t.Error("Hadamard(3): should return nil")
	}
	if Hadamard(0) != nil {
		t.Error("Hadamard(0): should return nil")
	}
}

func TestHilbert(t *testing.T) {
	h := Hilbert(3)
	expected := [][]float64{
		{1, 0.5, 1.0 / 3},
		{0.5, 1.0 / 3, 0.25},
		{1.0 / 3, 0.25, 0.2},
	}
	for i := range expected {
		for j := range expected[i] {
			if !approxEqual(h[i][j], expected[i][j], 1e-14) {
				t.Errorf("Hilbert[%d][%d] = %v, want %v", i, j, h[i][j], expected[i][j])
			}
		}
	}
}

func TestInvHilbert(t *testing.T) {
	n := 3
	h := Hilbert(n)
	hinv := InvHilbert(n)

	// Verify H * H^{-1} = I.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += h[i][k] * hinv[k][j]
			}
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if !approxEqual(sum, expected, 1e-6) {
				t.Errorf("InvHilbert: H*Hinv[%d][%d] = %v, want %v", i, j, sum, expected)
			}
		}
	}
}

func TestPascal(t *testing.T) {
	p := Pascal(4)
	expected := [][]float64{
		{1, 1, 1, 1},
		{1, 2, 3, 4},
		{1, 3, 6, 10},
		{1, 4, 10, 20},
	}
	for i := range expected {
		for j := range expected[i] {
			if p[i][j] != expected[i][j] {
				t.Errorf("Pascal[%d][%d] = %v, want %v", i, j, p[i][j], expected[i][j])
			}
		}
	}
}

func TestToeplitz(t *testing.T) {
	c := []float64{1, 2, 3}
	r := []float64{1, 4, 5}
	tp := Toeplitz(c, r)
	expected := [][]float64{
		{1, 4, 5},
		{2, 1, 4},
		{3, 2, 1},
	}
	for i := range expected {
		for j := range expected[i] {
			if tp[i][j] != expected[i][j] {
				t.Errorf("Toeplitz[%d][%d] = %v, want %v", i, j, tp[i][j], expected[i][j])
			}
		}
	}
}

func TestHankel(t *testing.T) {
	c := []float64{1, 2, 3}
	r := []float64{3, 4, 5}
	h := Hankel(c, r)
	expected := [][]float64{
		{1, 2, 3},
		{2, 3, 4},
		{3, 4, 5},
	}
	for i := range expected {
		for j := range expected[i] {
			if h[i][j] != expected[i][j] {
				t.Errorf("Hankel[%d][%d] = %v, want %v", i, j, h[i][j], expected[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Matrix Function Tests
// ---------------------------------------------------------------------------

func TestExpm(t *testing.T) {
	// e^(0) = I.
	zero := [][]float64{{0, 0}, {0, 0}}
	result, err := Expm(zero)
	if err != nil {
		t.Fatalf("Expm: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if !approxEqual(result[i][j], expected, 1e-10) {
				t.Errorf("Expm(0)[%d][%d] = %v, want %v", i, j, result[i][j], expected)
			}
		}
	}

	// e^(I) should have e on diagonal.
	eye := [][]float64{{1, 0}, {0, 1}}
	result, err = Expm(eye)
	if err != nil {
		t.Fatalf("Expm: %v", err)
	}
	if !approxEqual(result[0][0], math.E, 1e-8) {
		t.Errorf("Expm(I)[0][0] = %v, want %v", result[0][0], math.E)
	}
	if !approxEqual(result[1][1], math.E, 1e-8) {
		t.Errorf("Expm(I)[1][1] = %v, want %v", result[1][1], math.E)
	}
	if !approxEqual(result[0][1], 0, 1e-10) {
		t.Errorf("Expm(I)[0][1] = %v, want 0", result[0][1])
	}
}

func TestSqrtm(t *testing.T) {
	// sqrt(I) = I.
	eye := [][]float64{{1, 0}, {0, 1}}
	result, err := Sqrtm(eye)
	if err != nil {
		t.Fatalf("Sqrtm: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if !approxEqual(result[i][j], expected, 1e-10) {
				t.Errorf("Sqrtm(I)[%d][%d] = %v, want %v", i, j, result[i][j], expected)
			}
		}
	}

	// sqrt(4*I) = 2*I.
	a := [][]float64{{4, 0}, {0, 4}}
	result, err = Sqrtm(a)
	if err != nil {
		t.Fatalf("Sqrtm: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = 2.0
			}
			if !approxEqual(result[i][j], expected, 1e-8) {
				t.Errorf("Sqrtm(4I)[%d][%d] = %v, want %v", i, j, result[i][j], expected)
			}
		}
	}
}

func TestLogm(t *testing.T) {
	// log(I) = 0.
	eye := [][]float64{{1, 0}, {0, 1}}
	result, err := Logm(eye)
	if err != nil {
		t.Fatalf("Logm: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if !approxEqual(result[i][j], 0, 1e-8) {
				t.Errorf("Logm(I)[%d][%d] = %v, want 0", i, j, result[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Polar Decomposition Tests
// ---------------------------------------------------------------------------

func TestPolar(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	u, p, err := Polar(a)
	if err != nil {
		t.Fatalf("Polar: %v", err)
	}
	n := 2

	// Verify A = U * P.
	up := matMul(u, p, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(up[i][j], a[i][j], 1e-8) {
				t.Errorf("Polar: U*P[%d][%d] = %v, want %v", i, j, up[i][j], a[i][j])
			}
		}
	}

	// Verify U is orthogonal: U^T * U = I.
	uT := make([][]float64, n)
	for i := 0; i < n; i++ {
		uT[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			uT[i][j] = u[j][i]
		}
	}
	utu := matMul(uT, u, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if !approxEqual(utu[i][j], expected, 1e-8) {
				t.Errorf("Polar: U^T*U[%d][%d] = %v, want %v", i, j, utu[i][j], expected)
			}
		}
	}

	// Verify P is symmetric.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(p[i][j], p[j][i], 1e-10) {
				t.Errorf("Polar: P not symmetric at [%d][%d]", i, j)
			}
		}
	}
}

func TestPolarEmpty(t *testing.T) {
	_, _, err := Polar(nil)
	if err == nil {
		t.Error("Polar: expected error for nil input")
	}
}

// ---------------------------------------------------------------------------
// Fiedler Matrix Tests
// ---------------------------------------------------------------------------

func TestFiedler(t *testing.T) {
	a := []float64{1, 4, 2}
	f := Fiedler(a)
	expected := [][]float64{
		{0, 3, 1},
		{3, 0, 2},
		{1, 2, 0},
	}
	for i := range expected {
		for j := range expected[i] {
			if !approxEqual(f[i][j], expected[i][j], 1e-14) {
				t.Errorf("Fiedler[%d][%d] = %v, want %v", i, j, f[i][j], expected[i][j])
			}
		}
	}
	// Verify symmetry.
	for i := 0; i < len(f); i++ {
		for j := 0; j < len(f); j++ {
			if f[i][j] != f[j][i] {
				t.Errorf("Fiedler not symmetric at [%d][%d]", i, j)
			}
		}
	}
}

func TestFiedlerEmpty(t *testing.T) {
	f := Fiedler(nil)
	if f != nil {
		t.Error("Fiedler(nil) should return nil")
	}
}

// ---------------------------------------------------------------------------
// Leslie Matrix Tests
// ---------------------------------------------------------------------------

func TestLeslie(t *testing.T) {
	f := []float64{0.1, 2.0, 1.0}
	s := []float64{0.5, 0.3}
	l := Leslie(f, s)
	if l == nil {
		t.Fatal("Leslie returned nil")
	}
	expected := [][]float64{
		{0.1, 2.0, 1.0},
		{0.5, 0, 0},
		{0, 0.3, 0},
	}
	for i := range expected {
		for j := range expected[i] {
			if !approxEqual(l[i][j], expected[i][j], 1e-14) {
				t.Errorf("Leslie[%d][%d] = %v, want %v", i, j, l[i][j], expected[i][j])
			}
		}
	}
}

func TestLeslieEmpty(t *testing.T) {
	if Leslie(nil, nil) != nil {
		t.Error("Leslie(nil, nil) should return nil")
	}
}

func TestLeslieMismatch(t *testing.T) {
	// len(s) must be len(f)-1.
	if Leslie([]float64{1, 2}, []float64{0.5, 0.5}) != nil {
		t.Error("Leslie should return nil for mismatched lengths")
	}
}

// ---------------------------------------------------------------------------
// DFT Matrix Tests
// ---------------------------------------------------------------------------

func TestDFT(t *testing.T) {
	w, err := DFT(4)
	if err != nil {
		t.Fatalf("DFT: %v", err)
	}
	n := 4
	// Verify W * W^* = n * I (unitarity up to scale).
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := complex(0, 0)
			for k := 0; k < n; k++ {
				// W^* is conjugate: conj(W[k][j])
				sum += w[i][k] * complex(real(w[j][k]), -imag(w[j][k]))
			}
			expected := complex(0, 0)
			if i == j {
				expected = complex(float64(n), 0)
			}
			if math.Abs(real(sum)-real(expected)) > 1e-10 || math.Abs(imag(sum)-imag(expected)) > 1e-10 {
				t.Errorf("DFT: W*W^*[%d][%d] = %v, want %v", i, j, sum, expected)
			}
		}
	}
}

func TestDFTInvalid(t *testing.T) {
	_, err := DFT(0)
	if err == nil {
		t.Error("DFT(0): expected error")
	}
	_, err = DFT(-1)
	if err == nil {
		t.Error("DFT(-1): expected error")
	}
}

// ---------------------------------------------------------------------------
// LDL Decomposition Tests
// ---------------------------------------------------------------------------

func TestLDL(t *testing.T) {
	// Symmetric positive definite matrix.
	a := [][]float64{
		{4, 2, 1},
		{2, 5, 3},
		{1, 3, 6},
	}
	l, d, err := LDL(a)
	if err != nil {
		t.Fatalf("LDL: %v", err)
	}
	n := 3

	// Verify A = L * D * L^T.
	// First compute D as a diagonal matrix.
	dMat := make([][]float64, n)
	for i := 0; i < n; i++ {
		dMat[i] = make([]float64, n)
		dMat[i][i] = d[i]
	}
	lT := make([][]float64, n)
	for i := 0; i < n; i++ {
		lT[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			lT[i][j] = l[j][i]
		}
	}
	ld := matMul(l, dMat, n)
	ldlt := matMul(ld, lT, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(ldlt[i][j], a[i][j], 1e-10) {
				t.Errorf("LDL: L*D*L^T[%d][%d] = %v, want %v", i, j, ldlt[i][j], a[i][j])
			}
		}
	}

	// Verify L is unit lower triangular.
	for i := 0; i < n; i++ {
		if !approxEqual(l[i][i], 1, 1e-14) {
			t.Errorf("LDL: L[%d][%d] = %v, want 1", i, i, l[i][i])
		}
		for j := i + 1; j < n; j++ {
			if l[i][j] != 0 {
				t.Errorf("LDL: L[%d][%d] = %v, want 0", i, j, l[i][j])
			}
		}
	}
}

func TestLDLEmpty(t *testing.T) {
	_, _, err := LDL(nil)
	if err == nil {
		t.Error("LDL(nil): expected error")
	}
}

// ---------------------------------------------------------------------------
// Interpolative Decomposition Tests
// ---------------------------------------------------------------------------

func TestInterpolative(t *testing.T) {
	// 4x3 matrix with rank 2.
	a := [][]float64{
		{1, 2, 3},
		{4, 5, 9},
		{7, 8, 15},
		{2, 3, 5},
	}
	k := 2
	idx, proj, err := Interpolative(a, k)
	if err != nil {
		t.Fatalf("Interpolative: %v", err)
	}
	if len(idx) != k {
		t.Fatalf("Interpolative: got %d indices, want %d", len(idx), k)
	}

	// Reconstruct: A_approx = A[:, idx] * proj
	m := len(a)
	n := len(a[0])
	approxA := make([][]float64, m)
	for i := 0; i < m; i++ {
		approxA[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			for ki := 0; ki < k; ki++ {
				approxA[i][j] += a[i][idx[ki]] * proj[ki][j]
			}
		}
	}

	// Since the matrix has rank 2, the approximation with k=2 should be exact.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if !approxEqual(approxA[i][j], a[i][j], 1e-8) {
				t.Errorf("Interpolative: approx[%d][%d] = %v, want %v", i, j, approxA[i][j], a[i][j])
			}
		}
	}
}

func TestInterpolativeEmpty(t *testing.T) {
	_, _, err := Interpolative(nil, 1)
	if err == nil {
		t.Error("Interpolative(nil): expected error")
	}
}

func TestInterpolativeInvalidK(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	_, _, err := Interpolative(a, 0)
	if err == nil {
		t.Error("Interpolative k=0: expected error")
	}
	_, _, err = Interpolative(a, 3)
	if err == nil {
		t.Error("Interpolative k>n: expected error")
	}
}
