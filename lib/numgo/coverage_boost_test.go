//go:build unit

package numgo

import (
	"math"
	"testing"
)

// ---------- arithmetic.go: elementWise (0%) ----------

func TestElementWiseDirect(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	r := elementWise(a, b, func(x, y float64) float64 { return x + y })
	if r.data[0] != 5 || r.data[1] != 7 || r.data[2] != 9 {
		t.Fatalf("unexpected result: %v", r.data)
	}
}

func TestElementWisePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	a := FromSlice([]float64{1, 2})
	b := FromSlice([]float64{1, 2, 3})
	elementWise(a, b, func(x, y float64) float64 { return x + y })
}

// ---------- creation.go ----------

func TestArangeStepZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for step=0")
		}
	}()
	Arange(0, 10, 0)
}

func TestArangeNegativeStepCoverage(t *testing.T) {
	r := Arange(5, 0, -1)
	if r.Size() != 5 {
		t.Fatalf("expected 5 elements, got %d", r.Size())
	}
}

func TestArangeEmpty(t *testing.T) {
	r := Arange(5, 0, 1) // step>0 but start>stop => empty
	if r.Size() != 0 {
		t.Fatalf("expected 0 elements, got %d", r.Size())
	}
}

func TestLinspaceZero(t *testing.T) {
	r := Linspace(0, 1, 0)
	if r.Size() != 0 {
		t.Fatal("expected empty array")
	}
}

func TestLinspaceOne(t *testing.T) {
	r := Linspace(5, 10, 1)
	if r.Size() != 1 || r.data[0] != 5 {
		t.Fatal("expected single element equal to start")
	}
}

func TestGeomspacePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-positive start")
		}
	}()
	Geomspace(0, 10, 5)
}

func TestGeomspaceNegativeStop(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-positive stop")
		}
	}()
	Geomspace(1, -1, 5)
}

func TestGeomspaceZeroNum(t *testing.T) {
	r := Geomspace(1, 10, 0)
	if r.Size() != 0 {
		t.Fatal("expected empty array")
	}
}

func TestGeomspaceOneNum(t *testing.T) {
	r := Geomspace(5, 100, 1)
	if r.Size() != 1 || r.data[0] != 5 {
		t.Fatal("expected single element equal to start")
	}
}

func TestMeshgridEmpty(t *testing.T) {
	r := Meshgrid()
	if r != nil {
		t.Fatal("expected nil for empty input")
	}
}

func TestMeshgridPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-1D input")
		}
	}()
	Meshgrid(Zeros(2, 2))
}

func TestDiag1DNeg(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	r := Diag(a, -1)
	if r.shape[0] != 3 || r.shape[1] != 3 {
		t.Fatalf("expected 3x3, got %v", r.shape)
	}
	if r.Get(1, 0) != 1 || r.Get(2, 1) != 2 {
		t.Fatal("wrong diagonal placement")
	}
}

func TestDiag2DNeg(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
	r := Diag(a, -1)
	if r.Size() != 2 || r.data[0] != 4 || r.data[1] != 8 {
		t.Fatalf("wrong subdiagonal: %v", r.data)
	}
}

func TestDiag2DEmptyDiag(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	r := Diag(a, 5) // offset too large
	if r.Size() != 0 {
		t.Fatal("expected empty diagonal")
	}
}

func TestDiagPanic3D(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 3D input")
		}
	}()
	a := Zeros(2, 2, 2)
	Diag(a, 0)
}

func TestAbsPositive(t *testing.T) {
	if abs(5) != 5 {
		t.Fatal("abs(5) should be 5")
	}
	if abs(0) != 0 {
		t.Fatal("abs(0) should be 0")
	}
}

func TestTrilPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-2D")
		}
	}()
	Tril(FromSlice([]float64{1, 2}), 0)
}

func TestTriuPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-2D")
		}
	}()
	Triu(FromSlice([]float64{1, 2}), 0)
}

func TestVanderPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-1D")
		}
	}()
	Vander(Zeros(2, 2), 3)
}

func TestVanderDefaultN(t *testing.T) {
	x := FromSlice([]float64{1, 2, 3})
	r := Vander(x, 0)
	if r.shape[0] != 3 || r.shape[1] != 3 {
		t.Fatalf("expected 3x3, got %v", r.shape)
	}
}

func TestFromIterEarlyClose(t *testing.T) {
	ch := make(chan float64, 3)
	ch <- 1
	ch <- 2
	close(ch)
	r := FromIter(ch, 5) // request 5 but only 2 available
	if r.Size() != 2 {
		t.Fatalf("expected 2 elements, got %d", r.Size())
	}
}

// ---------- linalg.go ----------

func TestDotShapeMismatch1D(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	b := FromSlice([]float64{1, 2, 3})
	_, err := Dot(a, b)
	if err == nil {
		t.Fatal("expected error on 1D shape mismatch")
	}
}

func TestDot2D1DMismatch(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	b := FromSlice([]float64{1, 2, 3})
	_, err := Dot(a, b)
	if err == nil {
		t.Fatal("expected error on 2D-1D mismatch")
	}
}

func TestDotUnsupportedDims(t *testing.T) {
	a := Zeros(2, 2, 2)
	b := FromSlice([]float64{1, 2})
	_, err := Dot(a, b)
	if err == nil {
		t.Fatal("expected error on unsupported dims")
	}
}

func TestInnerNon1D(t *testing.T) {
	a := Zeros(2, 2)
	b := Zeros(2, 2)
	_, err := Inner(a, b)
	if err == nil {
		t.Fatal("expected error for non-1D Inner")
	}
}

func TestTensordotNegativeAxes(t *testing.T) {
	a := Zeros(2, 3)
	b := Zeros(3, 2)
	_, err := Tensordot(a, b, -1)
	if err == nil {
		t.Fatal("expected error for negative axes")
	}
}

func TestTensordotAxesTooLarge(t *testing.T) {
	a := Zeros(2, 3)
	b := Zeros(3, 2)
	_, err := Tensordot(a, b, 5)
	if err == nil {
		t.Fatal("expected error for axes exceeding dims")
	}
}

func TestTensordotDimMismatch(t *testing.T) {
	a := Zeros(2, 3)
	b := Zeros(4, 2) // contracted dim 3 != 4
	_, err := Tensordot(a, b, 1)
	if err == nil {
		t.Fatal("expected error for contracted dim mismatch")
	}
}

func TestTensordotScalarResult(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	r, err := Tensordot(a, b, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.data[0] != 32 {
		t.Fatalf("expected 32, got %f", r.data[0])
	}
}

func TestSolveNonSquare(t *testing.T) {
	a := Zeros(2, 3)
	b := FromSlice([]float64{1, 2})
	_, err := Solve(a, b)
	if err == nil {
		t.Fatal("expected error for non-square a")
	}
}

func TestSolveDimMismatch(t *testing.T) {
	a := Eye(2)
	b := FromSlice([]float64{1, 2, 3})
	_, err := Solve(a, b)
	if err == nil {
		t.Fatal("expected error for b dimension mismatch")
	}
}

func TestSolveSingular(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 0}, {0, 0}})
	b := FromSlice([]float64{1, 2})
	_, err := Solve(a, b)
	if err == nil {
		t.Fatal("expected error for singular matrix")
	}
}

func TestSolve2D(t *testing.T) {
	a := Eye(2)
	b := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	r, err := Solve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if r.shape[0] != 2 || r.shape[1] != 2 {
		t.Fatalf("expected 2x2 result, got %v", r.shape)
	}
}

func TestSolveBadDim(t *testing.T) {
	a := Eye(2)
	b := Zeros(2, 2, 2)
	_, err := Solve(a, b)
	if err == nil {
		t.Fatal("expected error for 3D b")
	}
}

func TestSolve2DDimMismatch(t *testing.T) {
	a := Eye(2)
	b := Zeros(3, 2)
	_, err := Solve(a, b)
	if err == nil {
		t.Fatal("expected error for dimension mismatch in 2D solve")
	}
}

func TestEighBasic(t *testing.T) {
	a := FromSlice2D([][]float64{{2, 1}, {1, 2}})
	vals, vecs, err := Eigh(a)
	if err != nil {
		t.Fatal(err)
	}
	if vals.Size() != 2 || vecs.Size() != 4 {
		t.Fatal("unexpected output sizes")
	}
}

func TestEighNonSquare(t *testing.T) {
	a := Zeros(2, 3)
	_, _, err := Eigh(a)
	if err == nil {
		t.Fatal("expected error for non-square")
	}
}

func TestSVDNon2D(t *testing.T) {
	a := Zeros(2, 2, 2)
	_, _, _, err := SVD(a)
	if err == nil {
		t.Fatal("expected error for non-2D")
	}
}

func TestSVDTallMatrix(t *testing.T) {
	// m > n case
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}, {5, 6}})
	u, s, vt, err := SVD(a)
	if err != nil {
		t.Fatal(err)
	}
	if u.shape[0] != 3 || u.shape[1] != 3 {
		t.Fatalf("u shape: %v", u.shape)
	}
	if s.Size() != 2 {
		t.Fatalf("s size: %d", s.Size())
	}
	if vt.shape[0] != 2 || vt.shape[1] != 2 {
		t.Fatalf("vt shape: %v", vt.shape)
	}
}

func TestCholeskyNotPositiveDefinite(t *testing.T) {
	a := FromSlice2D([][]float64{{-1, 0}, {0, 1}})
	_, err := Cholesky(a)
	if err == nil {
		t.Fatal("expected error for non-positive-definite matrix")
	}
}

func TestCholeskyNonSquare(t *testing.T) {
	_, err := Cholesky(Zeros(2, 3))
	if err == nil {
		t.Fatal("expected error for non-square")
	}
}

func TestCholeskyZeroDiag(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 0}, {0, 1}})
	_, err := Cholesky(a)
	if err == nil {
		t.Fatal("expected error for zero diagonal")
	}
}

func TestQRNon2D(t *testing.T) {
	_, _, err := QR(Zeros(2, 2, 2))
	if err == nil {
		t.Fatal("expected error for non-2D")
	}
}

func TestLstsqNon2D(t *testing.T) {
	_, err := Lstsq(FromSlice([]float64{1, 2}), FromSlice([]float64{1, 2}))
	if err == nil {
		t.Fatal("expected error for non-2D a")
	}
}

func TestNormOrd1(t *testing.T) {
	a := FromSlice([]float64{-1, 2, -3})
	r, err := Norm(a, 1, -1)
	if err != nil || r.data[0] != 6 {
		t.Fatalf("expected 6, got %v (err=%v)", r.data[0], err)
	}
}

func TestNormOrd0(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 0, 3})
	r, err := Norm(a, 0, -1)
	if err != nil || r.data[0] != 3 {
		t.Fatalf("expected 3, got %v", r.data[0])
	}
}

func TestNormOrdGeneral(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	r, err := Norm(a, 3, -1)
	if err != nil {
		t.Fatal(err)
	}
	expected := math.Pow(1+8+27, 1.0/3.0)
	if math.Abs(r.data[0]-expected) > 1e-10 {
		t.Fatalf("expected %f, got %f", expected, r.data[0])
	}
}

func TestNormAxisSpecific(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	_, err := Norm(a, 2, 0)
	if err == nil {
		t.Fatal("expected error for axis-specific norm")
	}
}

func TestCondBasic(t *testing.T) {
	a := Eye(2)
	c, err := Cond(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(c-1) > 0.5 {
		t.Fatalf("expected cond ~1, got %f", c)
	}
}

func TestCondSingular(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 0}, {0, 0}})
	c, err := Cond(a)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(c, 1) {
		t.Fatalf("expected Inf for singular matrix, got %f", c)
	}
}

func TestCondNon2D(t *testing.T) {
	_, err := Cond(FromSlice([]float64{1, 2}))
	if err == nil {
		t.Fatal("expected error for non-2D")
	}
}

func TestMatrixRankFull(t *testing.T) {
	a := Eye(3)
	r, err := MatrixRank(a)
	if err != nil || r != 3 {
		t.Fatalf("expected 3, got %d", r)
	}
}

func TestMatrixPowerZeroCoverage(t *testing.T) {
	a := FromSlice2D([][]float64{{2, 0}, {0, 3}})
	r, err := MatrixPower(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.Get(0, 0) != 1 || r.Get(1, 1) != 1 {
		t.Fatal("expected identity")
	}
}

func TestMatrixPowerNegativeCoverage(t *testing.T) {
	a := FromSlice2D([][]float64{{2, 0}, {0, 4}})
	r, err := MatrixPower(a, -1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.Get(0, 0)-0.5) > 1e-10 || math.Abs(r.Get(1, 1)-0.25) > 1e-10 {
		t.Fatal("expected inverse")
	}
}

func TestMatrixPowerNonSquare(t *testing.T) {
	_, err := MatrixPower(Zeros(2, 3), 2)
	if err == nil {
		t.Fatal("expected error for non-square")
	}
}

func TestSlogdetPositive(t *testing.T) {
	a := FromSlice2D([][]float64{{2, 0}, {0, 3}})
	sign, logdet, err := Slogdet(a)
	if err != nil {
		t.Fatal(err)
	}
	if sign != 1 || math.Abs(logdet-math.Log(6)) > 1e-10 {
		t.Fatalf("expected sign=1, logdet=%f, got sign=%f, logdet=%f", math.Log(6), sign, logdet)
	}
}

func TestSlogdetNegative(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 1}, {1, 0}})
	sign, logdet, err := Slogdet(a)
	if err != nil {
		t.Fatal(err)
	}
	if sign != -1 || math.Abs(logdet-0) > 1e-10 {
		t.Fatalf("expected sign=-1, logdet=0, got sign=%f, logdet=%f", sign, logdet)
	}
}

func TestSlogdetZero(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 0}, {0, 0}})
	sign, logdet, err := Slogdet(a)
	if err != nil {
		t.Fatal(err)
	}
	if sign != 0 || !math.IsInf(logdet, -1) {
		t.Fatalf("expected sign=0, logdet=-Inf, got sign=%f, logdet=%f", sign, logdet)
	}
}

func TestSlogdetNonSquare(t *testing.T) {
	_, _, err := Slogdet(Zeros(2, 3))
	if err == nil {
		t.Fatal("expected error for non-square")
	}
}

func TestTraceNon2D(t *testing.T) {
	_, err := Trace(FromSlice([]float64{1, 2}))
	if err == nil {
		t.Fatal("expected error for non-2D")
	}
}

func TestTraceRectangular(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2, 3}, {4, 5, 6}})
	tr, err := Trace(a)
	if err != nil {
		t.Fatal(err)
	}
	if tr != 6 {
		t.Fatalf("expected 6, got %f", tr)
	}
}

func TestCrossWrongSize(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	b := FromSlice([]float64{3, 4, 5})
	_, err := Cross(a, b)
	if err == nil {
		t.Fatal("expected error for non-3-element vectors")
	}
}

func TestDetPivotSwap(t *testing.T) {
	// Force a pivot swap
	a := FromSlice2D([][]float64{{0, 1}, {1, 0}})
	d, err := Det(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(d-(-1)) > 1e-10 {
		t.Fatalf("expected -1, got %f", d)
	}
}

func TestDetSingularCoverage(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 0}, {0, 0}})
	d, err := Det(a)
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Fatalf("expected 0, got %f", d)
	}
}

func TestInvSingularCoverage(t *testing.T) {
	a := FromSlice2D([][]float64{{0, 0}, {0, 0}})
	_, err := Inv(a)
	if err == nil {
		t.Fatal("expected error for singular matrix")
	}
}

func TestMatmulDimMismatch(t *testing.T) {
	a := Zeros(2, 3)
	b := Zeros(4, 2)
	_, err := Matmul(a, b)
	if err == nil {
		t.Fatal("expected error for inner dimension mismatch")
	}
}

func TestMatmulNon2D(t *testing.T) {
	_, err := Matmul(Zeros(2, 2, 2), Zeros(2, 2))
	if err == nil {
		t.Fatal("expected error for non-2D")
	}
}

func TestEigNonSquare(t *testing.T) {
	_, _, err := Eig(Zeros(2, 3))
	if err == nil {
		t.Fatal("expected error for non-square")
	}
}

func TestPinvBasic(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 0}, {0, 2}})
	p, err := Pinv(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p.Get(0, 0)-1) > 0.1 || math.Abs(p.Get(1, 1)-0.5) > 0.1 {
		t.Fatalf("unexpected pinv: %v", p.data)
	}
}

// ---------- manipulation.go ----------

func TestSwapaxesPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Swapaxes(Zeros(2, 3), 0, 5)
}

func TestSwapaxesPanicAxis1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Swapaxes(Zeros(2, 3), 5, 0)
}

func TestMoveaxisPanicSrc(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Moveaxis(Zeros(2, 3), 5, 0)
}

func TestMoveaxisPanicDst(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Moveaxis(Zeros(2, 3), 0, 5)
}

func TestRollaxisPanicAxis(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Rollaxis(Zeros(2, 3), 5, 0)
}

func TestRollaxisPanicStart(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Rollaxis(Zeros(2, 3), 0, -1)
}

func TestRollaxisSameAxisStart(t *testing.T) {
	a := Zeros(2, 3)
	r := Rollaxis(a, 0, 0) // axis==start => copy
	if !shapeEqual(r.shape, a.shape) {
		t.Fatal("expected same shape")
	}
}

func TestRollaxisStartGreaterThanAxis(t *testing.T) {
	a := Zeros(2, 3, 4)
	r := Rollaxis(a, 0, 2) // start > axis case
	if r.shape[0] != 3 || r.shape[1] != 2 || r.shape[2] != 4 {
		t.Fatalf("unexpected shape: %v", r.shape)
	}
}

func TestExpandDimsPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	ExpandDims(Zeros(2, 3), 5)
}

func TestConcatenateEmpty(t *testing.T) {
	_, err := Concatenate(nil, 0)
	if err == nil {
		t.Fatal("expected error for empty arrays")
	}
}

func TestConcatenateBadAxis(t *testing.T) {
	_, err := Concatenate([]*NDArray{Zeros(2, 3)}, 5)
	if err == nil {
		t.Fatal("expected error for bad axis")
	}
}

func TestConcatenateDimMismatch(t *testing.T) {
	_, err := Concatenate([]*NDArray{Zeros(2, 3), Zeros(2, 3, 4)}, 0)
	if err == nil {
		t.Fatal("expected error for dim mismatch")
	}
}

func TestConcatenateShapeMismatch(t *testing.T) {
	_, err := Concatenate([]*NDArray{Zeros(2, 3), Zeros(2, 4)}, 0)
	if err == nil {
		t.Fatal("expected error for shape mismatch on non-concat axis")
	}
}

func TestStackEmpty(t *testing.T) {
	_, err := Stack(nil, 0)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStackBadAxis(t *testing.T) {
	_, err := Stack([]*NDArray{FromSlice([]float64{1, 2})}, 5)
	if err == nil {
		t.Fatal("expected error for bad axis")
	}
}

func TestStackDimMismatch(t *testing.T) {
	_, err := Stack([]*NDArray{Zeros(2), Zeros(2, 3)}, 0)
	if err == nil {
		t.Fatal("expected error for dim mismatch")
	}
}

func TestStackShapeMismatch(t *testing.T) {
	_, err := Stack([]*NDArray{Zeros(2), Zeros(3)}, 0)
	if err == nil {
		t.Fatal("expected error for shape mismatch")
	}
}

func TestVstackEmpty(t *testing.T) {
	_, err := Vstack(nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestVstack2D(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}})
	b := FromSlice2D([][]float64{{3, 4}})
	r, err := Vstack([]*NDArray{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if r.shape[0] != 2 || r.shape[1] != 2 {
		t.Fatalf("expected 2x2, got %v", r.shape)
	}
}

func TestHstackEmpty(t *testing.T) {
	_, err := Hstack(nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestHstack2DCoverage(t *testing.T) {
	a := FromSlice2D([][]float64{{1}, {2}})
	b := FromSlice2D([][]float64{{3}, {4}})
	r, err := Hstack([]*NDArray{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if r.shape[0] != 2 || r.shape[1] != 2 {
		t.Fatalf("expected 2x2, got %v", r.shape)
	}
}

func TestDstackEmpty(t *testing.T) {
	_, err := Dstack(nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDstack3D(t *testing.T) {
	a := Zeros(2, 3, 1)
	b := Zeros(2, 3, 1)
	r, err := Dstack([]*NDArray{a, b})
	if err != nil {
		t.Fatal(err)
	}
	if r.shape[2] != 2 {
		t.Fatalf("expected depth 2, got %v", r.shape)
	}
}

func TestSplitBadAxis(t *testing.T) {
	_, err := Split(Zeros(4), 2, 5)
	if err == nil {
		t.Fatal("expected error for bad axis")
	}
}

func TestSplitBadSections(t *testing.T) {
	_, err := Split(Zeros(4), 0, 0)
	if err == nil {
		t.Fatal("expected error for zero sections")
	}
}

func TestSplitUnevenSections(t *testing.T) {
	_, err := Split(Zeros(5), 2, 0)
	if err == nil {
		t.Fatal("expected error for uneven split")
	}
}

func TestHsplitMultiDim(t *testing.T) {
	a := Zeros(2, 4)
	r, err := Hsplit(a, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(r))
	}
}

func TestVsplitNon2D(t *testing.T) {
	_, err := Vsplit(FromSlice([]float64{1, 2}), 1)
	if err == nil {
		t.Fatal("expected error for < 2D")
	}
}

func TestDsplitNon3D(t *testing.T) {
	_, err := Dsplit(Zeros(2, 2), 1)
	if err == nil {
		t.Fatal("expected error for < 3D")
	}
}

func TestFlipPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Flip(Zeros(2, 3), 5)
}

func TestFliplrPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Fliplr(FromSlice([]float64{1, 2}))
}

func TestFliplr2D(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	r := Fliplr(a)
	if r.Get(0, 0) != 2 || r.Get(0, 1) != 1 {
		t.Fatal("unexpected result")
	}
}

func TestRot90Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Rot90(FromSlice([]float64{1}), 1)
}

func TestRot90Zero(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	r := Rot90(a, 0)
	if !ArrayEqual(a, r) {
		t.Fatal("expected copy for k=0")
	}
}

func TestRollAxisNeg1(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	r := Roll(a, 1, -1)
	// Should flatten, roll, reshape
	if !shapeEqual(r.shape, a.shape) {
		t.Fatalf("expected same shape, got %v", r.shape)
	}
}

func TestRollZeroShift(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	r := Roll(a, 0, 0)
	if !ArrayEqual(a, r) {
		t.Fatal("expected same array for shift=0")
	}
}

func TestRollPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Roll(Zeros(2, 3), 1, 5)
}

func TestRepeatPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Repeat(Zeros(2, 3), 2, 5)
}

func TestDeleteBadAxis(t *testing.T) {
	_, err := Delete(Zeros(3), []int{0}, 5)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDeleteBadIndex(t *testing.T) {
	_, err := Delete(Zeros(3), []int{5}, 0)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestInsertBadAxis(t *testing.T) {
	_, err := Insert(Zeros(3), 0, Zeros(1), 5)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestInsertBadIndex(t *testing.T) {
	_, err := Insert(Zeros(3), 10, Zeros(1), 0)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestInsertDimMismatch(t *testing.T) {
	_, err := Insert(Zeros(3), 0, Zeros(2, 2), 0)
	if err == nil {
		t.Fatal("expected error for ndim mismatch")
	}
}

func TestInsertShapeMismatch(t *testing.T) {
	a := Zeros(2, 3)
	v := Zeros(2, 4) // wrong shape on non-axis dim
	_, err := Insert(a, 0, v, 0)
	if err == nil {
		t.Fatal("expected error for shape mismatch on non-axis dim")
	}
}

func TestTileHigherReps(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	r := Tile(a, []int{2, 3})
	// reps has more dims than a => result is 2D
	if r.shape[0] != 2 || r.shape[1] != 6 {
		t.Fatalf("expected (2,6), got %v", r.shape)
	}
}

// ---------- stats.go ----------

func TestPercentilePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for q out of range")
		}
	}()
	Percentile(FromSlice([]float64{1, 2, 3}), -1)
}

func TestPercentileHigh(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for q > 100")
		}
	}()
	Percentile(FromSlice([]float64{1, 2, 3}), 101)
}

func TestQuantilePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for q out of range")
		}
	}()
	Quantile(FromSlice([]float64{1, 2, 3}), -0.1)
}

func TestQuantileHigh(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for q > 1")
		}
	}()
	Quantile(FromSlice([]float64{1, 2, 3}), 1.1)
}

func TestQuantileWithAxis(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2, 3}, {4, 5, 6}})
	r := Quantile(a, 0.5, 1)
	if r.Size() != 2 {
		t.Fatalf("expected 2 elements, got %d", r.Size())
	}
}

func TestPercentileWithAxis(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2, 3}, {4, 5, 6}})
	r := Percentile(a, 50, 1)
	if r.Size() != 2 {
		t.Fatalf("expected 2 elements, got %d", r.Size())
	}
}

func TestInterpolatedQuantileEmpty(t *testing.T) {
	r := interpolatedQuantile([]float64{}, 0.5)
	if !math.IsNaN(r) {
		t.Fatal("expected NaN for empty slice")
	}
}

func TestInterpolatedQuantileSingle(t *testing.T) {
	r := interpolatedQuantile([]float64{42}, 0.5)
	if r != 42 {
		t.Fatalf("expected 42, got %f", r)
	}
}

func TestInterpolatedQuantileEdge(t *testing.T) {
	r := interpolatedQuantile([]float64{1, 2, 3}, 1.0)
	if r != 3 {
		t.Fatalf("expected 3, got %f", r)
	}
}

func TestAverageWeightedSizeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Average(FromSlice([]float64{1, 2}), FromSlice([]float64{1}))
}

func TestAverageMultiAxisPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for multi-axis weighted avg")
		}
	}()
	a := Zeros(2, 3)
	w := FromSlice([]float64{1, 2, 3})
	Average(a, w, 0, 1)
}

func TestAverageAxisBadRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-range axis")
		}
	}()
	Average(Zeros(2, 3), FromSlice([]float64{1, 2, 3}), 5)
}

func TestAverageWeightAxisMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for weight/axis size mismatch")
		}
	}()
	Average(Zeros(2, 3), FromSlice([]float64{1, 2}), 1) // axis 1 has size 3 but weights has 2
}

func TestNanmeanAllNaN(t *testing.T) {
	a := FromSlice([]float64{math.NaN(), math.NaN()})
	r := Nanmean(a)
	if !math.IsNaN(r.data[0]) {
		t.Fatal("expected NaN for all-NaN input")
	}
}

func TestNanvarAllNaN(t *testing.T) {
	a := FromSlice([]float64{math.NaN(), math.NaN()})
	r := Nanvar(a)
	if !math.IsNaN(r.data[0]) {
		t.Fatal("expected NaN for all-NaN input")
	}
}

func TestNanmaxAllNaN(t *testing.T) {
	a := FromSlice([]float64{math.NaN(), math.NaN()})
	r := Nanmax(a)
	if !math.IsNaN(r.data[0]) {
		t.Fatal("expected NaN for all-NaN input")
	}
}

func TestNanReduceWithAxis(t *testing.T) {
	a := FromSlice2D([][]float64{{1, math.NaN()}, {3, 4}})
	r := Nanmean(a, 1)
	if r.Size() != 2 {
		t.Fatalf("expected 2 elements, got %d", r.Size())
	}
}

func TestHistogramAllSame(t *testing.T) {
	a := FromSlice([]float64{5, 5, 5})
	counts, edges := Histogram(a, 3)
	if counts.Size() != 3 {
		t.Fatalf("expected 3 bins, got %d", counts.Size())
	}
	if edges.Size() != 4 {
		t.Fatalf("expected 4 edges, got %d", edges.Size())
	}
}

func TestHistogramPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for bins <= 0")
		}
	}()
	Histogram(FromSlice([]float64{1}), 0)
}

func TestBincountPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative value")
		}
	}()
	Bincount(FromSlice([]float64{-1}))
}

func TestCorrcoefNon1D(t *testing.T) {
	_, err := Corrcoef(Zeros(2, 2), Zeros(2, 2))
	if err == nil {
		t.Fatal("expected error for non-1D")
	}
}

func TestCorrcoefSizeMismatch(t *testing.T) {
	_, err := Corrcoef(FromSlice([]float64{1, 2}), FromSlice([]float64{1, 2, 3}))
	if err == nil {
		t.Fatal("expected error for size mismatch")
	}
}

func TestCovNon2D(t *testing.T) {
	_, err := Cov(Zeros(2, 2, 2))
	if err == nil {
		t.Fatal("expected error for 3D input")
	}
}

func TestCovSingleObs(t *testing.T) {
	a := FromSlice([]float64{5})
	r, err := Cov(a)
	if err != nil {
		t.Fatal(err)
	}
	if r.data[0] != 0 {
		t.Fatalf("expected 0 variance for single obs, got %f", r.data[0])
	}
}

func TestCov2DSingleObs(t *testing.T) {
	a := FromSlice2D([][]float64{{1}, {2}})
	r, err := Cov(a)
	if err != nil {
		t.Fatal(err)
	}
	if r.shape[0] != 2 || r.shape[1] != 2 {
		t.Fatal("expected 2x2 result")
	}
}

func TestCorrelateNon1D(t *testing.T) {
	_, err := Correlate(Zeros(2, 2), Zeros(2))
	if err == nil {
		t.Fatal("expected error for non-1D")
	}
}

func TestCumsumPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Cumsum(Zeros(3), 5)
}

func TestCumprodPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Cumprod(Zeros(3), 5)
}

// ---------- indexing.go ----------

func TestTakeFlatOOB(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	_, err := Take(a, []int{5}, -1)
	if err == nil {
		t.Fatal("expected error for OOB index")
	}
}

func TestTakeAxisOOB(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	_, err := Take(a, []int{0}, 5)
	if err == nil {
		t.Fatal("expected error for OOB axis")
	}
}

func TestTakeIndexOOBOnAxis(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2}, {3, 4}})
	_, err := Take(a, []int{5}, 0)
	if err == nil {
		t.Fatal("expected error for OOB index on axis")
	}
}

func TestTakeAlongAxisDimMismatch(t *testing.T) {
	a := Zeros(2, 3)
	idx := FromSlice([]float64{0, 1})
	_, err := TakeAlongAxis(a, idx, 0)
	if err == nil {
		t.Fatal("expected error for ndim mismatch")
	}
}

func TestTakeAlongAxisBadAxis(t *testing.T) {
	a := Zeros(2, 3)
	idx := Zeros(2, 3)
	_, err := TakeAlongAxis(a, idx, 5)
	if err == nil {
		t.Fatal("expected error for bad axis")
	}
}

func TestChooseNon1D(t *testing.T) {
	_, err := Choose(Zeros(2, 2), []*NDArray{Zeros(4)})
	if err == nil {
		t.Fatal("expected error for non-1D indices")
	}
}

func TestChooseOOBIndex(t *testing.T) {
	idx := FromSlice([]float64{5})
	_, err := Choose(idx, []*NDArray{FromSlice([]float64{1})})
	if err == nil {
		t.Fatal("expected error for OOB choice index")
	}
}

func TestChooseSizeMismatch(t *testing.T) {
	idx := FromSlice([]float64{0})
	choices := []*NDArray{FromSlice([]float64{1, 2, 3})}
	_, err := Choose(idx, choices)
	if err == nil {
		t.Fatal("expected error for choice size mismatch")
	}
}

func TestCompressFlatOOB(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	_, err := Compress([]bool{true, true, true}, a, -1) // condition longer than array
	if err == nil {
		t.Fatal("expected error for condition too long")
	}
}

func TestCompressFlatEmpty(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	r, err := Compress([]bool{false, false, false}, a, -1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Size() != 0 {
		t.Fatal("expected empty result")
	}
}

func TestCompressAxisOOB(t *testing.T) {
	a := Zeros(2, 3)
	_, err := Compress([]bool{true}, a, 5)
	if err == nil {
		t.Fatal("expected error for OOB axis")
	}
}

func TestCompressAxisCondTooLong(t *testing.T) {
	a := Zeros(2, 3)
	_, err := Compress([]bool{true, true, true}, a, 0) // axis 0 has size 2
	if err == nil {
		t.Fatal("expected error for condition too long")
	}
}

func TestDiagonalNon2D(t *testing.T) {
	_, err := Diagonal(FromSlice([]float64{1}), 0, 0, 1)
	if err == nil {
		t.Fatal("expected error for < 2D")
	}
}

func TestDiagonalBadAxes(t *testing.T) {
	_, err := Diagonal(Zeros(2, 3), 0, 0, 0) // same axes
	if err == nil {
		t.Fatal("expected error for same axes")
	}
}

func TestDiagonalOOBAxis(t *testing.T) {
	_, err := Diagonal(Zeros(2, 3), 0, 0, 5)
	if err == nil {
		t.Fatal("expected error for OOB axis")
	}
}

func TestDiagonalEmptyResult(t *testing.T) {
	a := Zeros(2, 3)
	r, err := Diagonal(a, 10, 0, 1) // offset too large
	if err != nil {
		t.Fatal(err)
	}
	if r.Size() != 0 {
		t.Fatal("expected empty diagonal")
	}
}

func TestDiagonalNegativeOffset(t *testing.T) {
	a := FromSlice2D([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
	r, err := Diagonal(a, -1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Size() != 2 || r.data[0] != 4 || r.data[1] != 8 {
		t.Fatalf("wrong subdiagonal: %v", r.data)
	}
}

func TestDiagonal3D(t *testing.T) {
	a := Zeros(2, 3, 4)
	_, err := Diagonal(a, 0, 0, 1)
	if err == nil {
		t.Fatal("expected error for 3D (not yet supported)")
	}
}

func TestSelectMismatchLen(t *testing.T) {
	_, err := Select([]*NDArray{Zeros(3)}, []*NDArray{Zeros(3), Zeros(3)}, 0)
	if err == nil {
		t.Fatal("expected error for length mismatch")
	}
}

func TestSelectEmpty(t *testing.T) {
	_, err := Select(nil, nil, 0)
	if err == nil {
		t.Fatal("expected error for empty conditions")
	}
}

func TestSelectCondSizeMismatch(t *testing.T) {
	_, err := Select([]*NDArray{Zeros(3), Zeros(2)}, []*NDArray{Zeros(3), Zeros(3)}, 0)
	if err == nil {
		t.Fatal("expected error for condition size mismatch")
	}
}

func TestSelectChoiceSizeMismatch(t *testing.T) {
	_, err := Select([]*NDArray{Zeros(3), Zeros(3)}, []*NDArray{Zeros(3), Zeros(2)}, 0)
	if err == nil {
		t.Fatal("expected error for choice size mismatch")
	}
}

// ---------- sorting.go ----------

func TestWhereWithBroadcast(t *testing.T) {
	cond := FromSlice([]float64{1, 0, 1})
	x := FromSlice([]float64{10, 20, 30})
	y := FromSlice([]float64{-1, -2, -3})
	r := Where(cond, x, y)
	if r.data[0] != 10 || r.data[1] != -2 || r.data[2] != 30 {
		t.Fatalf("unexpected result: %v", r.data)
	}
}

func TestLexsortEmpty(t *testing.T) {
	r := Lexsort(nil)
	if r.Size() != 0 {
		t.Fatal("expected empty result")
	}
}

func TestLexsortPanicNon1D(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Lexsort([]*NDArray{Zeros(2, 2)})
}

func TestLexsortPanicLenMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Lexsort([]*NDArray{Zeros(2), Zeros(3)})
}

func TestArgpartitionPanicAxis(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Argpartition(Zeros(3), 0, 5)
}

func TestArgpartitionPanicKth(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	Argpartition(Zeros(3), 5, 0)
}

func TestExtractAllFalse(t *testing.T) {
	cond := FromSlice([]float64{0, 0, 0})
	a := FromSlice([]float64{1, 2, 3})
	r := Extract(cond, a)
	if r.Size() != 0 {
		t.Fatal("expected empty result")
	}
}

func TestUniqueEmpty(t *testing.T) {
	a := NewNDArray([]int{0}, []float64{})
	r := Unique(a)
	if r.Size() != 0 {
		t.Fatal("expected empty result")
	}
}

// ---------- logic.go ----------

func TestArrayEquivDifferentShapes(t *testing.T) {
	a := FromSlice([]float64{1})
	b := FromSlice([]float64{1, 1, 1})
	// (1,) broadcasts to (3,)
	if !ArrayEquiv(a, b) {
		t.Fatal("expected equiv via broadcast")
	}
}

func TestArrayEquivIncompatible(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	b := FromSlice([]float64{1, 2, 3})
	if ArrayEquiv(a, b) {
		t.Fatal("expected not equiv for incompatible shapes")
	}
}

func TestArrayEquivNotEqual(t *testing.T) {
	a := FromSlice([]float64{1})
	b := FromSlice([]float64{2, 2, 2})
	if ArrayEquiv(a, b) {
		t.Fatal("expected not equiv when values differ")
	}
}

// ---------- ndarray.go ----------

func TestStringLargeArray(t *testing.T) {
	a := Zeros(200)
	s := a.String()
	if s == "" {
		t.Fatal("expected non-empty string")
	}
}

func TestString3D(t *testing.T) {
	a := Zeros(2, 3, 4)
	s := a.String()
	if s == "" {
		t.Fatal("expected non-empty string")
	}
}

// ---------- random.go ----------

func TestRandnOddSizeCoverage(t *testing.T) {
	rng := NewRNG(42)
	r := rng.Randn(5) // odd size triggers i+1 < size branch
	if r.Size() != 5 {
		t.Fatal("unexpected size")
	}
}

func TestBoxMullerSingleBasic(t *testing.T) {
	rng := NewRNG(42)
	v := rng.boxMullerSingle()
	if math.IsNaN(v) || math.IsInf(v, 0) {
		t.Fatal("unexpected value from boxMullerSingle")
	}
}

func TestExponentialPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for scale <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Exponential(0, 3)
}

func TestPoissonZero(t *testing.T) {
	rng := NewRNG(42)
	r := rng.Poisson(0, 5)
	for _, v := range r.Data() {
		if v != 0 {
			t.Fatal("expected all zeros for lam=0")
		}
	}
}

func TestPoissonLarge(t *testing.T) {
	rng := NewRNG(42)
	r := rng.Poisson(50, 10) // lam >= 30 triggers rejection method
	if r.Size() != 10 {
		t.Fatal("unexpected size")
	}
}

func TestPoissonPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative lambda")
		}
	}()
	rng := NewRNG(42)
	rng.Poisson(-1, 3)
}

func TestBinomialSamplePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative n")
		}
	}()
	rng := NewRNG(42)
	rng.BinomialSample(-1, 0.5, 3)
}

func TestBinomialSampleBadP(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for p > 1")
		}
	}()
	rng := NewRNG(42)
	rng.BinomialSample(5, 1.5, 3)
}

func TestBetaPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for a <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Beta(0, 1, 3)
}

func TestBetaPanicB(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for b <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Beta(1, 0, 3)
}

func TestGammaPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for shape <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Gamma(0, 1, 3)
}

func TestGammaPanicScale(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for scale <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Gamma(1, 0, 3)
}

func TestChisquarePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for df <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.Chisquare(0, 3)
}

func TestStandardTPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for df <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.StandardT(0, 3)
}

func TestMultinomialPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for n < 0")
		}
	}()
	rng := NewRNG(42)
	rng.Multinomial(-1, []float64{0.5, 0.5})
}

func TestMultinomialEmptyPvals(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for empty pvals")
		}
	}()
	rng := NewRNG(42)
	rng.Multinomial(5, nil)
}

func TestGammaVariateAlphaLessThan1(t *testing.T) {
	rng := NewRNG(42)
	v := rng.gammaVariate(0.5)
	if v <= 0 || math.IsNaN(v) {
		t.Fatalf("unexpected value: %f", v)
	}
}

func TestGammaVariatePanicZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for alpha <= 0")
		}
	}()
	rng := NewRNG(42)
	rng.gammaVariate(0)
}

func TestChoicePanics(t *testing.T) {
	rng := NewRNG(42)

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for n <= 0")
			}
		}()
		rng.Choice(0, 1, true)
	}()

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for size < 0")
			}
		}()
		rng.Choice(5, -1, true)
	}()

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for size > n without replacement")
			}
		}()
		rng.Choice(3, 5, false)
	}()
}

// ---------- einsum.go ----------

func TestEinsumEmptyNotation(t *testing.T) {
	_, err := Einsum("")
	if err == nil {
		t.Fatal("expected error for empty notation")
	}
}

func TestEinsumBadChar(t *testing.T) {
	_, err := Einsum("IJ,JK->IK", Zeros(2, 3), Zeros(3, 2))
	if err == nil {
		t.Fatal("expected error for uppercase chars")
	}
}

func TestEinsumBadOutputChar(t *testing.T) {
	_, err := Einsum("ij->I", Zeros(2, 2))
	if err == nil {
		t.Fatal("expected error for uppercase output char")
	}
}

func TestEinsumOutputLabelNotInInput(t *testing.T) {
	_, err := Einsum("ij->k", Zeros(2, 2))
	if err == nil {
		t.Fatal("expected error for output label not in input")
	}
}

func TestEinsumOperandCountMismatch(t *testing.T) {
	_, err := Einsum("ij,jk->ik", Zeros(2, 3))
	if err == nil {
		t.Fatal("expected error for operand count mismatch")
	}
}

func TestEinsumDimMismatch(t *testing.T) {
	_, err := Einsum("ij->i", Zeros(2, 3, 4)) // 3D operand but 2 subscripts
	if err == nil {
		t.Fatal("expected error for dim count mismatch")
	}
}

func TestEinsumLabelSizeMismatch(t *testing.T) {
	_, err := Einsum("ij,ji->", Zeros(2, 3), Zeros(2, 3)) // j=3 from first, j=2 from second
	if err == nil {
		t.Fatal("expected error for label size inconsistency")
	}
}

func TestMaxOpDimsEmpty(t *testing.T) {
	r := maxOpDims(nil)
	if r != 1 {
		t.Fatalf("expected 1, got %d", r)
	}
}

// ---------- setops.go ----------

func TestUnion1DEmpty(t *testing.T) {
	a := NewNDArray([]int{0}, []float64{})
	b := NewNDArray([]int{0}, []float64{})
	r := Union1D(a, b)
	if r.Size() != 0 {
		t.Fatal("expected empty result")
	}
}

// ---------- broadcasting.go ----------

func TestBroadcastToFewerDims(t *testing.T) {
	a := Zeros(2, 3)
	_, err := BroadcastTo(a, []int{3})
	if err == nil {
		t.Fatal("expected error for fewer target dims")
	}
}

func TestBroadcastToIncompatible(t *testing.T) {
	a := Zeros(2, 3)
	_, err := BroadcastTo(a, []int{2, 4})
	if err == nil {
		t.Fatal("expected error for incompatible broadcast")
	}
}
