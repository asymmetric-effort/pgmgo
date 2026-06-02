//go:build unit

package numgo

import (
	"math"
	"testing"
)

const einsumTol = 1e-12

func einsumClose(a, b float64) bool {
	return math.Abs(a-b) < einsumTol
}

func einsumCheckData(t *testing.T, label string, got *NDArray, wantShape []int, wantData []float64) {
	t.Helper()
	gotShape := got.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("%s: shape length mismatch: got %v, want %v", label, gotShape, wantShape)
	}
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Fatalf("%s: shape mismatch at dim %d: got %v, want %v", label, i, gotShape, wantShape)
		}
	}
	gotData := got.Data()
	if len(gotData) != len(wantData) {
		t.Fatalf("%s: data length mismatch: got %d, want %d", label, len(gotData), len(wantData))
	}
	for i := range wantData {
		if !einsumClose(gotData[i], wantData[i]) {
			t.Fatalf("%s: data mismatch at index %d: got %f, want %f", label, i, gotData[i], wantData[i])
		}
	}
}

// TestEinsumMatmul tests "ij,jk->ik" (matrix multiplication).
// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
// C = A @ B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
func TestEinsumMatmul(t *testing.T) {
	a := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b := NewNDArray([]int{2, 2}, []float64{5, 6, 7, 8})
	result, err := Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatalf("Einsum matmul: %v", err)
	}
	einsumCheckData(t, "matmul", result, []int{2, 2}, []float64{19, 22, 43, 50})
}

// TestEinsumMatmulNonSquare tests matmul with non-square matrices.
// A(2x3) @ B(3x2) -> C(2x2)
func TestEinsumMatmulNonSquare(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := NewNDArray([]int{3, 2}, []float64{7, 8, 9, 10, 11, 12})
	result, err := Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatalf("Einsum matmul non-square: %v", err)
	}
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	einsumCheckData(t, "matmul non-square", result, []int{2, 2}, []float64{58, 64, 139, 154})
}

// TestEinsumTrace tests "ii->" (matrix trace).
// A = [[1,2],[3,4]], trace = 1+4 = 5
func TestEinsumTrace(t *testing.T) {
	a := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	result, err := Einsum("ii->", a)
	if err != nil {
		t.Fatalf("Einsum trace: %v", err)
	}
	einsumCheckData(t, "trace", result, []int{}, []float64{5})
}

// TestEinsumTrace3x3 tests trace of a 3x3 identity matrix.
func TestEinsumTrace3x3(t *testing.T) {
	a := Eye(3)
	result, err := Einsum("ii->", a)
	if err != nil {
		t.Fatalf("Einsum trace 3x3: %v", err)
	}
	einsumCheckData(t, "trace 3x3", result, []int{}, []float64{3})
}

// TestEinsumSumAll tests "ij->" (sum all elements).
// A = [[1,2],[3,4]], sum = 10
func TestEinsumSumAll(t *testing.T) {
	a := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	result, err := Einsum("ij->", a)
	if err != nil {
		t.Fatalf("Einsum sum all: %v", err)
	}
	einsumCheckData(t, "sum all", result, []int{}, []float64{10})
}

// TestEinsumRowSums tests "ij->i" (row sums).
// A = [[1,2,3],[4,5,6]], row sums = [6, 15]
func TestEinsumRowSums(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	result, err := Einsum("ij->i", a)
	if err != nil {
		t.Fatalf("Einsum row sums: %v", err)
	}
	einsumCheckData(t, "row sums", result, []int{2}, []float64{6, 15})
}

// TestEinsumColumnSums tests "ij->j" (column sums).
// A = [[1,2,3],[4,5,6]], col sums = [5, 7, 9]
func TestEinsumColumnSums(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	result, err := Einsum("ij->j", a)
	if err != nil {
		t.Fatalf("Einsum column sums: %v", err)
	}
	einsumCheckData(t, "column sums", result, []int{3}, []float64{5, 7, 9})
}

// TestEinsumOuterProduct tests "i,j->ij" (outer product).
// a = [1,2,3], b = [4,5]
// outer = [[4,5],[8,10],[12,15]]
func TestEinsumOuterProduct(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5})
	result, err := Einsum("i,j->ij", a, b)
	if err != nil {
		t.Fatalf("Einsum outer product: %v", err)
	}
	einsumCheckData(t, "outer product", result, []int{3, 2}, []float64{4, 5, 8, 10, 12, 15})
}

// TestEinsumDotProduct tests "i,i->" (dot product).
// a = [1,2,3], b = [4,5,6], dot = 4+10+18 = 32
func TestEinsumDotProduct(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	result, err := Einsum("i,i->", a, b)
	if err != nil {
		t.Fatalf("Einsum dot product: %v", err)
	}
	einsumCheckData(t, "dot product", result, []int{}, []float64{32})
}

// TestEinsumBatchMatmul tests "bij,bjk->bik" (batch matrix multiply).
// Batch of 2 matrix multiplications, each 2x2.
// Batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
// Batch 1: [[1,0],[0,1]] @ [[9,8],[7,6]] = [[9,8],[7,6]]
func TestEinsumBatchMatmul(t *testing.T) {
	a := NewNDArray([]int{2, 2, 2}, []float64{
		1, 2, 3, 4, // batch 0
		1, 0, 0, 1, // batch 1 (identity)
	})
	b := NewNDArray([]int{2, 2, 2}, []float64{
		5, 6, 7, 8, // batch 0
		9, 8, 7, 6, // batch 1
	})
	result, err := Einsum("bij,bjk->bik", a, b)
	if err != nil {
		t.Fatalf("Einsum batch matmul: %v", err)
	}
	einsumCheckData(t, "batch matmul", result, []int{2, 2, 2}, []float64{
		19, 22, 43, 50, // batch 0
		9, 8, 7, 6, // batch 1
	})
}

// TestEinsumImplicitMode tests implicit mode (no "->").
// "ij" with a 2x3 matrix: labels i and j each appear once, so output is "ij" (copy).
func TestEinsumImplicitMode(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	result, err := Einsum("ij", a)
	if err != nil {
		t.Fatalf("Einsum implicit mode: %v", err)
	}
	einsumCheckData(t, "implicit copy", result, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
}

// TestEinsumImplicitMatmul tests implicit mode for matmul.
// "ij,jk" -> j appears twice so is summed; output labels are "ik" (sorted).
func TestEinsumImplicitMatmul(t *testing.T) {
	a := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b := NewNDArray([]int{2, 2}, []float64{5, 6, 7, 8})
	result, err := Einsum("ij,jk", a, b)
	if err != nil {
		t.Fatalf("Einsum implicit matmul: %v", err)
	}
	einsumCheckData(t, "implicit matmul", result, []int{2, 2}, []float64{19, 22, 43, 50})
}

// TestEinsumImplicitDiag tests implicit mode "ii" which sums repeated labels.
// "ii" -> no labels appear exactly once, so output is scalar (trace).
func TestEinsumImplicitDiag(t *testing.T) {
	a := NewNDArray([]int{3, 3}, []float64{
		1, 0, 0,
		0, 2, 0,
		0, 0, 3,
	})
	result, err := Einsum("ii", a)
	if err != nil {
		t.Fatalf("Einsum implicit diag: %v", err)
	}
	// i appears twice -> summed out -> scalar trace = 6
	einsumCheckData(t, "implicit diag", result, []int{}, []float64{6})
}

// TestEinsumDiagonalExtract tests "ii->i" (extract diagonal).
// A = [[1,2,3],[4,5,6],[7,8,9]], diag = [1,5,9]
func TestEinsumDiagonalExtract(t *testing.T) {
	a := NewNDArray([]int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	result, err := Einsum("ii->i", a)
	if err != nil {
		t.Fatalf("Einsum diagonal extract: %v", err)
	}
	einsumCheckData(t, "diagonal extract", result, []int{3}, []float64{1, 5, 9})
}

// TestEinsumTranspose tests "ij->ji" (transpose).
// A = [[1,2,3],[4,5,6]], A^T = [[1,4],[2,5],[3,6]]
func TestEinsumTranspose(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	result, err := Einsum("ij->ji", a)
	if err != nil {
		t.Fatalf("Einsum transpose: %v", err)
	}
	einsumCheckData(t, "transpose", result, []int{3, 2}, []float64{1, 4, 2, 5, 3, 6})
}

// TestEinsumElementwiseMul tests "ij,ij->ij" (Hadamard product).
func TestEinsumElementwiseMul(t *testing.T) {
	a := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	b := NewNDArray([]int{2, 2}, []float64{5, 6, 7, 8})
	result, err := Einsum("ij,ij->ij", a, b)
	if err != nil {
		t.Fatalf("Einsum elementwise mul: %v", err)
	}
	einsumCheckData(t, "elementwise mul", result, []int{2, 2}, []float64{5, 12, 21, 32})
}

// TestEinsumScalarResult tests a single-operand sum to scalar.
// "i->" with [1,2,3,4] = 10
func TestEinsumScalarResult(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	result, err := Einsum("i->", a)
	if err != nil {
		t.Fatalf("Einsum scalar result: %v", err)
	}
	einsumCheckData(t, "scalar result", result, []int{}, []float64{10})
}

// TestEinsumErrorOperandCount tests that mismatched operand count is caught.
func TestEinsumErrorOperandCount(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	_, err := Einsum("ij,jk->ik", a)
	if err == nil {
		t.Fatal("expected error for wrong operand count")
	}
}

// TestEinsumErrorDimMismatch tests that dimension mismatch is caught.
func TestEinsumErrorDimMismatch(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	_, err := Einsum("i->", a)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
}

// TestEinsumErrorSizeInconsistency tests that label size inconsistency is caught.
func TestEinsumErrorSizeInconsistency(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := NewNDArray([]int{4, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	_, err := Einsum("ij,jk->ik", a, b)
	if err == nil {
		t.Fatal("expected error for inconsistent label sizes")
	}
}

// TestEinsumErrorEmptyNotation tests that empty notation is caught.
func TestEinsumErrorEmptyNotation(t *testing.T) {
	_, err := Einsum("")
	if err == nil {
		t.Fatal("expected error for empty notation")
	}
}

// TestEinsumErrorInvalidChar tests that invalid subscript characters are caught.
func TestEinsumErrorInvalidChar(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	_, err := Einsum("I->", a)
	if err == nil {
		t.Fatal("expected error for invalid subscript character")
	}
}

// TestEinsumBatchMatmulNonSquare tests batch matmul with non-square inner matrices.
// A(2,2,3) @ B(2,3,4) -> C(2,2,4)
func TestEinsumBatchMatmulNonSquare(t *testing.T) {
	a := NewNDArray([]int{2, 2, 3}, []float64{
		1, 2, 3, 4, 5, 6, // batch 0: [[1,2,3],[4,5,6]]
		1, 0, 0, 0, 1, 0, // batch 1: [[1,0,0],[0,1,0]]
	})
	b := NewNDArray([]int{2, 3, 4}, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // batch 0
		1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, // batch 1 (3x4 identity-like)
	})
	result, err := Einsum("bij,bjk->bik", a, b)
	if err != nil {
		t.Fatalf("Einsum batch matmul non-square: %v", err)
	}

	// Batch 0: [[1,2,3],[4,5,6]] @ [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
	// Row 0: [1+10+27, 2+12+30, 3+14+33, 4+16+36] = [38, 44, 50, 56]
	// Row 1: [4+25+54, 8+30+60, 12+35+66, 16+40+72] = [83, 98, 113, 128]
	// Batch 1: [[1,0,0],[0,1,0]] @ [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
	// Row 0: [1,0,0,0]
	// Row 1: [0,1,0,0]
	einsumCheckData(t, "batch matmul non-square", result, []int{2, 2, 4}, []float64{
		38, 44, 50, 56, 83, 98, 113, 128,
		1, 0, 0, 0, 0, 1, 0, 0,
	})
}
