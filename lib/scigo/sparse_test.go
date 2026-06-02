//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func denseEqual(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > 1e-12 {
				return false
			}
		}
	}
	return true
}

func vecEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Reference dense matrix used across many tests:
//
//	[1  0  2]
//	[0  0  3]
//	[4  5  6]
var refDense = [][]float64{
	{1, 0, 2},
	{0, 0, 3},
	{4, 5, 6},
}

// ---------------------------------------------------------------------------
// COO Tests
// ---------------------------------------------------------------------------

func TestNewCOO(t *testing.T) {
	rows := []int{0, 0, 1, 2, 2, 2}
	cols := []int{0, 2, 2, 0, 1, 2}
	vals := []float64{1, 2, 3, 4, 5, 6}
	shape := [2]int{3, 3}

	coo, err := NewCOO(rows, cols, vals, shape)
	if err != nil {
		t.Fatalf("NewCOO: unexpected error: %v", err)
	}
	if coo.Shape() != shape {
		t.Errorf("Shape = %v, want %v", coo.Shape(), shape)
	}
	if coo.NNZ() != 6 {
		t.Errorf("NNZ = %d, want 6", coo.NNZ())
	}
}

func TestNewCOO_LengthMismatch(t *testing.T) {
	_, err := NewCOO([]int{0, 1}, []int{0}, []float64{1, 2}, [2]int{2, 2})
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
}

func TestNewCOO_OutOfBounds(t *testing.T) {
	_, err := NewCOO([]int{5}, []int{0}, []float64{1}, [2]int{3, 3})
	if err == nil {
		t.Error("expected error for out-of-bounds row index")
	}
	_, err = NewCOO([]int{0}, []int{5}, []float64{1}, [2]int{3, 3})
	if err == nil {
		t.Error("expected error for out-of-bounds col index")
	}
}

func TestNewCOO_BadShape(t *testing.T) {
	_, err := NewCOO(nil, nil, nil, [2]int{0, 3})
	if err == nil {
		t.Error("expected error for zero shape dimension")
	}
}

func TestCOOGet(t *testing.T) {
	coo, _ := NewCOO(
		[]int{0, 0, 1, 2, 2, 2},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	tests := [][3]float64{
		{0, 0, 1}, {0, 1, 0}, {0, 2, 2},
		{1, 0, 0}, {1, 1, 0}, {1, 2, 3},
		{2, 0, 4}, {2, 1, 5}, {2, 2, 6},
	}
	for _, tt := range tests {
		got := coo.Get(int(tt[0]), int(tt[1]))
		if got != tt[2] {
			t.Errorf("Get(%d,%d) = %v, want %v", int(tt[0]), int(tt[1]), got, tt[2])
		}
	}
}

func TestCOOGetDuplicates(t *testing.T) {
	// Two entries at (0,0): should sum.
	coo, _ := NewCOO([]int{0, 0}, []int{0, 0}, []float64{3, 7}, [2]int{1, 1})
	if got := coo.Get(0, 0); got != 10 {
		t.Errorf("Get(0,0) with duplicates = %v, want 10", got)
	}
}

func TestCOOSet(t *testing.T) {
	coo, _ := NewCOO(nil, nil, nil, [2]int{2, 2})
	coo.Set(0, 1, 5)
	coo.Set(1, 0, 3)
	if coo.NNZ() != 2 {
		t.Errorf("NNZ after Set = %d, want 2", coo.NNZ())
	}
	if got := coo.Get(0, 1); got != 5 {
		t.Errorf("Get(0,1) = %v, want 5", got)
	}
	if got := coo.Get(1, 0); got != 3 {
		t.Errorf("Get(1,0) = %v, want 3", got)
	}
}

func TestCOOToDense(t *testing.T) {
	coo, _ := NewCOO(
		[]int{0, 0, 1, 2, 2, 2},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	dense := coo.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("ToDense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestCOOToCSR(t *testing.T) {
	coo, _ := NewCOO(
		[]int{0, 0, 1, 2, 2, 2},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	csr := coo.ToCSR()
	dense := csr.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("COO->CSR->Dense =\n%v\nwant\n%v", dense, refDense)
	}
}

// ---------------------------------------------------------------------------
// CSR Tests
// ---------------------------------------------------------------------------

func TestNewCSR(t *testing.T) {
	indptr := []int{0, 2, 3, 6}
	indices := []int{0, 2, 2, 0, 1, 2}
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := [2]int{3, 3}

	csr, err := NewCSR(indptr, indices, data, shape)
	if err != nil {
		t.Fatalf("NewCSR: unexpected error: %v", err)
	}
	if csr.Shape() != shape {
		t.Errorf("Shape = %v, want %v", csr.Shape(), shape)
	}
	if csr.NNZ() != 6 {
		t.Errorf("NNZ = %d, want 6", csr.NNZ())
	}
}

func TestNewCSR_Errors(t *testing.T) {
	tests := []struct {
		name    string
		indptr  []int
		indices []int
		data    []float64
		shape   [2]int
	}{
		{"bad shape", []int{0, 0}, []int{}, []float64{}, [2]int{0, 3}},
		{"wrong indptr len", []int{0, 1}, []int{0}, []float64{1}, [2]int{3, 3}},
		{"indices/data mismatch", []int{0, 1, 1, 1}, []int{0, 1}, []float64{1}, [2]int{3, 3}},
		{"indptr[0] != 0", []int{1, 1, 1, 1}, []int{}, []float64{}, [2]int{3, 3}},
		{"indptr[-1] != nnz", []int{0, 1, 1, 2}, []int{0}, []float64{1}, [2]int{3, 3}},
		{"decreasing indptr", []int{0, 2, 1, 3}, []int{0, 1, 2}, []float64{1, 2, 3}, [2]int{3, 3}},
		{"col out of bounds", []int{0, 1, 1, 1}, []int{5}, []float64{1}, [2]int{3, 3}},
	}
	for _, tt := range tests {
		_, err := NewCSR(tt.indptr, tt.indices, tt.data, tt.shape)
		if err == nil {
			t.Errorf("NewCSR(%s): expected error", tt.name)
		}
	}
}

func TestCSRGet(t *testing.T) {
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	tests := [][3]float64{
		{0, 0, 1}, {0, 1, 0}, {0, 2, 2},
		{1, 0, 0}, {1, 1, 0}, {1, 2, 3},
		{2, 0, 4}, {2, 1, 5}, {2, 2, 6},
	}
	for _, tt := range tests {
		got := csr.Get(int(tt[0]), int(tt[1]))
		if got != tt[2] {
			t.Errorf("Get(%d,%d) = %v, want %v", int(tt[0]), int(tt[1]), got, tt[2])
		}
	}
}

func TestCSRRow(t *testing.T) {
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	idx, vals := csr.Row(0)
	if !intSliceEqual(idx, []int{0, 2}) {
		t.Errorf("Row(0) indices = %v, want [0 2]", idx)
	}
	if !vecEqual(vals, []float64{1, 2}, 1e-12) {
		t.Errorf("Row(0) values = %v, want [1 2]", vals)
	}

	idx, vals = csr.Row(1)
	if !intSliceEqual(idx, []int{2}) {
		t.Errorf("Row(1) indices = %v, want [2]", idx)
	}
	if !vecEqual(vals, []float64{3}, 1e-12) {
		t.Errorf("Row(1) values = %v, want [3]", vals)
	}
}

func TestCSRToDense(t *testing.T) {
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	dense := csr.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("ToDense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestCSRToCOO(t *testing.T) {
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	coo := csr.ToCOO()
	dense := coo.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("CSR->COO->Dense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestCSRTranspose(t *testing.T) {
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	csrT := csr.Transpose()
	if csrT.Shape() != [2]int{3, 3} {
		t.Errorf("Transpose shape = %v, want [3 3]", csrT.Shape())
	}
	expected := [][]float64{
		{1, 0, 4},
		{0, 0, 5},
		{2, 3, 6},
	}
	dense := csrT.ToDense()
	if !denseEqual(dense, expected) {
		t.Errorf("Transpose->Dense =\n%v\nwant\n%v", dense, expected)
	}
}

func TestCSRTransposeNonSquare(t *testing.T) {
	// 2x3 matrix:
	// [1 0 2]
	// [0 3 0]
	csr, _ := NewCSR(
		[]int{0, 2, 3},
		[]int{0, 2, 1},
		[]float64{1, 2, 3},
		[2]int{2, 3},
	)
	csrT := csr.Transpose()
	if csrT.Shape() != [2]int{3, 2} {
		t.Errorf("Transpose shape = %v, want [3 2]", csrT.Shape())
	}
	expected := [][]float64{
		{1, 0},
		{0, 3},
		{2, 0},
	}
	dense := csrT.ToDense()
	if !denseEqual(dense, expected) {
		t.Errorf("Transpose->Dense =\n%v\nwant\n%v", dense, expected)
	}
}

func TestCSRMulVec(t *testing.T) {
	// A = refDense, x = [1, 2, 3]
	// A*x = [1*1+0*2+2*3, 0*1+0*2+3*3, 4*1+5*2+6*3] = [7, 9, 32]
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	x := []float64{1, 2, 3}
	y := csr.MulVec(x)
	expected := []float64{7, 9, 32}
	if !vecEqual(y, expected, 1e-12) {
		t.Errorf("MulVec = %v, want %v", y, expected)
	}
}

func TestCSRMulVecPanic(t *testing.T) {
	csr, _ := NewCSR([]int{0, 1, 1}, []int{0}, []float64{1}, [2]int{2, 2})
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for dimension mismatch")
		}
	}()
	csr.MulVec([]float64{1, 2, 3}) // wrong length
}

func TestCSRMulDense(t *testing.T) {
	// A (3x3) * B (3x2)
	// A = refDense
	// B = [[1,2],[3,4],[5,6]]
	// A*B = [[1*1+0*3+2*5, 1*2+0*4+2*6], [0+0+3*5, 0+0+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
	//     = [[11, 14], [15, 18], [49, 64]]
	csr, _ := NewCSR(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	B := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	result := csr.MulDense(B)
	expected := [][]float64{
		{11, 14},
		{15, 18},
		{49, 64},
	}
	if !denseEqual(result, expected) {
		t.Errorf("MulDense =\n%v\nwant\n%v", result, expected)
	}
}

func TestCSRMulDensePanic(t *testing.T) {
	csr, _ := NewCSR([]int{0, 1, 1}, []int{0}, []float64{1}, [2]int{2, 2})
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for dimension mismatch")
		}
	}()
	csr.MulDense([][]float64{{1}}) // 1 row, need 2
}

// ---------------------------------------------------------------------------
// CSC Tests
// ---------------------------------------------------------------------------

func TestNewCSC(t *testing.T) {
	// Column 0: rows 0,2 vals 1,4
	// Column 1: row 2  val 5
	// Column 2: rows 0,1,2 vals 2,3,6
	indptr := []int{0, 2, 3, 6}
	indices := []int{0, 2, 2, 0, 1, 2}
	data := []float64{1, 4, 5, 2, 3, 6}
	shape := [2]int{3, 3}

	csc, err := NewCSC(indptr, indices, data, shape)
	if err != nil {
		t.Fatalf("NewCSC: unexpected error: %v", err)
	}
	if csc.Shape() != shape {
		t.Errorf("Shape = %v, want %v", csc.Shape(), shape)
	}
	if csc.NNZ() != 6 {
		t.Errorf("NNZ = %d, want 6", csc.NNZ())
	}
}

func TestNewCSC_Errors(t *testing.T) {
	tests := []struct {
		name    string
		indptr  []int
		indices []int
		data    []float64
		shape   [2]int
	}{
		{"bad shape", []int{0, 0}, []int{}, []float64{}, [2]int{3, 0}},
		{"wrong indptr len", []int{0, 1}, []int{0}, []float64{1}, [2]int{3, 3}},
		{"indices/data mismatch", []int{0, 1, 1, 1}, []int{0, 1}, []float64{1}, [2]int{3, 3}},
		{"indptr[0] != 0", []int{1, 1, 1, 1}, []int{}, []float64{}, [2]int{3, 3}},
		{"row out of bounds", []int{0, 1, 1, 1}, []int{5}, []float64{1}, [2]int{3, 3}},
	}
	for _, tt := range tests {
		_, err := NewCSC(tt.indptr, tt.indices, tt.data, tt.shape)
		if err == nil {
			t.Errorf("NewCSC(%s): expected error", tt.name)
		}
	}
}

func TestCSCGet(t *testing.T) {
	csc, _ := NewCSC(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 4, 5, 2, 3, 6},
		[2]int{3, 3},
	)
	tests := [][3]float64{
		{0, 0, 1}, {0, 1, 0}, {0, 2, 2},
		{1, 0, 0}, {1, 1, 0}, {1, 2, 3},
		{2, 0, 4}, {2, 1, 5}, {2, 2, 6},
	}
	for _, tt := range tests {
		got := csc.Get(int(tt[0]), int(tt[1]))
		if got != tt[2] {
			t.Errorf("Get(%d,%d) = %v, want %v", int(tt[0]), int(tt[1]), got, tt[2])
		}
	}
}

func TestCSCCol(t *testing.T) {
	csc, _ := NewCSC(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 4, 5, 2, 3, 6},
		[2]int{3, 3},
	)
	idx, vals := csc.Col(0)
	if !intSliceEqual(idx, []int{0, 2}) {
		t.Errorf("Col(0) indices = %v, want [0 2]", idx)
	}
	if !vecEqual(vals, []float64{1, 4}, 1e-12) {
		t.Errorf("Col(0) values = %v, want [1 4]", vals)
	}

	idx, vals = csc.Col(1)
	if !intSliceEqual(idx, []int{2}) {
		t.Errorf("Col(1) indices = %v, want [2]", idx)
	}
	if !vecEqual(vals, []float64{5}, 1e-12) {
		t.Errorf("Col(1) values = %v, want [5]", vals)
	}
}

func TestCSCToDense(t *testing.T) {
	csc, _ := NewCSC(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 4, 5, 2, 3, 6},
		[2]int{3, 3},
	)
	dense := csc.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("ToDense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestCSCToCSR(t *testing.T) {
	csc, _ := NewCSC(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 4, 5, 2, 3, 6},
		[2]int{3, 3},
	)
	csr := csc.ToCSR()
	dense := csr.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("CSC->CSR->Dense =\n%v\nwant\n%v", dense, refDense)
	}
}

// ---------------------------------------------------------------------------
// Dense conversion helpers
// ---------------------------------------------------------------------------

func TestDenseToCSR(t *testing.T) {
	csr := DenseToCSR(refDense)
	if csr.Shape() != [2]int{3, 3} {
		t.Errorf("Shape = %v, want [3 3]", csr.Shape())
	}
	if csr.NNZ() != 6 {
		t.Errorf("NNZ = %d, want 6", csr.NNZ())
	}
	dense := csr.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("DenseToCSR->ToDense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestDenseToCOO(t *testing.T) {
	coo := DenseToCOO(refDense)
	if coo.Shape() != [2]int{3, 3} {
		t.Errorf("Shape = %v, want [3 3]", coo.Shape())
	}
	if coo.NNZ() != 6 {
		t.Errorf("NNZ = %d, want 6", coo.NNZ())
	}
	dense := coo.ToDense()
	if !denseEqual(dense, refDense) {
		t.Errorf("DenseToCOO->ToDense =\n%v\nwant\n%v", dense, refDense)
	}
}

func TestDenseToCSREmpty(t *testing.T) {
	csr := DenseToCSR([][]float64{})
	if csr.NNZ() != 0 {
		t.Errorf("NNZ = %d, want 0", csr.NNZ())
	}
}

func TestDenseToCOOEmpty(t *testing.T) {
	coo := DenseToCOO([][]float64{})
	if coo.NNZ() != 0 {
		t.Errorf("NNZ = %d, want 0", coo.NNZ())
	}
}

// ---------------------------------------------------------------------------
// Round-trip conversion tests
// ---------------------------------------------------------------------------

func TestRoundTripCOO_CSR_COO(t *testing.T) {
	coo, _ := NewCOO(
		[]int{0, 0, 1, 2, 2, 2},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 2, 3, 4, 5, 6},
		[2]int{3, 3},
	)
	csr := coo.ToCSR()
	coo2 := csr.ToCOO()
	if !denseEqual(coo2.ToDense(), refDense) {
		t.Error("COO->CSR->COO round-trip failed")
	}
}

func TestRoundTripCSC_CSR_Dense(t *testing.T) {
	csc, _ := NewCSC(
		[]int{0, 2, 3, 6},
		[]int{0, 2, 2, 0, 1, 2},
		[]float64{1, 4, 5, 2, 3, 6},
		[2]int{3, 3},
	)
	csr := csc.ToCSR()
	dense := csr.ToDense()
	if !denseEqual(dense, refDense) {
		t.Error("CSC->CSR->Dense round-trip failed")
	}
}

func TestRoundTripDense_CSR_Transpose_Dense(t *testing.T) {
	csr := DenseToCSR(refDense)
	csrT := csr.Transpose()
	csrTT := csrT.Transpose()
	dense := csrTT.ToDense()
	if !denseEqual(dense, refDense) {
		t.Error("Dense->CSR->T->T->Dense round-trip failed")
	}
}

// ---------------------------------------------------------------------------
// Identity / edge cases
// ---------------------------------------------------------------------------

func TestCSRIdentityMulVec(t *testing.T) {
	// 3x3 identity
	csr, _ := NewCSR(
		[]int{0, 1, 2, 3},
		[]int{0, 1, 2},
		[]float64{1, 1, 1},
		[2]int{3, 3},
	)
	x := []float64{7, 8, 9}
	y := csr.MulVec(x)
	if !vecEqual(y, x, 1e-12) {
		t.Errorf("Identity MulVec = %v, want %v", y, x)
	}
}

func TestCSRAllZeroDense(t *testing.T) {
	dense := [][]float64{
		{0, 0},
		{0, 0},
	}
	csr := DenseToCSR(dense)
	if csr.NNZ() != 0 {
		t.Errorf("NNZ = %d, want 0 for all-zero matrix", csr.NNZ())
	}
	got := csr.ToDense()
	if !denseEqual(got, dense) {
		t.Errorf("all-zero round-trip failed")
	}
}

func TestCSRSingleElement(t *testing.T) {
	csr, _ := NewCSR([]int{0, 1}, []int{0}, []float64{42}, [2]int{1, 1})
	if csr.Get(0, 0) != 42 {
		t.Errorf("Get(0,0) = %v, want 42", csr.Get(0, 0))
	}
}

func TestCOOToCSRDuplicatesSummed(t *testing.T) {
	// Two entries at (0,0) with values 3 and 7 should sum to 10.
	coo, _ := NewCOO([]int{0, 0}, []int{0, 0}, []float64{3, 7}, [2]int{1, 1})
	csr := coo.ToCSR()
	if csr.Get(0, 0) != 10 {
		t.Errorf("Duplicate sum = %v, want 10", csr.Get(0, 0))
	}
}

func TestCSRMulVecNonSquare(t *testing.T) {
	// 2x3 matrix * 3-vector
	// [[1 0 2], [0 3 0]] * [1,2,3] = [7, 6]
	csr, _ := NewCSR(
		[]int{0, 2, 3},
		[]int{0, 2, 1},
		[]float64{1, 2, 3},
		[2]int{2, 3},
	)
	y := csr.MulVec([]float64{1, 2, 3})
	expected := []float64{7, 6}
	if !vecEqual(y, expected, 1e-12) {
		t.Errorf("MulVec non-square = %v, want %v", y, expected)
	}
}

func TestCSRMulDenseNonSquare(t *testing.T) {
	// 2x3 * 3x1
	csr, _ := NewCSR(
		[]int{0, 2, 3},
		[]int{0, 2, 1},
		[]float64{1, 2, 3},
		[2]int{2, 3},
	)
	B := [][]float64{{1}, {2}, {3}}
	result := csr.MulDense(B)
	expected := [][]float64{{7}, {6}}
	if !denseEqual(result, expected) {
		t.Errorf("MulDense non-square =\n%v\nwant\n%v", result, expected)
	}
}

func TestCSCToCSRNonSquare(t *testing.T) {
	// 2x3 matrix in CSC:
	// [[1 0 2], [0 3 0]]
	// Col 0: row 0 val 1
	// Col 1: row 1 val 3
	// Col 2: row 0 val 2
	csc, _ := NewCSC(
		[]int{0, 1, 2, 3},
		[]int{0, 1, 0},
		[]float64{1, 3, 2},
		[2]int{2, 3},
	)
	csr := csc.ToCSR()
	expected := [][]float64{{1, 0, 2}, {0, 3, 0}}
	dense := csr.ToDense()
	if !denseEqual(dense, expected) {
		t.Errorf("CSC->CSR non-square =\n%v\nwant\n%v", dense, expected)
	}
}

// ---------------------------------------------------------------------------
// Data isolation tests (inputs should be copied)
// ---------------------------------------------------------------------------

func TestCOOInputIsolation(t *testing.T) {
	rows := []int{0}
	cols := []int{0}
	vals := []float64{1}
	coo, _ := NewCOO(rows, cols, vals, [2]int{1, 1})
	rows[0] = 99
	vals[0] = 99
	if coo.Get(0, 0) != 1 {
		t.Error("COO data should be isolated from input slice mutation")
	}
}

func TestCSRInputIsolation(t *testing.T) {
	indptr := []int{0, 1}
	indices := []int{0}
	data := []float64{5}
	csr, _ := NewCSR(indptr, indices, data, [2]int{1, 1})
	data[0] = 99
	if csr.Get(0, 0) != 5 {
		t.Error("CSR data should be isolated from input slice mutation")
	}
}
