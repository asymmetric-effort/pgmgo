//go:build unit

package numgo

import "testing"

// ---------------------------------------------------------------------------
// Intersect1D
// ---------------------------------------------------------------------------

func TestIntersect1DBasic(t *testing.T) {
	a := FromSlice([]float64{1, 3, 5, 7})
	b := FromSlice([]float64{2, 3, 5, 8})
	got := Intersect1D(a, b)
	assertSetData(t, "Intersect1DBasic", got, []float64{3, 5})
}

func TestIntersect1DNoOverlap(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	got := Intersect1D(a, b)
	assertSetData(t, "Intersect1DNoOverlap", got, []float64{})
}

func TestIntersect1DFullOverlap(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{3, 2, 1})
	got := Intersect1D(a, b)
	assertSetData(t, "Intersect1DFull", got, []float64{1, 2, 3})
}

func TestIntersect1DWithDuplicates(t *testing.T) {
	a := FromSlice([]float64{1, 1, 2, 2, 3})
	b := FromSlice([]float64{2, 2, 3, 3, 4})
	got := Intersect1D(a, b)
	assertSetData(t, "Intersect1DDups", got, []float64{2, 3})
}

func TestIntersect1DEmpty(t *testing.T) {
	a := NewNDArray([]int{0}, []float64{})
	b := FromSlice([]float64{1, 2})
	got := Intersect1D(a, b)
	if got.Size() != 0 {
		t.Fatalf("Intersect1DEmpty: expected size 0, got %d", got.Size())
	}
}

// ---------------------------------------------------------------------------
// Union1D
// ---------------------------------------------------------------------------

func TestUnion1DBasic(t *testing.T) {
	a := FromSlice([]float64{1, 3, 5})
	b := FromSlice([]float64{2, 3, 4})
	got := Union1D(a, b)
	assertSetData(t, "Union1DBasic", got, []float64{1, 2, 3, 4, 5})
}

func TestUnion1DIdentical(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{1, 2, 3})
	got := Union1D(a, b)
	assertSetData(t, "Union1DIdentical", got, []float64{1, 2, 3})
}

func TestUnion1DNoOverlap(t *testing.T) {
	a := FromSlice([]float64{1, 2})
	b := FromSlice([]float64{3, 4})
	got := Union1D(a, b)
	assertSetData(t, "Union1DNoOverlap", got, []float64{1, 2, 3, 4})
}

func TestUnion1DWithDuplicates(t *testing.T) {
	a := FromSlice([]float64{1, 1, 2})
	b := FromSlice([]float64{2, 3, 3})
	got := Union1D(a, b)
	assertSetData(t, "Union1DDups", got, []float64{1, 2, 3})
}

func TestUnion1DEmptyA(t *testing.T) {
	a := NewNDArray([]int{0}, []float64{})
	b := FromSlice([]float64{1, 2})
	got := Union1D(a, b)
	assertSetData(t, "Union1DEmptyA", got, []float64{1, 2})
}

// ---------------------------------------------------------------------------
// SetDiff1D
// ---------------------------------------------------------------------------

func TestSetDiff1DBasic(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5})
	b := FromSlice([]float64{2, 4})
	got := SetDiff1D(a, b)
	assertSetData(t, "SetDiff1DBasic", got, []float64{1, 3, 5})
}

func TestSetDiff1DNoOverlap(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	got := SetDiff1D(a, b)
	assertSetData(t, "SetDiff1DNoOverlap", got, []float64{1, 2, 3})
}

func TestSetDiff1DFullOverlap(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{1, 2, 3})
	got := SetDiff1D(a, b)
	if got.Size() != 0 {
		t.Fatalf("SetDiff1DFull: expected size 0, got %d", got.Size())
	}
}

func TestSetDiff1DWithDuplicates(t *testing.T) {
	a := FromSlice([]float64{1, 1, 2, 3, 3})
	b := FromSlice([]float64{1})
	got := SetDiff1D(a, b)
	assertSetData(t, "SetDiff1DDups", got, []float64{2, 3})
}

func TestSetDiff1DEmptyB(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := NewNDArray([]int{0}, []float64{})
	got := SetDiff1D(a, b)
	assertSetData(t, "SetDiff1DEmptyB", got, []float64{1, 2, 3})
}

func TestSetDiff1DWithNegatives(t *testing.T) {
	a := FromSlice([]float64{-3, -1, 0, 2, 4})
	b := FromSlice([]float64{-1, 2})
	got := SetDiff1D(a, b)
	assertSetData(t, "SetDiff1DNeg", got, []float64{-3, 0, 4})
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

func assertSetData(t *testing.T, label string, got *NDArray, want []float64) {
	t.Helper()
	d := got.Data()
	if len(d) != len(want) {
		t.Fatalf("%s: length %d, want %d (got %v)", label, len(d), len(want), d)
	}
	for i := range d {
		if d[i] != want[i] {
			t.Fatalf("%s: index %d = %g, want %g (full: %v)", label, i, d[i], want[i], d)
		}
	}
}
