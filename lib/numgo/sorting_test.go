//go:build unit

package numgo

import (
	"testing"
)

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

func TestSort1D(t *testing.T) {
	a := FromSlice([]float64{5, 3, 1, 4, 2})
	got := Sort(a, 0)
	want := []float64{1, 2, 3, 4, 5}
	assertData(t, "Sort1D", got, want)
}

func TestSort1DAlreadySorted(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	got := Sort(a, 0)
	assertData(t, "Sort1DAlreadySorted", got, []float64{1, 2, 3})
}

func TestSort1DWithDuplicates(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2, 1, 3})
	got := Sort(a, 0)
	assertData(t, "Sort1DDups", got, []float64{1, 1, 2, 3, 3})
}

func TestSort1DWithNegatives(t *testing.T) {
	a := FromSlice([]float64{0, -3, 2, -1})
	got := Sort(a, 0)
	assertData(t, "Sort1DNeg", got, []float64{-3, -1, 0, 2})
}

func TestSort2DAxis0(t *testing.T) {
	// [[3, 1], [2, 4]] -> sort columns -> [[2, 1], [3, 4]]
	a := NewNDArray([]int{2, 2}, []float64{3, 1, 2, 4})
	got := Sort(a, 0)
	assertData(t, "Sort2DAxis0", got, []float64{2, 1, 3, 4})
	assertShape(t, "Sort2DAxis0", got, []int{2, 2})
}

func TestSort2DAxis1(t *testing.T) {
	// [[3, 1], [4, 2]] -> sort rows -> [[1, 3], [2, 4]]
	a := NewNDArray([]int{2, 2}, []float64{3, 1, 4, 2})
	got := Sort(a, 1)
	assertData(t, "Sort2DAxis1", got, []float64{1, 3, 2, 4})
	assertShape(t, "Sort2DAxis1", got, []int{2, 2})
}

func TestSort2DLarger(t *testing.T) {
	// 2x3 array, sort along axis 1 (rows)
	a := NewNDArray([]int{2, 3}, []float64{
		9, 3, 6,
		2, 8, 1,
	})
	got := Sort(a, 1)
	assertData(t, "Sort2DLarger", got, []float64{3, 6, 9, 1, 2, 8})
}

func TestSortDoesNotMutateOriginal(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2})
	_ = Sort(a, 0)
	assertData(t, "SortNoMutate", a, []float64{3, 1, 2})
}

func TestSortPanicsOnBadAxis(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for out-of-range axis")
		}
	}()
	Sort(FromSlice([]float64{1, 2}), 1)
}

// ---------------------------------------------------------------------------
// ArgSort
// ---------------------------------------------------------------------------

func TestArgSort1D(t *testing.T) {
	a := FromSlice([]float64{30, 10, 20})
	got := ArgSort(a, 0)
	// sorted order: 10(1), 20(2), 30(0) => indices [1, 2, 0]
	assertData(t, "ArgSort1D", got, []float64{1, 2, 0})
}

func TestArgSort1DAlreadySorted(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	got := ArgSort(a, 0)
	assertData(t, "ArgSort1DSorted", got, []float64{0, 1, 2})
}

func TestArgSort2DAxis1(t *testing.T) {
	// [[30, 10, 20]] => argsort along axis 1 => [[1, 2, 0]]
	a := NewNDArray([]int{1, 3}, []float64{30, 10, 20})
	got := ArgSort(a, 1)
	assertData(t, "ArgSort2DAxis1", got, []float64{1, 2, 0})
	assertShape(t, "ArgSort2DAxis1", got, []int{1, 3})
}

func TestArgSort2DAxis0(t *testing.T) {
	// [[3, 1], [1, 3]] => argsort along axis 0
	// col 0: [3,1] => [1,0]; col 1: [1,3] => [0,1]
	a := NewNDArray([]int{2, 2}, []float64{3, 1, 1, 3})
	got := ArgSort(a, 0)
	assertData(t, "ArgSort2DAxis0", got, []float64{1, 0, 0, 1})
}

// ---------------------------------------------------------------------------
// Unique
// ---------------------------------------------------------------------------

func TestUnique(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2, 1, 3, 2})
	got := Unique(a)
	assertData(t, "Unique", got, []float64{1, 2, 3})
}

func TestUniqueNoDuplicates(t *testing.T) {
	a := FromSlice([]float64{5, 3, 1})
	got := Unique(a)
	assertData(t, "UniqueNoDups", got, []float64{1, 3, 5})
}

func TestUniqueSingleElement(t *testing.T) {
	a := FromSlice([]float64{42})
	got := Unique(a)
	assertData(t, "UniqueSingle", got, []float64{42})
}

func TestUniqueAllSame(t *testing.T) {
	a := FromSlice([]float64{7, 7, 7})
	got := Unique(a)
	assertData(t, "UniqueAllSame", got, []float64{7})
}

func TestUnique2DFlattened(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 2, 3, 4})
	got := Unique(a)
	assertData(t, "Unique2D", got, []float64{1, 2, 3, 4})
	assertShape(t, "Unique2D", got, []int{4})
}

// ---------------------------------------------------------------------------
// Where
// ---------------------------------------------------------------------------

func TestWhereBasic(t *testing.T) {
	cond := FromSlice([]float64{1, 0, 1, 0})
	x := FromSlice([]float64{10, 20, 30, 40})
	y := FromSlice([]float64{-1, -2, -3, -4})
	got := Where(cond, x, y)
	assertData(t, "WhereBasic", got, []float64{10, -2, 30, -4})
}

func TestWhereAllTrue(t *testing.T) {
	cond := FromSlice([]float64{1, 1, 1})
	x := FromSlice([]float64{10, 20, 30})
	y := FromSlice([]float64{-1, -2, -3})
	got := Where(cond, x, y)
	assertData(t, "WhereAllTrue", got, []float64{10, 20, 30})
}

func TestWhereAllFalse(t *testing.T) {
	cond := FromSlice([]float64{0, 0, 0})
	x := FromSlice([]float64{10, 20, 30})
	y := FromSlice([]float64{-1, -2, -3})
	got := Where(cond, x, y)
	assertData(t, "WhereAllFalse", got, []float64{-1, -2, -3})
}

func TestWhereBroadcastScalarXY(t *testing.T) {
	// condition is 1D, x and y are scalars (1-element arrays)
	cond := FromSlice([]float64{1, 0, 1})
	x := FromSlice([]float64{99})
	y := FromSlice([]float64{-1})
	got := Where(cond, x, y)
	assertData(t, "WhereBroadcast", got, []float64{99, -1, 99})
}

func TestWhere2D(t *testing.T) {
	cond := NewNDArray([]int{2, 2}, []float64{1, 0, 0, 1})
	x := NewNDArray([]int{2, 2}, []float64{10, 20, 30, 40})
	y := NewNDArray([]int{2, 2}, []float64{-1, -2, -3, -4})
	got := Where(cond, x, y)
	assertData(t, "Where2D", got, []float64{10, -2, -3, 40})
	assertShape(t, "Where2D", got, []int{2, 2})
}

func TestWhereBroadcastCondition(t *testing.T) {
	// cond shape [2,1], x/y shape [2,3] => broadcast cond to [2,3]
	cond := NewNDArray([]int{2, 1}, []float64{1, 0})
	x := NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	y := NewNDArray([]int{2, 3}, []float64{10, 20, 30, 40, 50, 60})
	got := Where(cond, x, y)
	// row 0: cond=1 => x; row 1: cond=0 => y
	assertData(t, "WhereBroadcastCond", got, []float64{1, 2, 3, 40, 50, 60})
	assertShape(t, "WhereBroadcastCond", got, []int{2, 3})
}

// ---------------------------------------------------------------------------
// Nonzero
// ---------------------------------------------------------------------------

func TestNonzero1D(t *testing.T) {
	a := FromSlice([]float64{0, 3, 0, 5, 0})
	got := Nonzero(a)
	if len(got) != 2 {
		t.Fatalf("Nonzero1D: expected 2 results, got %d", len(got))
	}
	assertCoord(t, "Nonzero1D[0]", got[0], []int{1})
	assertCoord(t, "Nonzero1D[1]", got[1], []int{3})
}

func TestNonzero2D(t *testing.T) {
	a := NewNDArray([]int{2, 3}, []float64{0, 1, 0, 2, 0, 3})
	got := Nonzero(a)
	if len(got) != 3 {
		t.Fatalf("Nonzero2D: expected 3 results, got %d", len(got))
	}
	assertCoord(t, "Nonzero2D[0]", got[0], []int{0, 1})
	assertCoord(t, "Nonzero2D[1]", got[1], []int{1, 0})
	assertCoord(t, "Nonzero2D[2]", got[2], []int{1, 2})
}

func TestNonzeroAllZero(t *testing.T) {
	a := FromSlice([]float64{0, 0, 0})
	got := Nonzero(a)
	if len(got) != 0 {
		t.Fatalf("NonzeroAllZero: expected 0 results, got %d", len(got))
	}
}

func TestNonzeroAllNonzero(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	got := Nonzero(a)
	if len(got) != 3 {
		t.Fatalf("NonzeroAllNonzero: expected 3 results, got %d", len(got))
	}
}

// ---------------------------------------------------------------------------
// SearchSorted
// ---------------------------------------------------------------------------

func TestSearchSortedBasic(t *testing.T) {
	sorted := FromSlice([]float64{1, 3, 5, 7, 9})
	values := FromSlice([]float64{0, 3, 4, 9, 10})
	got := SearchSorted(sorted, values)
	// 0->0, 3->1, 4->2, 9->4, 10->5
	assertData(t, "SearchSortedBasic", got, []float64{0, 1, 2, 4, 5})
}

func TestSearchSortedEmpty(t *testing.T) {
	sorted := NewNDArray([]int{0}, []float64{})
	values := FromSlice([]float64{1, 2})
	got := SearchSorted(sorted, values)
	assertData(t, "SearchSortedEmpty", got, []float64{0, 0})
}

func TestSearchSortedSingle(t *testing.T) {
	sorted := FromSlice([]float64{5})
	values := FromSlice([]float64{3, 5, 7})
	got := SearchSorted(sorted, values)
	assertData(t, "SearchSortedSingle", got, []float64{0, 0, 1})
}

func TestSearchSortedDuplicates(t *testing.T) {
	sorted := FromSlice([]float64{1, 2, 2, 3, 3, 3})
	values := FromSlice([]float64{2, 3})
	got := SearchSorted(sorted, values)
	// leftmost: 2 -> index 1, 3 -> index 3
	assertData(t, "SearchSortedDups", got, []float64{1, 3})
}

func TestSearchSortedPanicsNon1D(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for non-1D sorted array")
		}
	}()
	sorted := NewNDArray([]int{2, 2}, []float64{1, 2, 3, 4})
	SearchSorted(sorted, FromSlice([]float64{1}))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func assertData(t *testing.T, label string, got *NDArray, want []float64) {
	t.Helper()
	d := got.Data()
	if len(d) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(d), len(want))
	}
	for i := range d {
		if d[i] != want[i] {
			t.Fatalf("%s: index %d = %g, want %g (full: %v)", label, i, d[i], want[i], d)
		}
	}
}

func assertShape(t *testing.T, label string, got *NDArray, want []int) {
	t.Helper()
	s := got.Shape()
	if len(s) != len(want) {
		t.Fatalf("%s: shape ndim %d, want %d", label, len(s), len(want))
	}
	for i := range s {
		if s[i] != want[i] {
			t.Fatalf("%s: shape[%d] = %d, want %d", label, i, s[i], want[i])
		}
	}
}

func assertCoord(t *testing.T, label string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: coord length %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: coord[%d] = %d, want %d", label, i, got[i], want[i])
		}
	}
}
