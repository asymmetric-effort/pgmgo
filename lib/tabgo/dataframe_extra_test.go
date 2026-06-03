//go:build unit

package tabgo

import (
	"strings"
	"testing"
)

func TestDtypes(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b", "c"},
		[][]any{
			{1.0, "hello", true},
			{2.0, "world", false},
		},
	)
	dtypes := df.Dtypes()
	if dtypes["a"] != "float64" {
		t.Errorf("expected float64 for 'a', got %q", dtypes["a"])
	}
	if dtypes["b"] != "string" {
		t.Errorf("expected string for 'b', got %q", dtypes["b"])
	}
	if dtypes["c"] != "bool" {
		t.Errorf("expected bool for 'c', got %q", dtypes["c"])
	}
}

func TestInfo(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1.0, nil},
			{2.0, "x"},
		},
	)
	info := df.Info()
	if !strings.Contains(info, "2 rows") {
		t.Errorf("expected '2 rows' in info, got:\n%s", info)
	}
	if !strings.Contains(info, "2 columns") {
		t.Errorf("expected '2 columns' in info")
	}
	if !strings.Contains(info, "a") || !strings.Contains(info, "b") {
		t.Errorf("expected column names in info")
	}
}

func TestMemory(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1.0, "hello"},
			{2.0, "world"},
		},
	)
	mem := df.Memory()
	if mem <= 0 {
		t.Errorf("expected positive memory usage, got %d", mem)
	}
}

func TestNunique(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1, "x"},
			{2, "x"},
			{1, "y"},
		},
	)
	nu := df.Nunique()
	if nu["a"] != 2 {
		t.Errorf("expected 2 unique for 'a', got %d", nu["a"])
	}
	if nu["b"] != 2 {
		t.Errorf("expected 2 unique for 'b', got %d", nu["b"])
	}
}

func TestDuplicated(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1, "x"},
			{2, "y"},
			{1, "x"},
		},
	)
	dups := df.Duplicated()
	if dups[0] {
		t.Error("first row should not be a duplicate")
	}
	if dups[1] {
		t.Error("second row should not be a duplicate")
	}
	if !dups[2] {
		t.Error("third row should be a duplicate")
	}
}

func TestDropDuplicates(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1, "x"},
			{2, "y"},
			{1, "x"},
			{3, "z"},
			{2, "y"},
		},
	)
	result := df.DropDuplicates()
	if result.Len() != 3 {
		t.Errorf("expected 3 rows, got %d", result.Len())
	}
}

func TestReplace(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{1, "x"},
			{2, "y"},
			{1, "z"},
		},
	)
	result := df.Replace(1, 99)
	vals := result.Column("a").Values()
	if vals[0] != 99 {
		t.Errorf("expected 99, got %v", vals[0])
	}
	if vals[1] != 2 {
		t.Errorf("expected 2, got %v", vals[1])
	}
	if vals[2] != 99 {
		t.Errorf("expected 99, got %v", vals[2])
	}
}

func TestClip(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{-5.0, "x"},
			{3.0, "y"},
			{10.0, "z"},
		},
	)
	result := df.Clip(0, 5)
	vals := result.Column("a").Values()
	if vals[0] != 0.0 {
		t.Errorf("expected 0, got %v", vals[0])
	}
	if vals[1] != 3.0 {
		t.Errorf("expected 3, got %v", vals[1])
	}
	if vals[2] != 5.0 {
		t.Errorf("expected 5, got %v", vals[2])
	}
	// String column should be unchanged.
	bVals := result.Column("b").Values()
	if bVals[0] != "x" {
		t.Errorf("expected 'x', got %v", bVals[0])
	}
}

func TestAbs(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{-5.0, "x"},
			{3.0, "y"},
			{-1.0, nil},
		},
	)
	result := df.Abs()
	vals := result.Column("a").Values()
	if vals[0] != 5.0 {
		t.Errorf("expected 5, got %v", vals[0])
	}
	if vals[1] != 3.0 {
		t.Errorf("expected 3, got %v", vals[1])
	}
	if vals[2] != 1.0 {
		t.Errorf("expected 1, got %v", vals[2])
	}
}

func TestRound(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a", "b"},
		[][]any{
			{3.14159, "x"},
			{2.71828, nil},
		},
	)
	result := df.Round(2)
	vals := result.Column("a").Values()
	if vals[0] != 3.14 {
		t.Errorf("expected 3.14, got %v", vals[0])
	}
	if vals[1] != 2.72 {
		t.Errorf("expected 2.72, got %v", vals[1])
	}
}

func TestPipe(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1.0}, {2.0}, {3.0}},
	)
	result := df.Pipe(func(d *DataFrame) *DataFrame {
		return d.Head(2)
	})
	if result.Len() != 2 {
		t.Errorf("expected 2 rows, got %d", result.Len())
	}
}

func TestDataFrameShift_Positive(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1.0}, {2.0}, {3.0}},
	)
	result := df.Shift(1)
	vals := result.Column("a").Values()
	if vals[0] != nil {
		t.Errorf("expected nil, got %v", vals[0])
	}
	if vals[1] != 1.0 {
		t.Errorf("expected 1, got %v", vals[1])
	}
	if vals[2] != 2.0 {
		t.Errorf("expected 2, got %v", vals[2])
	}
}

func TestDataFrameShift_Negative(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1.0}, {2.0}, {3.0}},
	)
	result := df.Shift(-1)
	vals := result.Column("a").Values()
	if vals[0] != 2.0 {
		t.Errorf("expected 2, got %v", vals[0])
	}
	if vals[1] != 3.0 {
		t.Errorf("expected 3, got %v", vals[1])
	}
	if vals[2] != nil {
		t.Errorf("expected nil, got %v", vals[2])
	}
}

func TestDataFrameShift_Zero(t *testing.T) {
	df := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1.0}, {2.0}},
	)
	result := df.Shift(0)
	vals := result.Column("a").Values()
	if vals[0] != 1.0 || vals[1] != 2.0 {
		t.Errorf("shift(0) should not change values")
	}
}
