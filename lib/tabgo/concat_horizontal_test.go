//go:build unit

package tabgo

import (
	"testing"
)

func TestConcatHorizontal_Basic(t *testing.T) {
	df1 := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1}, {2}},
	)
	df2 := NewDataFrameFromRows(
		[]string{"b"},
		[][]any{{"x"}, {"y"}},
	)
	result, err := ConcatHorizontal([]*DataFrame{df1, df2})
	if err != nil {
		t.Fatalf("ConcatHorizontal error: %v", err)
	}
	if result.Len() != 2 {
		t.Errorf("expected 2 rows, got %d", result.Len())
	}
	cols := result.Columns()
	if len(cols) != 2 {
		t.Errorf("expected 2 columns, got %d", len(cols))
	}
}

func TestConcatHorizontal_NameConflict(t *testing.T) {
	df1 := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1}, {2}},
	)
	df2 := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{3}, {4}},
	)
	result, err := ConcatHorizontal([]*DataFrame{df1, df2})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	cols := result.Columns()
	if len(cols) != 2 {
		t.Errorf("expected 2 columns, got %d", len(cols))
	}
	// Second column should have a suffix.
	if cols[0] == cols[1] {
		t.Error("column names should differ")
	}
}

func TestConcatHorizontal_MismatchedRows(t *testing.T) {
	df1 := NewDataFrameFromRows(
		[]string{"a"},
		[][]any{{1}, {2}},
	)
	df2 := NewDataFrameFromRows(
		[]string{"b"},
		[][]any{{1}},
	)
	_, err := ConcatHorizontal([]*DataFrame{df1, df2})
	if err == nil {
		t.Error("expected error for mismatched rows")
	}
}

func TestConcatHorizontal_Empty(t *testing.T) {
	result, err := ConcatHorizontal(nil)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if result.Len() != 0 {
		t.Errorf("expected 0 rows, got %d", result.Len())
	}
}

func TestConcatHorizontal_MultipleFrames(t *testing.T) {
	df1 := NewDataFrameFromRows([]string{"a"}, [][]any{{1}, {2}})
	df2 := NewDataFrameFromRows([]string{"b"}, [][]any{{3}, {4}})
	df3 := NewDataFrameFromRows([]string{"c"}, [][]any{{5}, {6}})

	result, err := ConcatHorizontal([]*DataFrame{df1, df2, df3})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(result.Columns()) != 3 {
		t.Errorf("expected 3 columns, got %d", len(result.Columns()))
	}
}
