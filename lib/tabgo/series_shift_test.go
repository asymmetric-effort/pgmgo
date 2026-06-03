//go:build unit

package tabgo

import (
	"testing"
)

func TestSeriesShift_Positive(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0})
	result := s.Shift(1)
	vals := result.Values()
	if vals[0] != nil {
		t.Errorf("expected nil, got %v", vals[0])
	}
	if vals[1] != 1.0 {
		t.Errorf("expected 1, got %v", vals[1])
	}
	if vals[2] != 2.0 {
		t.Errorf("expected 2, got %v", vals[2])
	}
	if vals[3] != 3.0 {
		t.Errorf("expected 3, got %v", vals[3])
	}
}

func TestSeriesShift_Negative(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0})
	result := s.Shift(-1)
	vals := result.Values()
	if vals[0] != 2.0 {
		t.Errorf("expected 2, got %v", vals[0])
	}
	if vals[1] != 3.0 {
		t.Errorf("expected 3, got %v", vals[1])
	}
	if vals[2] != 4.0 {
		t.Errorf("expected 4, got %v", vals[2])
	}
	if vals[3] != nil {
		t.Errorf("expected nil, got %v", vals[3])
	}
}

func TestSeriesShift_Zero(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0})
	result := s.Shift(0)
	vals := result.Values()
	if vals[0] != 1.0 || vals[1] != 2.0 || vals[2] != 3.0 {
		t.Error("shift(0) should not change values")
	}
}

func TestSeriesShift_LargePeriod(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0})
	result := s.Shift(5)
	vals := result.Values()
	if vals[0] != nil || vals[1] != nil {
		t.Error("all values should be nil for large shift")
	}
}

func TestSeriesShift_Empty(t *testing.T) {
	s := NewSeries("x", []any{})
	result := s.Shift(1)
	if result.Len() != 0 {
		t.Errorf("expected empty, got %d elements", result.Len())
	}
}
