//go:build unit

package factors

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Maximize
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Maximize(t *testing.T) {
	// Factor over A(2), B(3) with values:
	// A=0,B=0: 1  A=0,B=1: 5  A=0,B=2: 3
	// A=1,B=0: 4  A=1,B=1: 2  A=1,B=2: 6
	f, err := NewDiscreteFactor(
		[]string{"A", "B"},
		[]int{2, 3},
		[]float64{1, 5, 3, 4, 2, 6},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Maximize out B -> factor over A.
	result, err := f.Maximize([]string{"B"})
	if err != nil {
		t.Fatal(err)
	}
	if vars := result.Variables(); len(vars) != 1 || vars[0] != "A" {
		t.Errorf("Variables() = %v", vars)
	}
	// max over B for A=0: max(1,5,3) = 5
	// max over B for A=1: max(4,2,6) = 6
	data := result.Values().Data()
	if !floatEq(data[0], 5) {
		t.Errorf("A=0: got %f, want 5", data[0])
	}
	if !floatEq(data[1], 6) {
		t.Errorf("A=1: got %f, want 6", data[1])
	}
}

func TestDiscreteFactor_Maximize_Empty(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{1, 2, 3})
	result, err := f.Maximize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Values().Data()[1] != 2 {
		t.Error("empty maximize should return copy")
	}
}

func TestDiscreteFactor_Maximize_Errors(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})

	_, err := f.Maximize([]string{"C"})
	if err == nil {
		t.Error("expected error for unknown variable")
	}

	_, err = f.Maximize([]string{"A", "B"})
	if err == nil {
		t.Error("expected error for maximizing all variables")
	}
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Sample(t *testing.T) {
	// Factor heavily weighted toward A=1.
	f, err := NewDiscreteFactor(
		[]string{"A"},
		[]int{3},
		[]float64{0, 100, 0},
	)
	if err != nil {
		t.Fatal(err)
	}

	samples, err := f.Sample(100, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 100 {
		t.Fatalf("expected 100 samples, got %d", len(samples))
	}
	// All samples should be A=1.
	for i, s := range samples {
		if s["A"] != 1 {
			t.Errorf("sample %d: A=%d, want 1", i, s["A"])
		}
	}
}

func TestDiscreteFactor_Sample_MultiVariable(t *testing.T) {
	f, err := NewDiscreteFactor(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{1, 2, 3, 4},
	)
	if err != nil {
		t.Fatal(err)
	}

	samples, err := f.Sample(1000, 123)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1000 {
		t.Fatalf("expected 1000 samples, got %d", len(samples))
	}
	// Every sample should have both A and B keys with valid values.
	for i, s := range samples {
		if s["A"] < 0 || s["A"] > 1 || s["B"] < 0 || s["B"] > 1 {
			t.Errorf("sample %d: invalid assignment %v", i, s)
		}
	}
}

func TestDiscreteFactor_Sample_Errors(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 1})

	_, err := f.Sample(0, 1)
	if err == nil {
		t.Error("expected error for n=0")
	}

	fZero, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0, 0})
	_, err = fZero.Sample(1, 1)
	if err == nil {
		t.Error("expected error for all-zero factor")
	}
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Assignment(t *testing.T) {
	f, err := NewDiscreteFactor(
		[]string{"A", "B"},
		[]int{2, 3},
		[]float64{1, 2, 3, 4, 5, 6},
	)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		index int
		wantA int
		wantB int
	}{
		{0, 0, 0},
		{1, 0, 1},
		{2, 0, 2},
		{3, 1, 0},
		{4, 1, 1},
		{5, 1, 2},
	}
	for _, tt := range tests {
		a, err := f.Assignment(tt.index)
		if err != nil {
			t.Errorf("index %d: %v", tt.index, err)
			continue
		}
		if a["A"] != tt.wantA || a["B"] != tt.wantB {
			t.Errorf("index %d: got A=%d,B=%d; want A=%d,B=%d",
				tt.index, a["A"], a["B"], tt.wantA, tt.wantB)
		}
	}
}

func TestDiscreteFactor_Assignment_Errors(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{1, 2, 3})

	_, err := f.Assignment(-1)
	if err == nil {
		t.Error("expected error for negative index")
	}
	_, err = f.Assignment(3)
	if err == nil {
		t.Error("expected error for index out of range")
	}
}

// ---------------------------------------------------------------------------
// IdentityFactor
// ---------------------------------------------------------------------------

func TestIdentityFactor(t *testing.T) {
	f, err := IdentityFactor([]string{"A", "B"}, []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	data := f.Values().Data()
	for i, v := range data {
		if v != 1.0 {
			t.Errorf("data[%d] = %f, want 1.0", i, v)
		}
	}
	if len(data) != 6 {
		t.Errorf("expected 6 values, got %d", len(data))
	}
}

func TestIdentityFactor_Errors(t *testing.T) {
	_, err := IdentityFactor([]string{"A"}, []int{2, 3})
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
	_, err = IdentityFactor([]string{"A"}, []int{0})
	if err == nil {
		t.Error("expected error for zero cardinality")
	}
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Sum(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	f2, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{10, 20, 30, 40})

	result, err := f1.Sum(f2)
	if err != nil {
		t.Fatal(err)
	}
	data := result.Values().Data()
	expected := []float64{11, 22, 33, 44}
	for i, v := range data {
		if !floatEq(v, expected[i]) {
			t.Errorf("data[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestDiscreteFactor_Sum_Errors(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 2})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{1, 2})
	f3, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{1, 2, 3})

	_, err := f1.Sum(nil)
	if err == nil {
		t.Error("expected error for nil factor")
	}

	_, err = f1.Sum(f2)
	if err == nil {
		t.Error("expected error for variable name mismatch")
	}

	_, err = f1.Sum(f3)
	if err == nil {
		t.Error("expected error for cardinality mismatch")
	}
}

// ---------------------------------------------------------------------------
// IsValidCPD
// ---------------------------------------------------------------------------

func TestDiscreteFactor_IsValidCPD_True(t *testing.T) {
	// P(X | Y): X has 2 states, Y has 2 states.
	f, _ := NewDiscreteFactor(
		[]string{"X", "Y"},
		[]int{2, 2},
		[]float64{0.4, 0.9, 0.6, 0.1},
	)
	if !f.IsValidCPD() {
		t.Error("expected IsValidCPD() to be true")
	}
}

func TestDiscreteFactor_IsValidCPD_False(t *testing.T) {
	f, _ := NewDiscreteFactor(
		[]string{"X", "Y"},
		[]int{2, 2},
		[]float64{0.5, 0.5, 0.5, 0.5},
	)
	// Columns: col0 = 0.5+0.5=1.0, col1 = 0.5+0.5=1.0 -> valid
	if !f.IsValidCPD() {
		t.Error("expected IsValidCPD() to be true for equal probs")
	}

	// Now an invalid one.
	f2, _ := NewDiscreteFactor(
		[]string{"X", "Y"},
		[]int{2, 2},
		[]float64{0.3, 0.9, 0.6, 0.1},
	)
	// col0 = 0.3+0.6 = 0.9 != 1 -> invalid
	if f2.IsValidCPD() {
		t.Error("expected IsValidCPD() to be false")
	}
}

func TestDiscreteFactor_IsValidCPD_NoParents(t *testing.T) {
	// Single variable, just a distribution.
	f, _ := NewDiscreteFactor([]string{"X"}, []int{3}, []float64{0.2, 0.3, 0.5})
	if !f.IsValidCPD() {
		t.Error("expected IsValidCPD() to be true for marginal distribution")
	}
}

func TestDiscreteFactor_IsValidCPD_Empty(t *testing.T) {
	// Empty factor — should return false since there are no variables.
	f, _ := NewDiscreteFactor(nil, nil, []float64{1.0})
	if f.IsValidCPD() {
		t.Error("expected IsValidCPD() to be false for empty factor")
	}
}

// ---------------------------------------------------------------------------
// Maximize with multiple variables
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Maximize_MultiVar(t *testing.T) {
	// Factor over A(2), B(2), C(2).
	f, err := NewDiscreteFactor(
		[]string{"A", "B", "C"},
		[]int{2, 2, 2},
		[]float64{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Maximize out B and C -> factor over A.
	result, err := f.Maximize([]string{"B", "C"})
	if err != nil {
		t.Fatal(err)
	}
	data := result.Values().Data()
	// A=0: values are 1,2,3,4 -> max=4
	// A=1: values are 5,6,7,8 -> max=8
	if !floatEq(data[0], 4) {
		t.Errorf("A=0: got %f, want 4", data[0])
	}
	if !floatEq(data[1], 8) {
		t.Errorf("A=1: got %f, want 8", data[1])
	}
}

// ---------------------------------------------------------------------------
// Sample distribution test
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Sample_Distribution(t *testing.T) {
	// Verify sampling roughly follows the distribution.
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 3})
	samples, err := f.Sample(10000, 77)
	if err != nil {
		t.Fatal(err)
	}

	counts := make([]int, 2)
	for _, s := range samples {
		counts[s["A"]]++
	}

	// Expected: ~25% for A=0, ~75% for A=1.
	ratio := float64(counts[0]) / float64(len(samples))
	if math.Abs(ratio-0.25) > 0.05 {
		t.Errorf("A=0 ratio = %f, expected ~0.25", ratio)
	}
}

// ---------------------------------------------------------------------------
// IdentityFactor product identity
// ---------------------------------------------------------------------------

func TestIdentityFactor_ProductIdentity(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	identity, _ := IdentityFactor([]string{"A", "B"}, []int{2, 3})

	product, err := FactorProduct(f, identity)
	if err != nil {
		t.Fatal(err)
	}

	origData := f.Values().Data()
	prodData := product.Values().Data()
	for i := range origData {
		if !floatEq(origData[i], prodData[i]) {
			t.Errorf("index %d: %f != %f", i, origData[i], prodData[i])
		}
	}
}
