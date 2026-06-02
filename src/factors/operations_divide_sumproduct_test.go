//go:build unit

package factors

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// FactorDivide
// ---------------------------------------------------------------------------

func TestFactorDivide_MatchingVariables(t *testing.T) {
	// f1(A,B) / f2(A,B) — same variables, element-wise division.
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{10, 20, 30, 40})
	f2, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{2, 4, 5, 8})

	result, err := FactorDivide(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	vars := result.Variables()
	if len(vars) != 2 || vars[0] != "A" || vars[1] != "B" {
		t.Fatalf("expected [A B], got %v", vars)
	}

	expected := []struct {
		a, b int
		want float64
	}{
		{0, 0, 5.0},
		{0, 1, 5.0},
		{1, 0, 6.0},
		{1, 1, 5.0},
	}
	for _, tc := range expected {
		got := result.GetValue(map[string]int{"A": tc.a, "B": tc.b})
		if !floatEq(got, tc.want) {
			t.Errorf("f(A=%d,B=%d) = %f, want %f", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestFactorDivide_SubsetVariables(t *testing.T) {
	// f1(A,B) / f2(B) — f2 has fewer variables.
	// f1 values: A=0,B=0 -> 6; A=0,B=1 -> 8; A=1,B=0 -> 12; A=1,B=1 -> 16
	// f2 values: B=0 -> 2; B=1 -> 4
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{6, 8, 12, 16})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{2, 4})

	result, err := FactorDivide(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	vars := result.Variables()
	if len(vars) != 2 || vars[0] != "A" || vars[1] != "B" {
		t.Fatalf("expected [A B], got %v", vars)
	}

	expected := []struct {
		a, b int
		want float64
	}{
		{0, 0, 3.0}, // 6/2
		{0, 1, 2.0}, // 8/4
		{1, 0, 6.0}, // 12/2
		{1, 1, 4.0}, // 16/4
	}
	for _, tc := range expected {
		got := result.GetValue(map[string]int{"A": tc.a, "B": tc.b})
		if !floatEq(got, tc.want) {
			t.Errorf("f(A=%d,B=%d) = %f, want %f", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestFactorDivide_SubsetVariables_ThreeVars(t *testing.T) {
	// f1(A,B,C) / f2(A,C) — f2 has a non-contiguous subset.
	// A=2, B=2, C=2 => 8 values
	f1, _ := NewDiscreteFactor([]string{"A", "B", "C"}, []int{2, 2, 2},
		[]float64{2, 4, 6, 8, 10, 12, 14, 16})
	// f2(A,C): A=2, C=2
	f2, _ := NewDiscreteFactor([]string{"A", "C"}, []int{2, 2},
		[]float64{1, 2, 5, 4})

	result, err := FactorDivide(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	// Verify: for each (a,b,c), result = f1(a,b,c) / f2(a,c)
	f1Vals := []float64{2, 4, 6, 8, 10, 12, 14, 16}
	f2Vals := map[[2]int]float64{
		{0, 0}: 1, {0, 1}: 2, {1, 0}: 5, {1, 1}: 4,
	}
	idx := 0
	for a := 0; a < 2; a++ {
		for b := 0; b < 2; b++ {
			for c := 0; c < 2; c++ {
				want := f1Vals[idx] / f2Vals[[2]int{a, c}]
				got := result.GetValue(map[string]int{"A": a, "B": b, "C": c})
				if !floatEq(got, want) {
					t.Errorf("f(A=%d,B=%d,C=%d) = %f, want %f", a, b, c, got, want)
				}
				idx++
			}
		}
	}
}

func TestFactorDivide_DivisionByZero(t *testing.T) {
	// Division by zero should produce 0.
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{6, 9, 12})
	f2, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{3, 0, 4})

	result, err := FactorDivide(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{2.0, 0.0, 3.0}
	for i, want := range expected {
		got := result.GetValue(map[string]int{"A": i})
		if !floatEq(got, want) {
			t.Errorf("f(A=%d) = %f, want %f", i, got, want)
		}
	}
}

func TestFactorDivide_NotSubsetError(t *testing.T) {
	// f2 has a variable not in f1 — should error.
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 2})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{3, 4})

	_, err := FactorDivide(f1, f2)
	if err == nil {
		t.Error("expected error when f2 has variable not in f1")
	}
}

func TestFactorDivide_CardinalityMismatch(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 2})
	f2, _ := NewDiscreteFactor([]string{"A"}, []int{3}, []float64{1, 2, 3})

	_, err := FactorDivide(f1, f2)
	if err == nil {
		t.Error("expected error for mismatched cardinality")
	}
}

func TestFactorDivide_NilFactor(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{1, 2})

	_, err := FactorDivide(nil, f1)
	if err == nil {
		t.Error("expected error for nil f1")
	}

	_, err = FactorDivide(f1, nil)
	if err == nil {
		t.Error("expected error for nil f2")
	}
}

func TestFactorDivide_ProductThenDivide(t *testing.T) {
	// f1 * f2 / f2 should give back f1 (up to floating point).
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 3},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	f2, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 3},
		[]float64{1, 2, 3, 4, 5, 6})

	product, err := FactorProduct(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	result, err := FactorDivide(product, f2)
	if err != nil {
		t.Fatal(err)
	}

	for a := 0; a < 2; a++ {
		for b := 0; b < 3; b++ {
			got := result.GetValue(map[string]int{"A": a, "B": b})
			want := f1.GetValue(map[string]int{"A": a, "B": b})
			if !floatEq(got, want) {
				t.Errorf("f(A=%d,B=%d) = %f, want %f", a, b, got, want)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// FactorSumProduct
// ---------------------------------------------------------------------------

func TestFactorSumProduct_SimpleChain(t *testing.T) {
	// Chain: A -> B -> C
	// P(A): [0.4, 0.6]
	// P(B|A): A=2, B=2
	// P(C|B): B=2, C=2
	// Eliminate B, A to get P(C).
	pA, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBgA, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
		[]float64{0.2, 0.8, 0.5, 0.5})
	pCgB, _ := NewDiscreteFactor([]string{"B", "C"}, []int{2, 2},
		[]float64{0.3, 0.7, 0.9, 0.1})

	result, err := FactorSumProduct(
		[]*DiscreteFactor{pA, pBgA, pCgB},
		[]string{"A", "B"},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Manually compute P(C) by brute force.
	// P(C=c) = sum_a,b P(A=a) * P(B=b|A=a) * P(C=c|B=b)
	for c := 0; c < 2; c++ {
		want := 0.0
		for a := 0; a < 2; a++ {
			for b := 0; b < 2; b++ {
				want += pA.GetValue(map[string]int{"A": a}) *
					pBgA.GetValue(map[string]int{"A": a, "B": b}) *
					pCgB.GetValue(map[string]int{"B": b, "C": c})
			}
		}
		got := result.GetValue(map[string]int{"C": c})
		if !floatEq(got, want) {
			t.Errorf("P(C=%d) = %f, want %f", c, got, want)
		}
	}
}

func TestFactorSumProduct_MatchesNaiveElimination(t *testing.T) {
	// Test that FactorSumProduct produces the same result as manual
	// product-then-marginalize (i.e., VariableElimination by hand).
	// P(A) = [0.3, 0.7]
	// P(B|A) over (A,B): [0.6, 0.4, 0.1, 0.9]
	pA, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	pBgA, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
		[]float64{0.6, 0.4, 0.1, 0.9})

	// Naive: product all, then marginalize A.
	joint, err := FactorProduct(pA, pBgA)
	if err != nil {
		t.Fatal(err)
	}
	naiveResult, err := joint.Marginalize([]string{"A"})
	if err != nil {
		t.Fatal(err)
	}

	// Sum-product elimination of A.
	spResult, err := FactorSumProduct(
		[]*DiscreteFactor{pA, pBgA},
		[]string{"A"},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Compare.
	for b := 0; b < 2; b++ {
		got := spResult.GetValue(map[string]int{"B": b})
		want := naiveResult.GetValue(map[string]int{"B": b})
		if !floatEq(got, want) {
			t.Errorf("P(B=%d): sum-product=%f, naive=%f", b, got, want)
		}
	}
}

func TestFactorSumProduct_EliminateAll(t *testing.T) {
	// Eliminating all variables but one from a joint should yield a marginal.
	// P(A,B,C) as a single factor.
	// A=2, B=2, C=2
	vals := []float64{0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.05, 0.20}
	f, _ := NewDiscreteFactor([]string{"A", "B", "C"}, []int{2, 2, 2}, vals)

	result, err := FactorSumProduct(
		[]*DiscreteFactor{f},
		[]string{"B", "C"},
	)
	if err != nil {
		t.Fatal(err)
	}

	// P(A=0) = sum over B,C of f(0,B,C) = 0.05+0.10+0.15+0.20 = 0.50
	// P(A=1) = 0.10+0.15+0.05+0.20 = 0.50
	want0 := 0.05 + 0.10 + 0.15 + 0.20
	want1 := 0.10 + 0.15 + 0.05 + 0.20
	got0 := result.GetValue(map[string]int{"A": 0})
	got1 := result.GetValue(map[string]int{"A": 1})
	if !floatEq(got0, want0) {
		t.Errorf("P(A=0) = %f, want %f", got0, want0)
	}
	if !floatEq(got1, want1) {
		t.Errorf("P(A=1) = %f, want %f", got1, want1)
	}
}

func TestFactorSumProduct_OrderIndependence(t *testing.T) {
	// Different elimination orders should give the same final answer.
	pA, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBgA, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
		[]float64{0.2, 0.8, 0.5, 0.5})
	pCgB, _ := NewDiscreteFactor([]string{"B", "C"}, []int{2, 2},
		[]float64{0.3, 0.7, 0.9, 0.1})

	factors := []*DiscreteFactor{pA, pBgA, pCgB}

	r1, err := FactorSumProduct(factors, []string{"A", "B"})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := FactorSumProduct(factors, []string{"B", "A"})
	if err != nil {
		t.Fatal(err)
	}

	for c := 0; c < 2; c++ {
		v1 := r1.GetValue(map[string]int{"C": c})
		v2 := r2.GetValue(map[string]int{"C": c})
		if !floatEq(v1, v2) {
			t.Errorf("C=%d: order1=%f, order2=%f", c, v1, v2)
		}
	}
}

func TestFactorSumProduct_SkipsAbsentVariable(t *testing.T) {
	// Elimination order includes a variable not in any factor — should be harmless.
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{3, 7})

	result, err := FactorSumProduct([]*DiscreteFactor{f}, []string{"Z"})
	if err != nil {
		t.Fatal(err)
	}

	if !floatEq(result.GetValue(map[string]int{"A": 0}), 3) {
		t.Error("expected unchanged factor")
	}
}

func TestFactorSumProduct_Empty(t *testing.T) {
	_, err := FactorSumProduct(nil, []string{"A"})
	if err == nil {
		t.Error("expected error for empty factors")
	}
}

func TestFactorSumProduct_LargerNetwork(t *testing.T) {
	// Diamond: A -> B, A -> C, B -> D, C -> D
	// Compute P(D) by eliminating A, B, C.
	pA, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	pBgA, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
		[]float64{0.7, 0.3, 0.2, 0.8})
	pCgA, _ := NewDiscreteFactor([]string{"A", "C"}, []int{2, 2},
		[]float64{0.9, 0.1, 0.4, 0.6})
	pDgBC, _ := NewDiscreteFactor([]string{"B", "C", "D"}, []int{2, 2, 2},
		[]float64{0.95, 0.05, 0.3, 0.7, 0.8, 0.2, 0.1, 0.9})

	factors := []*DiscreteFactor{pA, pBgA, pCgA, pDgBC}
	result, err := FactorSumProduct(factors, []string{"A", "B", "C"})
	if err != nil {
		t.Fatal(err)
	}

	// Brute-force P(D=d).
	for d := 0; d < 2; d++ {
		want := 0.0
		for a := 0; a < 2; a++ {
			for b := 0; b < 2; b++ {
				for c := 0; c < 2; c++ {
					want += pA.GetValue(map[string]int{"A": a}) *
						pBgA.GetValue(map[string]int{"A": a, "B": b}) *
						pCgA.GetValue(map[string]int{"A": a, "C": c}) *
						pDgBC.GetValue(map[string]int{"B": b, "C": c, "D": d})
				}
			}
		}
		got := result.GetValue(map[string]int{"D": d})
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("P(D=%d) = %f, want %f", d, got, want)
		}
	}
}

func TestFactorSumProduct_DoesNotMutateInput(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{5, 6})

	// Save original values.
	orig1 := f1.Values().Data()
	orig2 := f2.Values().Data()
	save1 := make([]float64, len(orig1))
	save2 := make([]float64, len(orig2))
	copy(save1, orig1)
	copy(save2, orig2)

	_, err := FactorSumProduct([]*DiscreteFactor{f1, f2}, []string{"A"})
	if err != nil {
		t.Fatal(err)
	}

	// Verify inputs unchanged.
	for i, v := range f1.Values().Data() {
		if v != save1[i] {
			t.Errorf("f1 was mutated at index %d", i)
		}
	}
	for i, v := range f2.Values().Data() {
		if v != save2[i] {
			t.Errorf("f2 was mutated at index %d", i)
		}
	}
}
