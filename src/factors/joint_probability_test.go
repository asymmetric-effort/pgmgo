//go:build unit

package factors

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

func jpdFloatEq(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// ---------------------------------------------------------------------------
// NewJointProbabilityDistribution
// ---------------------------------------------------------------------------

func TestNewJPD_Valid(t *testing.T) {
	// 2 binary variables: P(A,B)
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if jpd == nil {
		t.Fatal("expected non-nil JPD")
	}
}

func TestNewJPD_InvalidSum(t *testing.T) {
	_, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.5}, // sums to 1.1
	)
	if err == nil {
		t.Fatal("expected error for values not summing to 1")
	}
}

func TestNewJPD_NegativeValues(t *testing.T) {
	_, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{-0.1, 0.3, 0.4, 0.4},
	)
	if err == nil {
		t.Fatal("expected error for negative values")
	}
}

func TestNewJPD_BadCardinality(t *testing.T) {
	_, err := NewJointProbabilityDistribution(
		[]string{"A"},
		[]int{2, 2},
		[]float64{0.5, 0.5},
	)
	if err == nil {
		t.Fatal("expected error for mismatched variables/cardinality")
	}
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

func TestValidate_GoodDistribution(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"X"},
		[]int{3},
		[]float64{0.2, 0.3, 0.5},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := jpd.Validate(); err != nil {
		t.Fatalf("validation should pass: %v", err)
	}
}

// ---------------------------------------------------------------------------
// MarginalDistribution — 2 variables
// ---------------------------------------------------------------------------

func TestMarginalDistribution_2Var(t *testing.T) {
	// P(A,B) with A in {0,1}, B in {0,1}
	// P(A=0,B=0)=0.1, P(A=0,B=1)=0.2, P(A=1,B=0)=0.3, P(A=1,B=1)=0.4
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Marginal P(A) = [0.1+0.2, 0.3+0.4] = [0.3, 0.7]
	mA, err := jpd.MarginalDistribution([]string{"A"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pA0 := mA.GetValue(map[string]int{"A": 0})
	pA1 := mA.GetValue(map[string]int{"A": 1})
	if !jpdFloatEq(pA0, 0.3) {
		t.Errorf("P(A=0) = %f, want 0.3", pA0)
	}
	if !jpdFloatEq(pA1, 0.7) {
		t.Errorf("P(A=1) = %f, want 0.7", pA1)
	}

	// Marginal P(B) = [0.1+0.3, 0.2+0.4] = [0.4, 0.6]
	mB, err := jpd.MarginalDistribution([]string{"B"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pB0 := mB.GetValue(map[string]int{"B": 0})
	pB1 := mB.GetValue(map[string]int{"B": 1})
	if !jpdFloatEq(pB0, 0.4) {
		t.Errorf("P(B=0) = %f, want 0.4", pB0)
	}
	if !jpdFloatEq(pB1, 0.6) {
		t.Errorf("P(B=1) = %f, want 0.6", pB1)
	}
}

func TestMarginalDistribution_NoVars(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A"},
		[]int{2},
		[]float64{0.4, 0.6},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = jpd.MarginalDistribution([]string{})
	if err == nil {
		t.Fatal("expected error when no variables specified")
	}
}

func TestMarginalDistribution_UnknownVar(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A"},
		[]int{2},
		[]float64{0.4, 0.6},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = jpd.MarginalDistribution([]string{"Z"})
	if err == nil {
		t.Fatal("expected error for unknown variable")
	}
}

// ---------------------------------------------------------------------------
// MarginalDistribution — 3 variables
// ---------------------------------------------------------------------------

func TestMarginalDistribution_3Var(t *testing.T) {
	// P(A,B,C) with A,B,C binary.
	// Row-major: (A=0,B=0,C=0), (A=0,B=0,C=1), (A=0,B=1,C=0), (A=0,B=1,C=1),
	//            (A=1,B=0,C=0), (A=1,B=0,C=1), (A=1,B=1,C=0), (A=1,B=1,C=1)
	vals := []float64{0.05, 0.10, 0.05, 0.10, 0.15, 0.10, 0.20, 0.25}
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B", "C"},
		[]int{2, 2, 2},
		vals,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Marginal P(A,B) - marginalize out C.
	mAB, err := jpd.MarginalDistribution([]string{"A", "B"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// P(A=0,B=0) = 0.05+0.10 = 0.15
	// P(A=0,B=1) = 0.05+0.10 = 0.15
	// P(A=1,B=0) = 0.15+0.10 = 0.25
	// P(A=1,B=1) = 0.20+0.25 = 0.45
	expected := map[[2]int]float64{
		{0, 0}: 0.15, {0, 1}: 0.15,
		{1, 0}: 0.25, {1, 1}: 0.45,
	}
	for k, want := range expected {
		got := mAB.GetValue(map[string]int{"A": k[0], "B": k[1]})
		if !jpdFloatEq(got, want) {
			t.Errorf("P(A=%d,B=%d) = %f, want %f", k[0], k[1], got, want)
		}
	}

	// Marginal P(C)
	mC, err := jpd.MarginalDistribution([]string{"C"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// P(C=0) = 0.05+0.05+0.15+0.20 = 0.45
	// P(C=1) = 0.10+0.10+0.10+0.25 = 0.55
	pC0 := mC.GetValue(map[string]int{"C": 0})
	pC1 := mC.GetValue(map[string]int{"C": 1})
	if !jpdFloatEq(pC0, 0.45) {
		t.Errorf("P(C=0) = %f, want 0.45", pC0)
	}
	if !jpdFloatEq(pC1, 0.55) {
		t.Errorf("P(C=1) = %f, want 0.55", pC1)
	}
}

// ---------------------------------------------------------------------------
// ConditionalDistribution
// ---------------------------------------------------------------------------

func TestConditionalDistribution_2Var(t *testing.T) {
	// P(A,B)
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// P(A | B=0): P(A=0,B=0)=0.1, P(A=1,B=0)=0.3 => P(B=0)=0.4
	// => P(A=0|B=0)=0.25, P(A=1|B=0)=0.75
	cond, err := jpd.ConditionalDistribution([]string{"A"}, map[string]int{"B": 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pA0gB0 := cond.GetValue(map[string]int{"A": 0})
	pA1gB0 := cond.GetValue(map[string]int{"A": 1})
	if !jpdFloatEq(pA0gB0, 0.25) {
		t.Errorf("P(A=0|B=0) = %f, want 0.25", pA0gB0)
	}
	if !jpdFloatEq(pA1gB0, 0.75) {
		t.Errorf("P(A=1|B=0) = %f, want 0.75", pA1gB0)
	}

	// P(B | A=1): P(A=1,B=0)=0.3, P(A=1,B=1)=0.4 => P(A=1)=0.7
	// => P(B=0|A=1)=3/7, P(B=1|A=1)=4/7
	cond2, err := jpd.ConditionalDistribution([]string{"B"}, map[string]int{"A": 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pB0gA1 := cond2.GetValue(map[string]int{"B": 0})
	pB1gA1 := cond2.GetValue(map[string]int{"B": 1})
	if !jpdFloatEq(pB0gA1, 3.0/7.0) {
		t.Errorf("P(B=0|A=1) = %f, want %f", pB0gA1, 3.0/7.0)
	}
	if !jpdFloatEq(pB1gA1, 4.0/7.0) {
		t.Errorf("P(B=1|A=1) = %f, want %f", pB1gA1, 4.0/7.0)
	}
}

func TestConditionalDistribution_3Var(t *testing.T) {
	// P(A,B,C) - using same distribution as above
	vals := []float64{0.05, 0.10, 0.05, 0.10, 0.15, 0.10, 0.20, 0.25}
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B", "C"},
		[]int{2, 2, 2},
		vals,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// P(A | B=1, C=0): need P(A=0,B=1,C=0)=0.05, P(A=1,B=1,C=0)=0.20
	// sum = 0.25 => P(A=0|B=1,C=0)=0.2, P(A=1|B=1,C=0)=0.8
	cond, err := jpd.ConditionalDistribution([]string{"A"}, map[string]int{"B": 1, "C": 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pA0 := cond.GetValue(map[string]int{"A": 0})
	pA1 := cond.GetValue(map[string]int{"A": 1})
	if !jpdFloatEq(pA0, 0.2) {
		t.Errorf("P(A=0|B=1,C=0) = %f, want 0.2", pA0)
	}
	if !jpdFloatEq(pA1, 0.8) {
		t.Errorf("P(A=1|B=1,C=0) = %f, want 0.8", pA1)
	}
}

func TestConditionalDistribution_Errors(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// No query variables
	_, err = jpd.ConditionalDistribution([]string{}, map[string]int{"B": 0})
	if err == nil {
		t.Fatal("expected error for empty query variables")
	}

	// No evidence
	_, err = jpd.ConditionalDistribution([]string{"A"}, map[string]int{})
	if err == nil {
		t.Fatal("expected error for empty evidence")
	}

	// Variable both query and evidence
	_, err = jpd.ConditionalDistribution([]string{"A"}, map[string]int{"A": 0})
	if err == nil {
		t.Fatal("expected error for variable in both query and evidence")
	}

	// Unknown variable
	_, err = jpd.ConditionalDistribution([]string{"Z"}, map[string]int{"B": 0})
	if err == nil {
		t.Fatal("expected error for unknown query variable")
	}
}

// ---------------------------------------------------------------------------
// CheckIndependence
// ---------------------------------------------------------------------------

func TestCheckIndependence_IndependentVars(t *testing.T) {
	// Construct a distribution where A and B are independent.
	// P(A=0)=0.3, P(A=1)=0.7, P(B=0)=0.4, P(B=1)=0.6
	// P(A,B) = P(A)*P(B):
	//   P(A=0,B=0) = 0.12
	//   P(A=0,B=1) = 0.18
	//   P(A=1,B=0) = 0.28
	//   P(A=1,B=1) = 0.42
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.12, 0.18, 0.28, 0.42},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !jpd.CheckIndependence("A", "B", nil, 1e-9) {
		t.Error("A and B should be independent")
	}
}

func TestCheckIndependence_DependentVars(t *testing.T) {
	// P(A,B) where A and B are NOT independent
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if jpd.CheckIndependence("A", "B", nil, 1e-9) {
		t.Error("A and B should NOT be independent")
	}
}

func TestCheckIndependence_ConditionallyIndependent(t *testing.T) {
	// Create P(A,B,C) where A _|_ B | C but A and B are marginally dependent.
	// Use a structure: C -> A, C -> B (A and B are conditionally independent given C).
	//
	// Let C binary, A binary, B binary.
	// P(C=0) = 0.5, P(C=1) = 0.5
	// P(A=0|C=0) = 0.8, P(A=1|C=0) = 0.2
	// P(A=0|C=1) = 0.2, P(A=1|C=1) = 0.8
	// P(B=0|C=0) = 0.7, P(B=1|C=0) = 0.3
	// P(B=0|C=1) = 0.3, P(B=1|C=1) = 0.7
	//
	// P(A,B,C) = P(A|C)*P(B|C)*P(C)
	// Order: A, B, C (row-major)
	// (A=0,B=0,C=0) = 0.8*0.7*0.5 = 0.28
	// (A=0,B=0,C=1) = 0.2*0.3*0.5 = 0.03
	// (A=0,B=1,C=0) = 0.8*0.3*0.5 = 0.12
	// (A=0,B=1,C=1) = 0.2*0.7*0.5 = 0.07
	// (A=1,B=0,C=0) = 0.2*0.7*0.5 = 0.07
	// (A=1,B=0,C=1) = 0.8*0.3*0.5 = 0.12
	// (A=1,B=1,C=0) = 0.2*0.3*0.5 = 0.03
	// (A=1,B=1,C=1) = 0.8*0.7*0.5 = 0.28
	vals := []float64{0.28, 0.03, 0.12, 0.07, 0.07, 0.12, 0.03, 0.28}
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B", "C"},
		[]int{2, 2, 2},
		vals,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// A _|_ B | C should be TRUE
	if !jpd.CheckIndependence("A", "B", []string{"C"}, 1e-9) {
		t.Error("A and B should be conditionally independent given C")
	}

	// A _|_ B (marginal) should be FALSE (they are marginally dependent)
	if jpd.CheckIndependence("A", "B", nil, 1e-9) {
		t.Error("A and B should be marginally dependent")
	}

	// A _|_ C should be FALSE
	if jpd.CheckIndependence("A", "C", nil, 1e-9) {
		t.Error("A and C should NOT be independent")
	}
}

func TestCheckIndependence_UnknownVariable(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.12, 0.18, 0.28, 0.42},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if jpd.CheckIndependence("A", "Z", nil, 1e-9) {
		t.Error("should return false for unknown variable")
	}
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

func TestCopy_JPD(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cp := jpd.Copy()

	// Modify original, check copy is unaffected.
	jpd.SetValue(map[string]int{"A": 0, "B": 0}, 0.99)
	orig := jpd.GetValue(map[string]int{"A": 0, "B": 0})
	copied := cp.GetValue(map[string]int{"A": 0, "B": 0})
	if !jpdFloatEq(orig, 0.99) {
		t.Errorf("original should be 0.99, got %f", orig)
	}
	if !jpdFloatEq(copied, 0.1) {
		t.Errorf("copy should be 0.1, got %f", copied)
	}
}

// ---------------------------------------------------------------------------
// MarginalDistribution keeps all variables (identity)
// ---------------------------------------------------------------------------

func TestMarginalDistribution_AllVars(t *testing.T) {
	jpd, err := NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Keeping all variables should return a copy.
	mAB, err := jpd.MarginalDistribution([]string{"A", "B"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	val := mAB.GetValue(map[string]int{"A": 0, "B": 1})
	if !jpdFloatEq(val, 0.2) {
		t.Errorf("P(A=0,B=1) = %f, want 0.2", val)
	}
}
