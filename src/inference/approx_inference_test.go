//go:build unit

package inference

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// approxTol is the tolerance for approximate inference tests.
// Likelihood-weighted sampling with uniform proposals can be noisy,
// so we use a generous tolerance.
const approxTol = 0.05

// ---------------------------------------------------------------------------
// NewApproxInference
// ---------------------------------------------------------------------------

func TestNewApproxInference(t *testing.T) {
	fl := studentFactors()
	ai := NewApproxInference(fl, 42)
	if ai == nil {
		t.Fatal("NewApproxInference returned nil")
	}
	if len(ai.factors) != 5 {
		t.Errorf("expected 5 factors, got %d", len(ai.factors))
	}
	// Verify deep copy: modifying original should not affect AI.
	fl[0].Normalize()
	origVal := ai.factors[0].GetValue(map[string]int{"D": 0})
	if !floatEq(origVal, 0.6) {
		t.Errorf("expected deep copy to preserve 0.6, got %f", origVal)
	}
}

// ---------------------------------------------------------------------------
// Query — error cases
// ---------------------------------------------------------------------------

func TestApproxQuery_EmptyQueryVars(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query(nil, nil, 1000)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestApproxQuery_ZeroSamples(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query([]string{"D"}, nil, 0)
	if err == nil {
		t.Error("expected error for zero nSamples")
	}
}

func TestApproxQuery_NegativeSamples(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query([]string{"D"}, nil, -10)
	if err == nil {
		t.Error("expected error for negative nSamples")
	}
}

func TestApproxQuery_UnknownQueryVar(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query([]string{"X"}, nil, 1000)
	if err == nil {
		t.Error("expected error for unknown query variable")
	}
}

func TestApproxQuery_UnknownEvidenceVar(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query([]string{"D"}, map[string]int{"X": 0}, 1000)
	if err == nil {
		t.Error("expected error for unknown evidence variable")
	}
}

func TestApproxQuery_EvidenceOutOfRange(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	_, err := ai.Query([]string{"D"}, map[string]int{"I": 5}, 1000)
	if err == nil {
		t.Error("expected error for evidence value out of range")
	}
}

// ---------------------------------------------------------------------------
// Query — marginal without evidence (convergence tests)
// ---------------------------------------------------------------------------

func TestApproxQuery_MarginalD(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	result, err := ai.Query([]string{"D"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"D": 0}), 0.6, approxTol, "P(D=0)")
	assertNear(t, result.GetValue(map[string]int{"D": 1}), 0.4, approxTol, "P(D=1)")
}

func TestApproxQuery_MarginalI(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 123)
	result, err := ai.Query([]string{"I"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"I": 0}), 0.7, approxTol, "P(I=0)")
	assertNear(t, result.GetValue(map[string]int{"I": 1}), 0.3, approxTol, "P(I=1)")
}

func TestApproxQuery_MarginalG(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 99)
	result, err := ai.Query([]string{"G"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(G=0)=0.362, P(G=1)=0.2884, P(G=2)=0.3496
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.362, approxTol, "P(G=0)")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.2884, approxTol, "P(G=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.3496, approxTol, "P(G=2)")
}

func TestApproxQuery_MarginalL(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 77)
	result, err := ai.Query([]string{"L"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(L=0)=0.497664
	assertNear(t, result.GetValue(map[string]int{"L": 0}), 0.497664, approxTol, "P(L=0)")
	assertNear(t, result.GetValue(map[string]int{"L": 1}), 1.0-0.497664, approxTol, "P(L=1)")
}

func TestApproxQuery_MarginalS(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 55)
	result, err := ai.Query([]string{"S"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(S=0)=0.725, P(S=1)=0.275
	assertNear(t, result.GetValue(map[string]int{"S": 0}), 0.725, approxTol, "P(S=0)")
	assertNear(t, result.GetValue(map[string]int{"S": 1}), 0.275, approxTol, "P(S=1)")
}

// ---------------------------------------------------------------------------
// Query — with evidence (convergence tests)
// ---------------------------------------------------------------------------

func TestApproxQuery_GradeGivenDifficulty(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	// P(G | D=1) — hard difficulty
	result, err := ai.Query([]string{"G"}, map[string]int{"D": 1}, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(G=0|D=1)=0.185, P(G=1|D=1)=0.265, P(G=2|D=1)=0.55
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.185, approxTol, "P(G=0|D=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.265, approxTol, "P(G=1|D=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.55, approxTol, "P(G=2|D=1)")
}

func TestApproxQuery_MultipleEvidence(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	// P(G | D=0, I=1) — should match the CPD column exactly with enough samples
	result, err := ai.Query([]string{"G"}, map[string]int{"D": 0, "I": 1}, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(G=0|D=0,I=1)=0.9, P(G=1|D=0,I=1)=0.08, P(G=2|D=0,I=1)=0.02
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.9, approxTol, "P(G=0|D=0,I=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.08, approxTol, "P(G=1|D=0,I=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.02, approxTol, "P(G=2|D=0,I=1)")
}

func TestApproxQuery_LetterGivenIntelligence(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	// P(L | I=1) — high intelligence
	result, err := ai.Query([]string{"L"}, map[string]int{"I": 1}, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(L=0|I=1)=0.23228
	assertNear(t, result.GetValue(map[string]int{"L": 0}), 0.23228, approxTol, "P(L=0|I=1)")
	assertNear(t, result.GetValue(map[string]int{"L": 1}), 1.0-0.23228, approxTol, "P(L=1|I=1)")
}

func TestApproxQuery_IntelligenceGivenGrade(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	// P(I | G=0) — good grade
	result, err := ai.Query([]string{"I"}, map[string]int{"G": 0}, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(I=0|G=0)=0.14/0.362 ≈ 0.38674, P(I=1|G=0) ≈ 0.61326
	assertNear(t, result.GetValue(map[string]int{"I": 0}), 0.14/0.362, approxTol, "P(I=0|G=0)")
	assertNear(t, result.GetValue(map[string]int{"I": 1}), 0.222/0.362, approxTol, "P(I=1|G=0)")
}

// ---------------------------------------------------------------------------
// Query — joint distribution
// ---------------------------------------------------------------------------

func TestApproxQuery_JointDI(t *testing.T) {
	ai := NewApproxInference(studentFactors(), 42)
	result, err := ai.Query([]string{"D", "I"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)

	// Exact: P(D=0,I=0)=0.42, P(D=0,I=1)=0.18, P(D=1,I=0)=0.28, P(D=1,I=1)=0.12
	assertNear(t, result.GetValue(map[string]int{"D": 0, "I": 0}), 0.42, approxTol, "P(D=0,I=0)")
	assertNear(t, result.GetValue(map[string]int{"D": 0, "I": 1}), 0.18, approxTol, "P(D=0,I=1)")
	assertNear(t, result.GetValue(map[string]int{"D": 1, "I": 0}), 0.28, approxTol, "P(D=1,I=0)")
	assertNear(t, result.GetValue(map[string]int{"D": 1, "I": 1}), 0.12, approxTol, "P(D=1,I=1)")
}

// ---------------------------------------------------------------------------
// Convergence: compare approximate vs exact VE
// ---------------------------------------------------------------------------

func TestApproxQuery_ConvergesToVE(t *testing.T) {
	// Run the same queries with VE and ApproxInference, verify they agree
	// within tolerance.
	fl := studentFactors()
	ve := NewVariableElimination(fl)
	ai := NewApproxInference(fl, 42)

	tests := []struct {
		name      string
		queryVars []string
		evidence  map[string]int
	}{
		{"P(D)", []string{"D"}, nil},
		{"P(I)", []string{"I"}, nil},
		{"P(G)", []string{"G"}, nil},
		{"P(L)", []string{"L"}, nil},
		{"P(S)", []string{"S"}, nil},
		{"P(G|D=1)", []string{"G"}, map[string]int{"D": 1}},
		{"P(I|G=0)", []string{"I"}, map[string]int{"G": 0}},
		{"P(L|I=1)", []string{"L"}, map[string]int{"I": 1}},
		{"P(G|D=0,I=1)", []string{"G"}, map[string]int{"D": 0, "I": 1}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exact, err := ve.Query(tc.queryVars, tc.evidence)
			if err != nil {
				t.Fatalf("VE query failed: %v", err)
			}

			approx, err := ai.Query(tc.queryVars, tc.evidence, 500000)
			if err != nil {
				t.Fatalf("ApproxInference query failed: %v", err)
			}

			// Compare all values.
			exactData := exact.Values().Data()
			approxData := approx.Values().Data()

			if len(exactData) != len(approxData) {
				t.Fatalf("result size mismatch: exact=%d, approx=%d", len(exactData), len(approxData))
			}

			for i := range exactData {
				if !floatNear(exactData[i], approxData[i], approxTol) {
					t.Errorf("index %d: exact=%f, approx=%f (tol=%f)", i, exactData[i], approxData[i], approxTol)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Simple two-variable network
// ---------------------------------------------------------------------------

func TestApproxQuery_SimpleNetwork(t *testing.T) {
	// A -> B
	// P(A) = [0.4, 0.6]
	// P(B|A): P(B=0|A=0)=0.2, P(B=0|A=1)=0.3, P(B=1|A=0)=0.8, P(B=1|A=1)=0.7
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	ai := NewApproxInference([]*factors.DiscreteFactor{pA, pBA}, 42)

	// P(B) = [0.26, 0.74]
	result, err := ai.Query([]string{"B"}, nil, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"B": 0}), 0.26, approxTol, "P(B=0)")
	assertNear(t, result.GetValue(map[string]int{"B": 1}), 0.74, approxTol, "P(B=1)")

	// P(A | B=0) = [0.08/0.26, 0.18/0.26]
	result2, err := ai.Query([]string{"A"}, map[string]int{"B": 0}, 500000)
	if err != nil {
		t.Fatal(err)
	}
	assertApproxSumsToOne(t, result2)
	assertNear(t, result2.GetValue(map[string]int{"A": 0}), 0.08/0.26, approxTol, "P(A=0|B=0)")
	assertNear(t, result2.GetValue(map[string]int{"A": 1}), 0.18/0.26, approxTol, "P(A=1|B=0)")
}

// ---------------------------------------------------------------------------
// Reproducibility: same seed gives same results
// ---------------------------------------------------------------------------

func TestApproxQuery_Reproducibility(t *testing.T) {
	fl := studentFactors()
	ai1 := NewApproxInference(fl, 42)
	ai2 := NewApproxInference(fl, 42)

	r1, err := ai1.Query([]string{"G"}, map[string]int{"D": 1}, 10000)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := ai2.Query([]string{"G"}, map[string]int{"D": 1}, 10000)
	if err != nil {
		t.Fatal(err)
	}

	d1 := r1.Values().Data()
	d2 := r2.Values().Data()
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Errorf("index %d: got %f and %f, expected identical", i, d1[i], d2[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

func assertApproxSumsToOne(t *testing.T, f *factors.DiscreteFactor) {
	t.Helper()
	data := f.Values().Data()
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("factor values sum to %f, want 1.0", sum)
	}
}
