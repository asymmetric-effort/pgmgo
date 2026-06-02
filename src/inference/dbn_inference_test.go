//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// DBN test helpers
// ---------------------------------------------------------------------------
//
// Simple 2-time-slice model:
//
//   X(t-1) -> X(t)
//
// X has cardinality 2.
// P(X_0) = [0.8, 0.2]  (initial distribution)
// P(X_t | X_{t-1}):
//   P(X=0 | X_prev=0) = 0.9,  P(X=1 | X_prev=0) = 0.1
//   P(X=0 | X_prev=1) = 0.3,  P(X=1 | X_prev=1) = 0.7
//
// The transition factor is over (X, X_prev) with shape [2, 2]:
//   row-major: X varies slowest
//   [P(X=0|Xp=0), P(X=0|Xp=1), P(X=1|Xp=0), P(X=1|Xp=1)]
//   = [0.9, 0.3, 0.1, 0.7]

func simpleDBNInitialFactors() []*factors.DiscreteFactor {
	pX0, _ := factors.NewDiscreteFactor(
		[]string{"X"}, []int{2},
		[]float64{0.8, 0.2},
	)
	return []*factors.DiscreteFactor{pX0}
}

func simpleDBNTransitionFactors() []*factors.DiscreteFactor {
	// P(X | X_prev) as a factor over (X, X_prev)
	pXgXp, _ := factors.NewDiscreteFactor(
		[]string{"X", "X_prev"}, []int{2, 2},
		[]float64{0.9, 0.3, 0.1, 0.7},
	)
	return []*factors.DiscreteFactor{pXgXp}
}

// ---------------------------------------------------------------------------
// DBNInference — construction
// ---------------------------------------------------------------------------

func TestNewDBNInference(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)
	if dbn == nil {
		t.Fatal("NewDBNInference returned nil")
	}
	if len(dbn.initialFactors) != 1 {
		t.Errorf("expected 1 initial factor, got %d", len(dbn.initialFactors))
	}
	if len(dbn.transitionFactors) != 1 {
		t.Errorf("expected 1 transition factor, got %d", len(dbn.transitionFactors))
	}
	if len(dbn.interfaceNodes) != 1 || dbn.interfaceNodes[0] != "X" {
		t.Errorf("expected interface nodes [X], got %v", dbn.interfaceNodes)
	}
}

func TestNewDBNInference_DeepCopy(t *testing.T) {
	init := simpleDBNInitialFactors()
	trans := simpleDBNTransitionFactors()
	dbn := NewDBNInference(init, trans, []string{"X"})

	// Modify originals.
	init[0].Normalize()
	trans[0].Normalize()

	// DBN's copies should be unchanged.
	val := dbn.initialFactors[0].GetValue(map[string]int{"X": 0})
	if !floatEq(val, 0.8) {
		t.Errorf("expected deep copy to preserve 0.8, got %f", val)
	}
}

// ---------------------------------------------------------------------------
// ForwardInference — single time step (t=0 only)
// ---------------------------------------------------------------------------

func TestDBN_ForwardInference_SingleStep(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Single time step, no evidence: should return P(X) = [0.8, 0.2]
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.8, 1e-6, "P(X=0) at t=0")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.2, 1e-6, "P(X=1) at t=0")
}

func TestDBN_ForwardInference_SingleStep_WithEvidence(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Single time step with evidence X=1: P(X=1|X=1) = 1.0
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{"X": 1}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.0, 1e-6, "P(X=0|X=1) at t=0")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 1.0, 1e-6, "P(X=1|X=1) at t=0")
}

// ---------------------------------------------------------------------------
// ForwardInference — two time steps
// ---------------------------------------------------------------------------

func TestDBN_ForwardInference_TwoSteps_NoEvidence(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Two time steps, no evidence at either step.
	// P(X_1) = sum_{X_0} P(X_1 | X_0) * P(X_0)
	// P(X_1=0) = P(X_1=0|X_0=0)*P(X_0=0) + P(X_1=0|X_0=1)*P(X_0=1)
	//          = 0.9*0.8 + 0.3*0.2 = 0.72 + 0.06 = 0.78
	// P(X_1=1) = 0.1*0.8 + 0.7*0.2 = 0.08 + 0.14 = 0.22
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{}, {}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.78, 1e-6, "P(X_1=0)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.22, 1e-6, "P(X_1=1)")
}

func TestDBN_ForwardInference_TwoSteps_EvidenceAtT0(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Evidence at t=0: X=1, then propagate forward.
	// P(X_0) given X_0=1: delta at X=1
	// P(X_1) = P(X_1 | X_0=1)
	// P(X_1=0) = 0.3, P(X_1=1) = 0.7
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{"X": 1}, {}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.3, 1e-6, "P(X_1=0|X_0=1)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.7, 1e-6, "P(X_1=1|X_0=1)")
}

func TestDBN_ForwardInference_TwoSteps_EvidenceAtT0_X0(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Evidence at t=0: X=0
	// P(X_1 | X_0=0) = [0.9, 0.1]
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{"X": 0}, {}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.9, 1e-6, "P(X_1=0|X_0=0)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.1, 1e-6, "P(X_1=1|X_0=0)")
}

// ---------------------------------------------------------------------------
// ForwardInference — three time steps
// ---------------------------------------------------------------------------

func TestDBN_ForwardInference_ThreeSteps_NoEvidence(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)

	// Three time steps, no evidence.
	// P(X_1) = [0.78, 0.22] (from two-step test)
	// P(X_2) = P(X_2=0|X_1=0)*0.78 + P(X_2=0|X_1=1)*0.22
	//        = 0.9*0.78 + 0.3*0.22 = 0.702 + 0.066 = 0.768
	// P(X_2=1) = 0.1*0.78 + 0.7*0.22 = 0.078 + 0.154 = 0.232
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{}, {}, {}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.768, 1e-6, "P(X_2=0)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.232, 1e-6, "P(X_2=1)")
}

// ---------------------------------------------------------------------------
// ForwardInference — two-variable DBN
// ---------------------------------------------------------------------------
//
// X(t-1) -> X(t)
// X(t)   -> Y(t)
//
// P(X_0) = [0.5, 0.5]
// P(X_t | X_{t-1}): same as before [0.9, 0.3; 0.1, 0.7]
// P(Y_t | X_t): P(Y=0|X=0)=0.8, P(Y=0|X=1)=0.1, P(Y=1|X=0)=0.2, P(Y=1|X=1)=0.9

func twoVarDBNInitialFactors() []*factors.DiscreteFactor {
	pX0, _ := factors.NewDiscreteFactor(
		[]string{"X"}, []int{2},
		[]float64{0.5, 0.5},
	)
	pYX, _ := factors.NewDiscreteFactor(
		[]string{"Y", "X"}, []int{2, 2},
		[]float64{0.8, 0.1, 0.2, 0.9},
	)
	return []*factors.DiscreteFactor{pX0, pYX}
}

func twoVarDBNTransitionFactors() []*factors.DiscreteFactor {
	pXgXp, _ := factors.NewDiscreteFactor(
		[]string{"X", "X_prev"}, []int{2, 2},
		[]float64{0.9, 0.3, 0.1, 0.7},
	)
	pYX, _ := factors.NewDiscreteFactor(
		[]string{"Y", "X"}, []int{2, 2},
		[]float64{0.8, 0.1, 0.2, 0.9},
	)
	return []*factors.DiscreteFactor{pXgXp, pYX}
}

func TestDBN_ForwardInference_TwoVarDBN_SingleStep(t *testing.T) {
	dbn := NewDBNInference(
		twoVarDBNInitialFactors(),
		twoVarDBNTransitionFactors(),
		[]string{"X"},
	)

	// P(Y) at t=0: P(Y=0) = P(Y=0|X=0)*0.5 + P(Y=0|X=1)*0.5 = 0.8*0.5 + 0.1*0.5 = 0.45
	result, err := dbn.ForwardInference(
		[]string{"Y"},
		[]map[string]int{{}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"Y": 0}), 0.45, 1e-6, "P(Y=0) at t=0")
	assertNear(t, result.GetValue(map[string]int{"Y": 1}), 0.55, 1e-6, "P(Y=1) at t=0")
}

func TestDBN_ForwardInference_TwoVarDBN_TwoSteps_ObserveY(t *testing.T) {
	dbn := NewDBNInference(
		twoVarDBNInitialFactors(),
		twoVarDBNTransitionFactors(),
		[]string{"X"},
	)

	// Observe Y=0 at t=0, then query X at t=1.
	// At t=0: P(X|Y=0) proportional to P(Y=0|X)*P(X)
	//   P(X=0|Y=0) prop 0.8*0.5 = 0.4
	//   P(X=1|Y=0) prop 0.1*0.5 = 0.05
	//   Normalized: P(X=0|Y=0) = 0.4/0.45 = 8/9, P(X=1|Y=0) = 0.05/0.45 = 1/9
	// At t=1: P(X_1) = P(X_1|X_0=0)*(8/9) + P(X_1|X_0=1)*(1/9)
	//   P(X_1=0) = 0.9*(8/9) + 0.3*(1/9) = 0.8 + 1/30 = 0.8 + 0.03333 = 0.83333
	//   P(X_1=1) = 0.1*(8/9) + 0.7*(1/9) = 0.08889 + 0.07778 = 0.16667
	result, err := dbn.ForwardInference(
		[]string{"X"},
		[]map[string]int{{"Y": 0}, {}},
	)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.83333, 1e-4, "P(X_1=0|Y_0=0)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.16667, 1e-4, "P(X_1=1|Y_0=0)")
}

// ---------------------------------------------------------------------------
// ForwardInference — error cases
// ---------------------------------------------------------------------------

func TestDBN_ForwardInference_EmptyQueryVars(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)
	_, err := dbn.ForwardInference(nil, []map[string]int{{}})
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestDBN_ForwardInference_EmptyEvidenceSequence(t *testing.T) {
	dbn := NewDBNInference(
		simpleDBNInitialFactors(),
		simpleDBNTransitionFactors(),
		[]string{"X"},
	)
	_, err := dbn.ForwardInference([]string{"X"}, nil)
	if err == nil {
		t.Error("expected error for empty evidenceSequence")
	}
}
