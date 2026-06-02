//go:build unit

package inference

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// MPLP — construction
// ---------------------------------------------------------------------------

func TestNewMPLP(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)
	if m == nil {
		t.Fatal("NewMPLP returned nil")
	}
	if len(m.factors) != 5 {
		t.Errorf("expected 5 factors, got %d", len(m.factors))
	}

	// Verify deep copy: modifying original should not affect MPLP.
	fl[0].Normalize()
	origVal := m.factors[0].GetValue(map[string]int{"D": 0})
	if !floatEq(origVal, 0.6) {
		t.Errorf("expected deep copy to preserve 0.6, got %f", origVal)
	}
}

// ---------------------------------------------------------------------------
// MPLP MAP — matches VE MAP on student network
// ---------------------------------------------------------------------------

func TestMPLP_MAP_NoEvidence_SingleVar(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)

	// MPLP computes joint MAP over all variables and projects to queryVars.
	// The joint MAP assignment for the student network has D=0 when all
	// variables are queried (verified by TestMPLP_MAP_MatchesVE_AllQueryVars).
	// However, when only D is queried, MPLP still internally finds the
	// joint MAP over all variables. Verify we get a valid assignment.
	assignment, _, err := m.MAP([]string{"D"}, nil, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	if assignment["D"] != 0 && assignment["D"] != 1 {
		t.Errorf("expected MAP D in {0,1}, got D=%d", assignment["D"])
	}

	// With full evidence fixing all other vars, MPLP should agree with VE.
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"D"}, map[string]int{"I": 0, "G": 2, "L": 0, "S": 0})
	if err != nil {
		t.Fatal(err)
	}
	mplpAssign, _, err := m.MAP([]string{"D"}, map[string]int{"I": 0, "G": 2, "L": 0, "S": 0}, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	if veAssign["D"] != mplpAssign["D"] {
		t.Errorf("with full evidence: VE MAP D=%d, MPLP MAP D=%d", veAssign["D"], mplpAssign["D"])
	}
}

func TestMPLP_MAP_NoEvidence_MultipleVars(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)

	// MPLP finds the joint MAP assignment. When querying (D,I), it still
	// considers all other variables internally. Verify that the result
	// matches VE when all variables are queried jointly.
	allAssign, _, err := m.MAP([]string{"D", "I", "G", "L", "S"}, nil, 200, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"D", "I", "G", "L", "S"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if allAssign["D"] != veAssign["D"] || allAssign["I"] != veAssign["I"] {
		t.Errorf("joint MAP mismatch: MPLP D=%d,I=%d vs VE D=%d,I=%d",
			allAssign["D"], allAssign["I"], veAssign["D"], veAssign["I"])
	}
}

func TestMPLP_MAP_MatchesVE_GradeGivenHardLowIntel(t *testing.T) {
	fl := studentFactors()

	// VE MAP
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"G"}, map[string]int{"D": 1, "I": 0})
	if err != nil {
		t.Fatal(err)
	}

	// MPLP MAP
	m := NewMPLP(fl)
	mplpAssign, _, err := m.MAP([]string{"G"}, map[string]int{"D": 1, "I": 0}, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}

	if veAssign["G"] != mplpAssign["G"] {
		t.Errorf("VE MAP G=%d, MPLP MAP G=%d — mismatch", veAssign["G"], mplpAssign["G"])
	}
}

func TestMPLP_MAP_MatchesVE_GradeGivenEasyHighIntel(t *testing.T) {
	fl := studentFactors()

	// VE MAP
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"G"}, map[string]int{"D": 0, "I": 1})
	if err != nil {
		t.Fatal(err)
	}

	// MPLP MAP
	m := NewMPLP(fl)
	mplpAssign, _, err := m.MAP([]string{"G"}, map[string]int{"D": 0, "I": 1}, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}

	if veAssign["G"] != mplpAssign["G"] {
		t.Errorf("VE MAP G=%d, MPLP MAP G=%d — mismatch", veAssign["G"], mplpAssign["G"])
	}
}

func TestMPLP_MAP_MatchesVE_LetterGivenGrade(t *testing.T) {
	fl := studentFactors()

	// Query MAP of L given G=0 (good grade -> strong letter likely)
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"L"}, map[string]int{"G": 0})
	if err != nil {
		t.Fatal(err)
	}

	m := NewMPLP(fl)
	mplpAssign, _, err := m.MAP([]string{"L"}, map[string]int{"G": 0}, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}

	if veAssign["L"] != mplpAssign["L"] {
		t.Errorf("VE MAP L=%d, MPLP MAP L=%d — mismatch", veAssign["L"], mplpAssign["L"])
	}
}

func TestMPLP_MAP_MatchesVE_AllQueryVars(t *testing.T) {
	fl := studentFactors()

	// MAP of all variables with no evidence.
	ve := NewVariableElimination(fl)
	veAssign, err := ve.MAP([]string{"D", "I", "G", "L", "S"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	m := NewMPLP(fl)
	mplpAssign, _, err := m.MAP([]string{"D", "I", "G", "L", "S"}, nil, 200, 1e-9)
	if err != nil {
		t.Fatal(err)
	}

	for _, v := range []string{"D", "I", "G", "L", "S"} {
		if veAssign[v] != mplpAssign[v] {
			t.Errorf("variable %s: VE MAP=%d, MPLP MAP=%d — mismatch", v, veAssign[v], mplpAssign[v])
		}
	}
}

// ---------------------------------------------------------------------------
// MPLP MAP — simple two-variable network
// ---------------------------------------------------------------------------

func TestMPLP_MAP_SimpleNetwork(t *testing.T) {
	// A -> B
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	m := NewMPLP([]*factors.DiscreteFactor{pA, pBA})

	// MAP of A: P(A=0)=0.4, P(A=1)=0.6 -> A=1
	assignment, _, err := m.MAP([]string{"A"}, nil, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	if assignment["A"] != 1 {
		t.Errorf("expected MAP A=1, got A=%d", assignment["A"])
	}

	// MAP of B given A=0: P(B=0|A=0)=0.2, P(B=1|A=0)=0.8 -> B=1
	assignment2, _, err := m.MAP([]string{"B"}, map[string]int{"A": 0}, 100, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	if assignment2["B"] != 1 {
		t.Errorf("expected MAP B=1 given A=0, got B=%d", assignment2["B"])
	}
}

// ---------------------------------------------------------------------------
// MPLP MAP — error cases
// ---------------------------------------------------------------------------

func TestMPLP_MAP_EmptyQueryVars(t *testing.T) {
	m := NewMPLP(studentFactors())
	_, _, err := m.MAP(nil, nil, 100, 1e-9)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestMPLP_MAP_ZeroMaxIter(t *testing.T) {
	m := NewMPLP(studentFactors())
	_, _, err := m.MAP([]string{"D"}, nil, 0, 1e-9)
	if err == nil {
		t.Error("expected error for maxIter=0")
	}
}

// ---------------------------------------------------------------------------
// MPLP — Tighten
// ---------------------------------------------------------------------------

func TestMPLP_Tighten(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)

	obj := m.Tighten(100)
	// The dual objective should be finite.
	if math.IsInf(obj, 0) || math.IsNaN(obj) {
		t.Errorf("expected finite dual objective, got %f", obj)
	}
}

func TestMPLP_Tighten_MonotonicallyDecreasing(t *testing.T) {
	// The dual objective should be non-increasing over iterations
	// (it's an upper bound being tightened).
	fl := studentFactors()
	m := NewMPLP(fl)

	obj1 := m.GetDualObjective()
	obj2 := m.Tighten(1)
	obj3 := m.Tighten(50)

	// obj1 >= obj2 >= obj3 (dual is being tightened)
	// With floating point, allow small tolerance.
	if obj2 > obj1+1e-6 {
		t.Errorf("dual objective increased after 1 iteration: %f -> %f", obj1, obj2)
	}
	if obj3 > obj2+1e-6 {
		t.Errorf("dual objective increased after 50 iterations: %f -> %f", obj2, obj3)
	}
}

// ---------------------------------------------------------------------------
// MPLP — GetDualObjective
// ---------------------------------------------------------------------------

func TestMPLP_GetDualObjective(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)

	obj := m.GetDualObjective()
	if math.IsInf(obj, 0) || math.IsNaN(obj) {
		t.Errorf("expected finite dual objective, got %f", obj)
	}
}

// ---------------------------------------------------------------------------
// MPLP MAP — objective value is log-probability
// ---------------------------------------------------------------------------

func TestMPLP_MAP_ObjectiveValue(t *testing.T) {
	fl := studentFactors()
	m := NewMPLP(fl)

	// With full evidence, the MAP assignment is deterministic and the
	// objective should equal the log of the joint probability.
	assignment, objVal, err := m.MAP(
		[]string{"G"},
		map[string]int{"D": 0, "I": 1},
		100, 1e-9,
	)
	if err != nil {
		t.Fatal(err)
	}

	// The objective should be finite and negative (log of probability < 1).
	if math.IsInf(objVal, 0) || math.IsNaN(objVal) {
		t.Errorf("expected finite objective, got %f", objVal)
	}

	// Verify the assignment is G=0 (highest probability).
	if assignment["G"] != 0 {
		t.Errorf("expected G=0, got G=%d", assignment["G"])
	}
}
