//go:build unit

package inference

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// Helper: build a simple A->B junction tree
// ---------------------------------------------------------------------------
//
// A -> B
// P(A) = [0.4, 0.6]
// P(B|A): B=0|A=0=0.2, B=0|A=1=0.3, B=1|A=0=0.8, B=1|A=1=0.7
//
// Junction tree: single clique {A, B}
func simpleABJunctionTree() *BeliefPropagation {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
	}
	return NewBeliefPropagation(cliques, separators, cliqueFactors)
}

// ---------------------------------------------------------------------------
// Helper: build a chain A->B->C junction tree
// ---------------------------------------------------------------------------
//
// A -> B -> C
// P(A) = [0.4, 0.6]
// P(B|A)
// P(C|B)
//
// Junction tree: clique0={A,B} -- {B} -- clique1={B,C}
func chainABCJunctionTree() (*BeliefPropagation, []*factors.DiscreteFactor) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{
		0.2, 0.3,
		0.8, 0.7,
	})
	pCB, _ := factors.NewDiscreteFactor([]string{"C", "B"}, []int{2, 2}, []float64{
		0.5, 0.1,
		0.5, 0.9,
	})

	cliques := [][]string{
		{"A", "B"},
		{"B", "C"},
	}
	separators := map[string][]string{
		edgeKey(0, 1): {"B"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
		1: {pCB},
	}
	return NewBeliefPropagation(cliques, separators, cliqueFactors),
		[]*factors.DiscreteFactor{pA, pBA, pCB}
}

// ---------------------------------------------------------------------------
// Helper: build the student network junction tree
// ---------------------------------------------------------------------------
//
// Student Bayesian network:
//
//	D -> G <- I
//	     G -> L
//	     I -> S
//
// A valid junction tree for this network:
//
//	clique0 = {D, G, I}  (contains P(D), P(I), P(G|D,I))
//	clique1 = {G, L}     (contains P(L|G))
//	clique2 = {I, S}     (contains P(S|I))
//
// Edges:
//
//	0-1 separator {G}
//	0-2 separator {I}
func studentJunctionTree() (*BeliefPropagation, []*factors.DiscreteFactor) {
	fl := studentFactors()
	pD := fl[0]   // P(D)
	pI := fl[1]   // P(I)
	pGDI := fl[2] // P(G|D,I)
	pLG := fl[3]  // P(L|G)
	pSI := fl[4]  // P(S|I)

	cliques := [][]string{
		{"D", "G", "I"},
		{"G", "L"},
		{"I", "S"},
	}
	separators := map[string][]string{
		edgeKey(0, 1): {"G"},
		edgeKey(0, 2): {"I"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pD, pI, pGDI},
		1: {pLG},
		2: {pSI},
	}

	return NewBeliefPropagation(cliques, separators, cliqueFactors), fl
}

// ---------------------------------------------------------------------------
// TestNewBeliefPropagation
// ---------------------------------------------------------------------------

func TestNewBeliefPropagation(t *testing.T) {
	bp, _ := studentJunctionTree()
	if bp == nil {
		t.Fatal("NewBeliefPropagation returned nil")
	}
	if len(bp.cliques) != 3 {
		t.Errorf("expected 3 cliques, got %d", len(bp.cliques))
	}
	if bp.IsCalibrated() {
		t.Error("should not be calibrated before calling Calibrate()")
	}
}

func TestNewBeliefPropagation_DeepCopy(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	cliques := [][]string{{"A"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA}}

	bp := NewBeliefPropagation(cliques, nil, cliqueFactors)

	// Modify original: should not affect BP.
	pA.Normalize() // already normalized, but modifies in-place
	cliques[0][0] = "Z"

	if bp.cliques[0][0] != "A" {
		t.Error("clique was not deep-copied")
	}
}

// ---------------------------------------------------------------------------
// TestIsCalibrated
// ---------------------------------------------------------------------------

func TestIsCalibrated_BeforeAndAfter(t *testing.T) {
	bp := simpleABJunctionTree()
	if bp.IsCalibrated() {
		t.Error("should not be calibrated before Calibrate()")
	}
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("should be calibrated after Calibrate()")
	}
}

// ---------------------------------------------------------------------------
// TestCalibrate — single clique
// ---------------------------------------------------------------------------

func TestCalibrate_SingleClique(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	belief := bp.GetCliqueBelief(0)
	if belief == nil {
		t.Fatal("clique belief is nil")
	}

	// The joint P(A,B) = P(A)*P(B|A).
	// P(A=0,B=0) = 0.4*0.2 = 0.08
	// P(A=0,B=1) = 0.4*0.8 = 0.32
	// P(A=1,B=0) = 0.6*0.3 = 0.18
	// P(A=1,B=1) = 0.6*0.7 = 0.42
	total := 0.08 + 0.32 + 0.18 + 0.42 // = 1.0
	_ = total

	// Check the unnormalized joint values.
	assertNear(t, belief.GetValue(map[string]int{"A": 0, "B": 0}), 0.08, 1e-9, "P(A=0,B=0)")
	assertNear(t, belief.GetValue(map[string]int{"A": 0, "B": 1}), 0.32, 1e-9, "P(A=0,B=1)")
	assertNear(t, belief.GetValue(map[string]int{"A": 1, "B": 0}), 0.18, 1e-9, "P(A=1,B=0)")
	assertNear(t, belief.GetValue(map[string]int{"A": 1, "B": 1}), 0.42, 1e-9, "P(A=1,B=1)")
}

// ---------------------------------------------------------------------------
// TestCalibrate — chain A->B->C
// ---------------------------------------------------------------------------

func TestCalibrate_Chain(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Fatal("not calibrated after Calibrate()")
	}

	// Verify both cliques are calibrated: marginalizing each clique's
	// belief over the separator {B} should yield the same distribution.
	belief0 := bp.GetCliqueBelief(0)
	belief1 := bp.GetCliqueBelief(1)

	marg0, err := belief0.Marginalize([]string{"A"})
	if err != nil {
		t.Fatal(err)
	}
	marg1, err := belief1.Marginalize([]string{"C"})
	if err != nil {
		t.Fatal(err)
	}

	for b := 0; b < 2; b++ {
		v0 := marg0.GetValue(map[string]int{"B": b})
		v1 := marg1.GetValue(map[string]int{"B": b})
		if !floatNear(v0, v1, 1e-9) {
			t.Errorf("separator inconsistency for B=%d: clique0 gives %f, clique1 gives %f", b, v0, v1)
		}
	}
}

// ---------------------------------------------------------------------------
// TestCalibrate — student network
// ---------------------------------------------------------------------------

func TestCalibrate_StudentNetwork(t *testing.T) {
	bp, _ := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	// Check separator consistency: clique0={D,G,I} and clique1={G,L}
	// share separator {G}. Marginalizing both beliefs to {G} should agree.
	b0 := bp.GetCliqueBelief(0)
	b1 := bp.GetCliqueBelief(1)

	marg0_G, err := b0.Marginalize([]string{"D", "I"})
	if err != nil {
		t.Fatal(err)
	}
	marg1_G, err := b1.Marginalize([]string{"L"})
	if err != nil {
		t.Fatal(err)
	}

	for g := 0; g < 3; g++ {
		v0 := marg0_G.GetValue(map[string]int{"G": g})
		v1 := marg1_G.GetValue(map[string]int{"G": g})
		if !floatNear(v0, v1, 1e-9) {
			t.Errorf("separator {G} inconsistency for G=%d: clique0=%f, clique1=%f", g, v0, v1)
		}
	}

	// Check separator consistency: clique0={D,G,I} and clique2={I,S}
	// share separator {I}.
	b2 := bp.GetCliqueBelief(2)

	marg0_I, err := b0.Marginalize([]string{"D", "G"})
	if err != nil {
		t.Fatal(err)
	}
	marg2_I, err := b2.Marginalize([]string{"S"})
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		v0 := marg0_I.GetValue(map[string]int{"I": i})
		v2 := marg2_I.GetValue(map[string]int{"I": i})
		if !floatNear(v0, v2, 1e-9) {
			t.Errorf("separator {I} inconsistency for I=%d: clique0=%f, clique2=%f", i, v0, v2)
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuery — simple network, compare with VE
// ---------------------------------------------------------------------------

func TestQuery_SimpleAB_MarginalB(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	result, err := bp.Query([]string{"B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	// P(B=0)=0.26, P(B=1)=0.74
	assertNear(t, result.GetValue(map[string]int{"B": 0}), 0.26, 1e-6, "P(B=0)")
	assertNear(t, result.GetValue(map[string]int{"B": 1}), 0.74, 1e-6, "P(B=1)")
}

func TestQuery_SimpleAB_AgivenB0(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	result, err := bp.Query([]string{"A"}, map[string]int{"B": 0})
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"A": 0}), 0.08/0.26, 1e-6, "P(A=0|B=0)")
	assertNear(t, result.GetValue(map[string]int{"A": 1}), 0.18/0.26, 1e-6, "P(A=1|B=0)")
}

// ---------------------------------------------------------------------------
// TestQuery — chain A->B->C
// ---------------------------------------------------------------------------

func TestQuery_Chain_MarginalC(t *testing.T) {
	bp, allFactors := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	// Compare with VE.
	ve := NewVariableElimination(allFactors)
	veResult, err := ve.Query([]string{"C"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for c := 0; c < 2; c++ {
		bpVal := bpResult.GetValue(map[string]int{"C": c})
		veVal := veResult.GetValue(map[string]int{"C": c})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(C=%d): BP=%f, VE=%f", c, bpVal, veVal)
		}
	}
}

func TestQuery_Chain_AgivenC1(t *testing.T) {
	bp, allFactors := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	evidence := map[string]int{"C": 1}
	bpResult, err := bp.Query([]string{"A"}, evidence)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(allFactors)
	veResult, err := ve.Query([]string{"A"}, evidence)
	if err != nil {
		t.Fatal(err)
	}

	for a := 0; a < 2; a++ {
		bpVal := bpResult.GetValue(map[string]int{"A": a})
		veVal := veResult.GetValue(map[string]int{"A": a})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(A=%d|C=1): BP=%f, VE=%f", a, bpVal, veVal)
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuery — student network marginals, compare with VE
// ---------------------------------------------------------------------------

func TestQuery_Student_MarginalD(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"D"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"D"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for d := 0; d < 2; d++ {
		bpVal := bpResult.GetValue(map[string]int{"D": d})
		veVal := veResult.GetValue(map[string]int{"D": d})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(D=%d): BP=%f, VE=%f", d, bpVal, veVal)
		}
	}
}

func TestQuery_Student_MarginalG(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"G"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"G"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for g := 0; g < 3; g++ {
		bpVal := bpResult.GetValue(map[string]int{"G": g})
		veVal := veResult.GetValue(map[string]int{"G": g})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(G=%d): BP=%f, VE=%f", g, bpVal, veVal)
		}
	}
}

func TestQuery_Student_MarginalL(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"L"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"L"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for l := 0; l < 2; l++ {
		bpVal := bpResult.GetValue(map[string]int{"L": l})
		veVal := veResult.GetValue(map[string]int{"L": l})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(L=%d): BP=%f, VE=%f", l, bpVal, veVal)
		}
	}
}

func TestQuery_Student_MarginalS(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"S"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"S"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for s := 0; s < 2; s++ {
		bpVal := bpResult.GetValue(map[string]int{"S": s})
		veVal := veResult.GetValue(map[string]int{"S": s})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(S=%d): BP=%f, VE=%f", s, bpVal, veVal)
		}
	}
}

func TestQuery_Student_MarginalI(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"I"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"I"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		bpVal := bpResult.GetValue(map[string]int{"I": i})
		veVal := veResult.GetValue(map[string]int{"I": i})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(I=%d): BP=%f, VE=%f", i, bpVal, veVal)
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuery — student network with evidence, compare with VE
// ---------------------------------------------------------------------------

func TestQuery_Student_GradeGivenDifficulty(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	evidence := map[string]int{"D": 1}
	bpResult, err := bp.Query([]string{"G"}, evidence)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"G"}, evidence)
	if err != nil {
		t.Fatal(err)
	}

	for g := 0; g < 3; g++ {
		bpVal := bpResult.GetValue(map[string]int{"G": g})
		veVal := veResult.GetValue(map[string]int{"G": g})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(G=%d|D=1): BP=%f, VE=%f", g, bpVal, veVal)
		}
	}
}

func TestQuery_Student_IntelligenceGivenGrade(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	evidence := map[string]int{"G": 0}
	bpResult, err := bp.Query([]string{"I"}, evidence)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"I"}, evidence)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		bpVal := bpResult.GetValue(map[string]int{"I": i})
		veVal := veResult.GetValue(map[string]int{"I": i})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(I=%d|G=0): BP=%f, VE=%f", i, bpVal, veVal)
		}
	}
}

func TestQuery_Student_LetterGivenIntelligence(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	evidence := map[string]int{"I": 1}
	bpResult, err := bp.Query([]string{"L"}, evidence)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"L"}, evidence)
	if err != nil {
		t.Fatal(err)
	}

	for l := 0; l < 2; l++ {
		bpVal := bpResult.GetValue(map[string]int{"L": l})
		veVal := veResult.GetValue(map[string]int{"L": l})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(L=%d|I=1): BP=%f, VE=%f", l, bpVal, veVal)
		}
	}
}

func TestQuery_Student_MultipleEvidence(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	evidence := map[string]int{"D": 0, "I": 1}
	bpResult, err := bp.Query([]string{"G"}, evidence)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"G"}, evidence)
	if err != nil {
		t.Fatal(err)
	}

	for g := 0; g < 3; g++ {
		bpVal := bpResult.GetValue(map[string]int{"G": g})
		veVal := veResult.GetValue(map[string]int{"G": g})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(G=%d|D=0,I=1): BP=%f, VE=%f", g, bpVal, veVal)
		}
	}
}

// ---------------------------------------------------------------------------
// TestQuery — joint distributions
// ---------------------------------------------------------------------------

func TestQuery_Student_JointDI(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	bpResult, err := bp.Query([]string{"D", "I"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, bpResult)

	ve := NewVariableElimination(fl)
	veResult, err := ve.Query([]string{"D", "I"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for d := 0; d < 2; d++ {
		for i := 0; i < 2; i++ {
			bpVal := bpResult.GetValue(map[string]int{"D": d, "I": i})
			veVal := veResult.GetValue(map[string]int{"D": d, "I": i})
			if !floatNear(bpVal, veVal, 1e-6) {
				t.Errorf("P(D=%d,I=%d): BP=%f, VE=%f", d, i, bpVal, veVal)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestGetCliqueBelief
// ---------------------------------------------------------------------------

func TestGetCliqueBelief_OutOfRange(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	if bp.GetCliqueBelief(-1) != nil {
		t.Error("expected nil for negative index")
	}
	if bp.GetCliqueBelief(999) != nil {
		t.Error("expected nil for out-of-range index")
	}
}

func TestGetCliqueBelief_ReturnsCopy(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	b1 := bp.GetCliqueBelief(0)
	b2 := bp.GetCliqueBelief(0)

	// Modify b1; b2 and internal state should be unaffected.
	b1.Normalize()
	origVal := b2.GetValue(map[string]int{"A": 0, "B": 0})
	assertNear(t, origVal, 0.08, 1e-9, "copy independence")
}

// ---------------------------------------------------------------------------
// TestQuery — error cases
// ---------------------------------------------------------------------------

func TestBP_Query_EmptyQueryVars(t *testing.T) {
	bp := simpleABJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	_, err := bp.Query(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestQuery_NotCalibrated(t *testing.T) {
	bp := simpleABJunctionTree()
	_, err := bp.Query([]string{"A"}, nil)
	if err == nil {
		t.Error("expected error when querying before calibration")
	}
}

func TestQuery_NoCliqueContainsQueryVars(t *testing.T) {
	// Chain A-B, B-C: no single clique contains {A, C}.
	bp, _ := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	_, err := bp.Query([]string{"A", "C"}, nil)
	if err == nil {
		t.Error("expected error when no clique contains all query vars")
	}
}

// ---------------------------------------------------------------------------
// TestCalibrate — empty junction tree
// ---------------------------------------------------------------------------

func TestCalibrate_Empty(t *testing.T) {
	bp := NewBeliefPropagation(nil, nil, nil)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("empty tree should be calibrated")
	}
}

// ---------------------------------------------------------------------------
// TestString
// ---------------------------------------------------------------------------

func TestString(t *testing.T) {
	bp := simpleABJunctionTree()
	s := bp.String()
	if len(s) == 0 {
		t.Error("String() returned empty")
	}
}

// ---------------------------------------------------------------------------
// Comprehensive VE comparison: all single-variable marginals for student net
// ---------------------------------------------------------------------------

func TestQuery_Student_AllMarginals_MatchVE(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	ve := NewVariableElimination(fl)

	vars := []struct {
		name string
		card int
	}{
		{"D", 2}, {"I", 2}, {"G", 3}, {"L", 2}, {"S", 2},
	}

	for _, v := range vars {
		bpResult, err := bp.Query([]string{v.name}, nil)
		if err != nil {
			t.Fatalf("BP Query(%s) failed: %v", v.name, err)
		}
		veResult, err := ve.Query([]string{v.name}, nil)
		if err != nil {
			t.Fatalf("VE Query(%s) failed: %v", v.name, err)
		}

		for state := 0; state < v.card; state++ {
			assign := map[string]int{v.name: state}
			bpVal := bpResult.GetValue(assign)
			veVal := veResult.GetValue(assign)
			if !floatNear(bpVal, veVal, 1e-6) {
				t.Errorf("P(%s=%d): BP=%f, VE=%f", v.name, state, bpVal, veVal)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Comprehensive VE comparison: various evidence combinations
// ---------------------------------------------------------------------------

func TestQuery_Student_VariousEvidence_MatchVE(t *testing.T) {
	bp, fl := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	ve := NewVariableElimination(fl)

	tests := []struct {
		query    []string
		evidence map[string]int
		card     []int // cardinalities of query vars
	}{
		{[]string{"G"}, map[string]int{"D": 0}, []int{3}},
		{[]string{"G"}, map[string]int{"I": 0}, []int{3}},
		{[]string{"L"}, map[string]int{"G": 2}, []int{2}},
		{[]string{"S"}, map[string]int{"I": 0}, []int{2}},
		{[]string{"D"}, map[string]int{"G": 1}, []int{2}},
		{[]string{"I"}, map[string]int{"G": 1}, []int{2}},
	}

	for _, tc := range tests {
		bpResult, err := bp.Query(tc.query, tc.evidence)
		if err != nil {
			t.Fatalf("BP Query(%v | %v) failed: %v", tc.query, tc.evidence, err)
		}
		veResult, err := ve.Query(tc.query, tc.evidence)
		if err != nil {
			t.Fatalf("VE Query(%v | %v) failed: %v", tc.query, tc.evidence, err)
		}
		assertSumsToOne(t, bpResult)

		totalStates := 1
		for _, c := range tc.card {
			totalStates *= c
		}
		for flat := 0; flat < totalStates; flat++ {
			assign := make(map[string]int)
			rem := flat
			for i := len(tc.query) - 1; i >= 0; i-- {
				assign[tc.query[i]] = rem % tc.card[i]
				rem /= tc.card[i]
			}
			bpVal := bpResult.GetValue(assign)
			veVal := veResult.GetValue(assign)
			if !floatNear(bpVal, veVal, 1e-6) {
				t.Errorf("Query(%v | %v) assign=%v: BP=%f, VE=%f",
					tc.query, tc.evidence, assign, bpVal, veVal)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Test with a diamond-shaped junction tree (4 cliques)
// ---------------------------------------------------------------------------
//
// Network: A->B, A->C, B->D, C->D
//
// Junction tree:
//   clique0={A,B}  --{B}--  clique1={B,D}
//        |                       |
//       {A}                     {D}
//        |                       |
//   clique2={A,C}  --{C}--  clique3={C,D}
//
// However a junction tree must be a tree (no cycles). A valid junction tree:
//   clique0={A,B,C}  --{B}-- clique1={B,D}
//   clique0={A,B,C}  --{C}-- clique2={C,D}
//
// This merges A,B,C into one clique.

func TestQuery_Diamond_MatchVE(t *testing.T) {
	// A -> B, A -> C, B -> D, C -> D
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{
		0.7, 0.3,
		0.3, 0.7,
	})
	pCA, _ := factors.NewDiscreteFactor([]string{"C", "A"}, []int{2, 2}, []float64{
		0.8, 0.2,
		0.2, 0.8,
	})
	pDBC, _ := factors.NewDiscreteFactor([]string{"D", "B", "C"}, []int{2, 2, 2}, []float64{
		0.9, 0.6, 0.5, 0.1,
		0.1, 0.4, 0.5, 0.9,
	})

	allFactors := []*factors.DiscreteFactor{pA, pBA, pCA, pDBC}

	// Junction tree: clique0={A,B,C}, clique1={B,C,D}
	// separator 0-1 = {B,C}
	cliques := [][]string{
		{"A", "B", "C"},
		{"B", "C", "D"},
	}
	separators := map[string][]string{
		edgeKey(0, 1): {"B", "C"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA, pCA},
		1: {pDBC},
	}

	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	ve := NewVariableElimination(allFactors)

	// Test all single-variable marginals.
	for _, v := range []struct {
		name string
		card int
	}{
		{"A", 2}, {"B", 2}, {"C", 2}, {"D", 2},
	} {
		bpResult, err := bp.Query([]string{v.name}, nil)
		if err != nil {
			t.Fatalf("BP Query(%s) failed: %v", v.name, err)
		}
		veResult, err := ve.Query([]string{v.name}, nil)
		if err != nil {
			t.Fatalf("VE Query(%s) failed: %v", v.name, err)
		}
		for state := 0; state < v.card; state++ {
			assign := map[string]int{v.name: state}
			bpVal := bpResult.GetValue(assign)
			veVal := veResult.GetValue(assign)
			if !floatNear(bpVal, veVal, 1e-6) {
				t.Errorf("Diamond P(%s=%d): BP=%f, VE=%f", v.name, state, bpVal, veVal)
			}
		}
	}

	// Test with evidence.
	bpResult, err := bp.Query([]string{"D"}, map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	veResult, err := ve.Query([]string{"D"}, map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	for d := 0; d < 2; d++ {
		bpVal := bpResult.GetValue(map[string]int{"D": d})
		veVal := veResult.GetValue(map[string]int{"D": d})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("Diamond P(D=%d|A=0): BP=%f, VE=%f", d, bpVal, veVal)
		}
	}
}

// ---------------------------------------------------------------------------
// Test with ternary variables
// ---------------------------------------------------------------------------

func TestQuery_TernaryVars_MatchVE(t *testing.T) {
	// X(3) -> Y(3)
	pX, _ := factors.NewDiscreteFactor([]string{"X"}, []int{3}, []float64{0.2, 0.5, 0.3})
	pYX, _ := factors.NewDiscreteFactor([]string{"Y", "X"}, []int{3, 3}, []float64{
		0.1, 0.3, 0.6,
		0.5, 0.4, 0.3,
		0.4, 0.3, 0.1,
	})

	allFactors := []*factors.DiscreteFactor{pX, pYX}

	cliques := [][]string{{"X", "Y"}}
	bp := NewBeliefPropagation(cliques, map[string][]string{}, map[int][]*factors.DiscreteFactor{
		0: {pX, pYX},
	})
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	ve := NewVariableElimination(allFactors)

	bpResult, err := bp.Query([]string{"Y"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	veResult, err := ve.Query([]string{"Y"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, bpResult)
	for y := 0; y < 3; y++ {
		bpVal := bpResult.GetValue(map[string]int{"Y": y})
		veVal := veResult.GetValue(map[string]int{"Y": y})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(Y=%d): BP=%f, VE=%f", y, bpVal, veVal)
		}
	}

	// With evidence.
	bpResult2, err := bp.Query([]string{"X"}, map[string]int{"Y": 2})
	if err != nil {
		t.Fatal(err)
	}
	veResult2, err := ve.Query([]string{"X"}, map[string]int{"Y": 2})
	if err != nil {
		t.Fatal(err)
	}

	assertSumsToOne(t, bpResult2)
	for x := 0; x < 3; x++ {
		bpVal := bpResult2.GetValue(map[string]int{"X": x})
		veVal := veResult2.GetValue(map[string]int{"X": x})
		if !floatNear(bpVal, veVal, 1e-6) {
			t.Errorf("P(X=%d|Y=2): BP=%f, VE=%f", x, bpVal, veVal)
		}
	}
}

// ---------------------------------------------------------------------------
// Test calibration property: separator marginals agree
// ---------------------------------------------------------------------------

func TestCalibration_SeparatorConsistency_Chain(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	// Clique 0 = {A,B}, Clique 1 = {B,C}, separator = {B}.
	b0 := bp.GetCliqueBelief(0)
	b1 := bp.GetCliqueBelief(1)

	// Marginalize each to {B}.
	m0, _ := b0.Marginalize([]string{"A"})
	m1, _ := b1.Marginalize([]string{"C"})

	for b := 0; b < 2; b++ {
		v0 := m0.GetValue(map[string]int{"B": b})
		v1 := m1.GetValue(map[string]int{"B": b})
		if !floatNear(v0, v1, 1e-9) {
			t.Errorf("separator {B} mismatch at B=%d: %f vs %f", b, v0, v1)
		}
	}
}

// ---------------------------------------------------------------------------
// Test: beliefs sum to the joint probability (unnormalized)
// ---------------------------------------------------------------------------

func TestCalibration_BeliefsSumToJoint(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	// For a calibrated junction tree, the joint probability can be
	// recovered from any clique belief by marginalizing to a single
	// variable and summing. The sum of all entries in any clique belief
	// (when divided by the separator belief) gives the partition function.
	// Here, just verify that summing the entries of each clique belief
	// is consistent.

	b0 := bp.GetCliqueBelief(0)
	b1 := bp.GetCliqueBelief(1)

	// The sum of clique 0 = sum over A,B of phi(A,B)
	// After calibration this should equal Z (the partition function).
	d0 := b0.Values().Data()
	sum0 := 0.0
	for _, v := range d0 {
		sum0 += v
	}

	// Sum of separator B marginal.
	marg, _ := b0.Marginalize([]string{"A"})
	dSep := marg.Values().Data()
	sumSep := 0.0
	for _, v := range dSep {
		sumSep += v
	}

	// Z from clique 1 / separator should also equal Z from clique 0.
	d1 := b1.Values().Data()
	sum1 := 0.0
	for _, v := range d1 {
		sum1 += v
	}

	// Both cliques should have the same total mass (the partition function)
	// because the junction tree formula says:
	// P(all) = product(clique beliefs) / product(separator beliefs)
	// For a two-clique tree: Z = sum(clique0) and
	// sum(clique1) / sum(separator) should also equal Z.
	//
	// Actually, after Hugin calibration, each clique belief is the marginal
	// over the clique variables (times Z). The separator belief divides
	// them, so sum(clique_i) = Z for each clique.
	if !floatNear(sum0, sum1, 1e-9) {
		t.Errorf("partition function mismatch: clique0 sum=%f, clique1 sum=%f", sum0, sum1)
	}

	// Z should be 1.0 since original factors form a valid joint distribution.
	if !floatNear(sum0, 1.0, 1e-9) {
		t.Errorf("expected Z=1.0, got %f", sum0)
	}
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

func TestCalibrate_SingleVariable(t *testing.T) {
	pX, _ := factors.NewDiscreteFactor([]string{"X"}, []int{3}, []float64{0.2, 0.5, 0.3})
	bp := NewBeliefPropagation(
		[][]string{{"X"}},
		map[string][]string{},
		map[int][]*factors.DiscreteFactor{0: {pX}},
	)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	result, err := bp.Query([]string{"X"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"X": 0}), 0.2, 1e-9, "P(X=0)")
	assertNear(t, result.GetValue(map[string]int{"X": 1}), 0.5, 1e-9, "P(X=1)")
	assertNear(t, result.GetValue(map[string]int{"X": 2}), 0.3, 1e-9, "P(X=2)")
}

// ---------------------------------------------------------------------------
// Test numerical accuracy with extreme probabilities
// ---------------------------------------------------------------------------

func TestQuery_ExtremeProbs(t *testing.T) {
	// Very skewed probabilities.
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.999, 0.001})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{
		0.99, 0.01,
		0.01, 0.99,
	})

	allFactors := []*factors.DiscreteFactor{pA, pBA}
	bp := NewBeliefPropagation(
		[][]string{{"A", "B"}},
		map[string][]string{},
		map[int][]*factors.DiscreteFactor{0: {pA, pBA}},
	)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	ve := NewVariableElimination(allFactors)

	bpResult, _ := bp.Query([]string{"B"}, nil)
	veResult, _ := ve.Query([]string{"B"}, nil)

	for b := 0; b < 2; b++ {
		bpVal := bpResult.GetValue(map[string]int{"B": b})
		veVal := veResult.GetValue(map[string]int{"B": b})
		if math.Abs(bpVal-veVal) > 1e-9 {
			t.Errorf("extreme P(B=%d): BP=%f, VE=%f", b, bpVal, veVal)
		}
	}
}
