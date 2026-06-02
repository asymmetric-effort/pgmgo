//go:build unit

package inference

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ---------------------------------------------------------------------------
// Student BayesianNetwork builder
// ---------------------------------------------------------------------------
//
// Classic student Bayesian network (Koller & Friedman):
//
//	Difficulty(D) -> Grade(G) <- Intelligence(I)
//	                 Grade(G) -> Letter(L)
//	                 Intelligence(I) -> SAT(S)
//
// Variable cardinalities:
//
//	D: 2 (d0=easy, d1=hard)
//	I: 2 (i0=low, i1=high)
//	G: 3 (g1, g2, g3)
//	L: 2 (l0=weak, l1=strong)
//	S: 2 (s0=low, s1=high)
func buildStudentBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()

	for _, n := range []string{"D", "G", "I", "L", "S"} {
		if err := bn.AddNode(n); err != nil {
			t.Fatalf("AddNode(%s): %v", n, err)
		}
	}
	for _, e := range [][2]string{{"D", "G"}, {"I", "G"}, {"G", "L"}, {"I", "S"}} {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			t.Fatalf("AddEdge(%s->%s): %v", e[0], e[1], err)
		}
	}

	// P(D)
	cpdD, err := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6}, {0.4}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// P(I)
	cpdI, err := factors.NewTabularCPD("I", 2,
		[][]float64{{0.7}, {0.3}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// P(G | D, I) — rows=G states, cols=(D,I) configs in row-major
	cpdG, err := factors.NewTabularCPD("G", 3,
		[][]float64{
			{0.3, 0.9, 0.05, 0.5},
			{0.4, 0.08, 0.25, 0.3},
			{0.3, 0.02, 0.7, 0.2},
		},
		[]string{"D", "I"}, []int{2, 2})
	if err != nil {
		t.Fatal(err)
	}

	// P(L | G)
	cpdL, err := factors.NewTabularCPD("L", 2,
		[][]float64{
			{0.1, 0.4, 0.99},
			{0.9, 0.6, 0.01},
		},
		[]string{"G"}, []int{3})
	if err != nil {
		t.Fatal(err)
	}

	// P(S | I)
	cpdS, err := factors.NewTabularCPD("S", 2,
		[][]float64{
			{0.95, 0.2},
			{0.05, 0.8},
		},
		[]string{"I"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}

	for _, cpd := range []*factors.TabularCPD{cpdD, cpdI, cpdG, cpdL, cpdS} {
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatal(err)
		}
	}

	return bn
}

// ---------------------------------------------------------------------------
// NewCausalInference
// ---------------------------------------------------------------------------

func TestNewCausalInference_NilBN(t *testing.T) {
	_, err := NewCausalInference(nil)
	if err == nil {
		t.Fatal("expected error for nil BN")
	}
}

func TestNewCausalInference_Valid(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ci == nil {
		t.Fatal("expected non-nil CausalInference")
	}
}

// ---------------------------------------------------------------------------
// Query — observational (no do)
// ---------------------------------------------------------------------------

func TestQuery_Observational_MarginalD(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// With no do-variables, should match standard VE.
	result, err := ci.Query([]string{"D"}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"D": 0}), 0.6, 1e-6, "P(D=0)")
	assertNear(t, result.GetValue(map[string]int{"D": 1}), 0.4, 1e-6, "P(D=1)")
}

func TestQuery_Observational_GradeGivenEvidence(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(G | D=1, I=1) — conditioning, no intervention.
	result, err := ci.Query([]string{"G"}, nil, map[string]int{"D": 1, "I": 1})
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	// Direct CPD values for D=1, I=1: G=0:0.5, G=1:0.3, G=2:0.2
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.5, 1e-6, "P(G=0|D=1,I=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.3, 1e-6, "P(G=1|D=1,I=1)")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.2, 1e-6, "P(G=2|D=1,I=1)")
}

// ---------------------------------------------------------------------------
// Query — interventional (do-calculus)
// ---------------------------------------------------------------------------

func TestQuery_DoIntervention_DoDifficulty(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(G | do(D=1)): Since D has no parents, do(D=1) = observing D=1.
	// P(G | do(D=1)) = sum_I P(G|D=1,I)*P(I)
	// P(G=0|do(D=1)) = 0.05*0.7 + 0.5*0.3 = 0.035 + 0.15 = 0.185
	// P(G=1|do(D=1)) = 0.25*0.7 + 0.3*0.3 = 0.175 + 0.09 = 0.265
	// P(G=2|do(D=1)) = 0.7*0.7 + 0.2*0.3  = 0.49 + 0.06  = 0.55
	result, err := ci.Query([]string{"G"}, map[string]int{"D": 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.185, 1e-6, "P(G=0|do(D=1))")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.265, 1e-6, "P(G=1|do(D=1))")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.55, 1e-6, "P(G=2|do(D=1))")
}

func TestQuery_DoIntervention_DoGrade(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(L | do(G=0)): Intervening on G severs D->G and I->G links.
	// P(L|do(G=0)) = P(L|G=0) = [0.1, 0.9]
	result, err := ci.Query([]string{"L"}, map[string]int{"G": 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"L": 0}), 0.1, 1e-6, "P(L=0|do(G=0))")
	assertNear(t, result.GetValue(map[string]int{"L": 1}), 0.9, 1e-6, "P(L=1|do(G=0))")
}

func TestQuery_DoVsObservational_DifferentResults(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// Observational: P(I | G=0) — observing G=0 gives info about I through
	// the common cause structure. Intelligence is not independent of Grade.
	obsResult, err := ci.Query([]string{"I"}, nil, map[string]int{"G": 0})
	if err != nil {
		t.Fatal(err)
	}

	// Interventional: P(I | do(G=0)) — intervening on G severs the D->G and
	// I->G edges, so I becomes independent of G. P(I|do(G=0)) = P(I).
	doResult, err := ci.Query([]string{"I"}, map[string]int{"G": 0}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// The interventional result should equal the prior on I.
	assertNear(t, doResult.GetValue(map[string]int{"I": 0}), 0.7, 1e-6, "P(I=0|do(G=0))")
	assertNear(t, doResult.GetValue(map[string]int{"I": 1}), 0.3, 1e-6, "P(I=1|do(G=0))")

	// The observational result should differ from the prior.
	obsI0 := obsResult.GetValue(map[string]int{"I": 0})
	if floatNear(obsI0, 0.7, 1e-3) {
		t.Errorf("P(I=0|G=0) should differ from prior 0.7, got %f", obsI0)
	}
}

func TestQuery_DoIntervention_WithEvidence(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(L | do(D=0), I=1): Intervene on D, observe I.
	// In the mutilated graph, D is set to 0. The remaining model uses P(I).
	// But we also condition on I=1.
	// P(G|D=0,I=1): G=0:0.9, G=1:0.08, G=2:0.02
	// P(L=0|do(D=0),I=1) = P(L=0|G=0)*P(G=0|D=0,I=1) + P(L=0|G=1)*P(G=1|D=0,I=1) + P(L=0|G=2)*P(G=2|D=0,I=1)
	//                     = 0.1*0.9 + 0.4*0.08 + 0.99*0.02
	//                     = 0.09 + 0.032 + 0.0198 = 0.1418
	result, err := ci.Query([]string{"L"}, map[string]int{"D": 0}, map[string]int{"I": 1})
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"L": 0}), 0.1418, 1e-4, "P(L=0|do(D=0),I=1)")
}

func TestQuery_DoIntervention_DoSAT(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(D | do(S=1)): Intervening on SAT severs I->S. D and S are
	// d-separated in the original graph (and certainly after mutilation).
	// So P(D | do(S=1)) = P(D) = [0.6, 0.4].
	result, err := ci.Query([]string{"D"}, map[string]int{"S": 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"D": 0}), 0.6, 1e-6, "P(D=0|do(S=1))")
	assertNear(t, result.GetValue(map[string]int{"D": 1}), 0.4, 1e-6, "P(D=1|do(S=1))")
}

// ---------------------------------------------------------------------------
// Query — error cases
// ---------------------------------------------------------------------------

func TestCausalQuery_EmptyQueryVars(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.Query(nil, nil, nil)
	if err == nil {
		t.Fatal("expected error for empty queryVars")
	}
}

func TestCausalQuery_InvalidDoValue(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.Query([]string{"G"}, map[string]int{"D": 5}, nil)
	if err == nil {
		t.Fatal("expected error for out-of-range do-value")
	}
}

// ---------------------------------------------------------------------------
// ATE
// ---------------------------------------------------------------------------

func TestATE_DifficultyOnGrade(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// ATE of D on G: E[G|do(D=1)] - E[G|do(D=0)]
	// Since D is a root node, do(D=v) = condition on D=v.
	//
	// E[G|do(D=0)] = sum_I sum_g g * P(G=g|D=0,I) * P(I)
	//   P(G=0|do(D=0)) = 0.3*0.7 + 0.9*0.3 = 0.21 + 0.27 = 0.48
	//   P(G=1|do(D=0)) = 0.4*0.7 + 0.08*0.3 = 0.28 + 0.024 = 0.304
	//   P(G=2|do(D=0)) = 0.3*0.7 + 0.02*0.3 = 0.21 + 0.006 = 0.216
	//   E[G|do(D=0)] = 0*0.48 + 1*0.304 + 2*0.216 = 0.736
	//
	// E[G|do(D=1)] = sum_I sum_g g * P(G=g|D=1,I) * P(I)
	//   P(G=0|do(D=1)) = 0.05*0.7 + 0.5*0.3 = 0.035 + 0.15 = 0.185
	//   P(G=1|do(D=1)) = 0.25*0.7 + 0.3*0.3 = 0.175 + 0.09 = 0.265
	//   P(G=2|do(D=1)) = 0.7*0.7 + 0.2*0.3  = 0.49 + 0.06  = 0.55
	//   E[G|do(D=1)] = 0*0.185 + 1*0.265 + 2*0.55 = 1.365
	//
	// ATE = 1.365 - 0.736 = 0.629
	ate, err := ci.ATE("D", "G", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	assertNear(t, ate, 0.629, 1e-4, "ATE(D->G)")
}

func TestATE_IntelligenceOnSAT(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// ATE of I on S: E[S|do(I=1)] - E[S|do(I=0)]
	// Since I is a root, do(I=v) = condition on I=v.
	// E[S|do(I=0)] = 0*0.95 + 1*0.05 = 0.05
	// E[S|do(I=1)] = 0*0.2 + 1*0.8 = 0.8
	// ATE = 0.8 - 0.05 = 0.75
	ate, err := ci.ATE("I", "S", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	assertNear(t, ate, 0.75, 1e-6, "ATE(I->S)")
}

func TestATE_DifficultyOnLetter(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// ATE of D on L (indirect through G):
	// P(L=1|do(D=0)) = sum_G P(L=1|G)*P(G|do(D=0))
	//   = 0.9*0.48 + 0.6*0.304 + 0.01*0.216
	//   = 0.432 + 0.1824 + 0.00216 = 0.61656
	// P(L=1|do(D=1)) = sum_G P(L=1|G)*P(G|do(D=1))
	//   = 0.9*0.185 + 0.6*0.265 + 0.01*0.55
	//   = 0.1665 + 0.159 + 0.0055 = 0.331
	// E[L|do(D=0)] = 0*P(L=0|do(D=0)) + 1*P(L=1|do(D=0)) = 0.61656
	// E[L|do(D=1)] = 0.331
	// ATE = 0.331 - 0.61656 = -0.28556
	ate, err := ci.ATE("D", "L", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	assertNear(t, ate, -0.28556, 1e-3, "ATE(D->L)")
}

// ---------------------------------------------------------------------------
// IsValidBackdoor
// ---------------------------------------------------------------------------

func TestIsValidBackdoor_ValidSet(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// D -> G <- I; G -> L; I -> S
	// Causal effect of D on G: empty set is valid backdoor since D is a root
	// (no backdoor paths exist from D to G that don't go through D->G).
	if !ci.IsValidBackdoor("D", "G", nil) {
		t.Error("empty set should be valid backdoor for D->G")
	}
}

func TestIsValidBackdoor_WithConfounders(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// Causal effect of D on L: the path is D->G->L.
	// Backdoor paths from D to L: D<-...-?->L. Since D is a root, no
	// backdoor paths exist. Empty set is valid.
	if !ci.IsValidBackdoor("D", "L", nil) {
		t.Error("empty set should be valid backdoor for D->L")
	}

	// I is not a descendant of D, and conditioning on I is valid.
	if !ci.IsValidBackdoor("D", "L", []string{"I"}) {
		t.Error("{I} should be valid backdoor for D->L")
	}
}

func TestIsValidBackdoor_InvalidSet_DescendantOfTreatment(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// G is a descendant of D. Adjusting for G violates criterion 1.
	if ci.IsValidBackdoor("D", "L", []string{"G"}) {
		t.Error("{G} should NOT be valid backdoor for D->L (G is descendant of D)")
	}
}

func TestIsValidBackdoor_InvalidSet_DescendantOfTreatment_Letter(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// L is a descendant of D (through G). Should fail criterion 1.
	if ci.IsValidBackdoor("D", "S", []string{"L"}) {
		t.Error("{L} should NOT be valid backdoor for D->S (L is descendant of D)")
	}
}

func TestIsValidBackdoor_IntelligenceOnGrade(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// Causal effect of I on G. D is not a descendant of I.
	// Backdoor paths: I <- ... -> G. The only backdoor path would need to
	// go through a common cause of I and G. D is a parent of G, not of I.
	// So no confounding. Empty set is valid.
	if !ci.IsValidBackdoor("I", "G", nil) {
		t.Error("empty set should be valid backdoor for I->G")
	}

	// D is not a descendant of I, so {D} is also valid.
	if !ci.IsValidBackdoor("I", "G", []string{"D"}) {
		t.Error("{D} should be valid backdoor for I->G")
	}
}

// ---------------------------------------------------------------------------
// BackdoorAdjustment
// ---------------------------------------------------------------------------

func TestBackdoorAdjustment_SimpleData(t *testing.T) {
	// Build a simple BN: D -> Y
	// Use a small dataset where the ATE can be computed manually.
	// D=0: outcome values  [1, 1, 0, 1, 0]   -> mean = 0.6
	// D=1: outcome values  [0, 0, 1, 0, 0]   -> mean = 0.2
	// Since D is a root node, empty adjustment set is valid, and
	// backdoor adjustment = E[Y|T=1] - E[Y|T=0] = 0.2 - 0.6 = -0.4
	simpleBN := models.NewBayesianNetwork()
	if err := simpleBN.AddNode("D"); err != nil {
		t.Fatal(err)
	}
	if err := simpleBN.AddNode("Y"); err != nil {
		t.Fatal(err)
	}
	if err := simpleBN.AddEdge("D", "Y"); err != nil {
		t.Fatal(err)
	}
	cpdD, err := factors.NewTabularCPD("D", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cpdY, err := factors.NewTabularCPD("Y", 2,
		[][]float64{
			{0.4, 0.8}, // P(Y=0|D=0)=0.4, P(Y=0|D=1)=0.8
			{0.6, 0.2}, // P(Y=1|D=0)=0.6, P(Y=1|D=1)=0.2
		},
		[]string{"D"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if err := simpleBN.AddCPD(cpdD); err != nil {
		t.Fatal(err)
	}
	if err := simpleBN.AddCPD(cpdY); err != nil {
		t.Fatal(err)
	}

	simpleCi, err := NewCausalInference(simpleBN)
	if err != nil {
		t.Fatal(err)
	}

	// Data: 5 rows with D=0, 5 rows with D=1
	data := tabgo.NewDataFrameFromRows(
		[]string{"D", "Y"},
		[][]any{
			{0, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 0},
			{1, 0}, {1, 0}, {1, 1}, {1, 0}, {1, 0},
		},
	)

	ate, err := simpleCi.BackdoorAdjustment("D", "Y", nil, data)
	if err != nil {
		t.Fatal(err)
	}
	// E[Y|D=1] - E[Y|D=0] = 0.2 - 0.6 = -0.4
	assertNear(t, ate, -0.4, 1e-6, "BackdoorAdjustment ATE")
}

func TestBackdoorAdjustment_WithAdjustmentSet(t *testing.T) {
	// Build a confounded model: Z -> D, Z -> Y, D -> Y
	// Without adjusting for Z, the observational estimate is biased.
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"Z", "D", "Y"} {
		if err := bn.AddNode(n); err != nil {
			t.Fatal(err)
		}
	}
	for _, e := range [][2]string{{"Z", "D"}, {"Z", "Y"}, {"D", "Y"}} {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			t.Fatal(err)
		}
	}

	cpdZ, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	cpdD, _ := factors.NewTabularCPD("D", 2,
		[][]float64{{0.8, 0.2}, {0.2, 0.8}},
		[]string{"Z"}, []int{2})
	cpdY, _ := factors.NewTabularCPD("Y", 2,
		[][]float64{
			{0.9, 0.6, 0.7, 0.3}, // P(Y=0|D,Z)
			{0.1, 0.4, 0.3, 0.7}, // P(Y=1|D,Z)
		},
		[]string{"D", "Z"}, []int{2, 2})

	for _, cpd := range []*factors.TabularCPD{cpdZ, cpdD, cpdY} {
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatal(err)
		}
	}

	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// {Z} is a valid backdoor for D->Y
	if !ci.IsValidBackdoor("D", "Y", []string{"Z"}) {
		t.Fatal("Z should be valid backdoor for D->Y")
	}

	// Build data with 1000 rows sampling from the model distributions
	// Z=0: D=0 (80%), D=1 (20%)
	// Z=1: D=0 (20%), D=1 (80%)
	// For simplicity, use a deterministic dataset:
	// Z=0, D=0: 40 rows, Y outcomes: 90% are 0, 10% are 1 => 36x Y=0, 4x Y=1
	// Z=0, D=1: 10 rows, Y outcomes: 60% are 0, 40% are 1 => 6x Y=0, 4x Y=1
	// Z=1, D=0: 10 rows, Y outcomes: 70% are 0, 30% are 1 => 7x Y=0, 3x Y=1
	// Z=1, D=1: 40 rows, Y outcomes: 30% are 0, 70% are 1 => 12x Y=0, 28x Y=1
	var rows [][]any
	for i := 0; i < 36; i++ {
		rows = append(rows, []any{0, 0, 0})
	}
	for i := 0; i < 4; i++ {
		rows = append(rows, []any{0, 0, 1})
	}
	for i := 0; i < 6; i++ {
		rows = append(rows, []any{0, 1, 0})
	}
	for i := 0; i < 4; i++ {
		rows = append(rows, []any{0, 1, 1})
	}
	for i := 0; i < 7; i++ {
		rows = append(rows, []any{1, 0, 0})
	}
	for i := 0; i < 3; i++ {
		rows = append(rows, []any{1, 0, 1})
	}
	for i := 0; i < 12; i++ {
		rows = append(rows, []any{1, 1, 0})
	}
	for i := 0; i < 28; i++ {
		rows = append(rows, []any{1, 1, 1})
	}

	data := tabgo.NewDataFrameFromRows([]string{"Z", "D", "Y"}, rows)

	ate, err := ci.BackdoorAdjustment("D", "Y", []string{"Z"}, data)
	if err != nil {
		t.Fatal(err)
	}

	// Expected ATE using backdoor:
	// E[Y|do(D=1)] = sum_z E[Y|D=1,Z=z]*P(Z=z)
	// P(Z=0) = 50/100 = 0.5,  P(Z=1) = 50/100 = 0.5
	// E[Y|D=1,Z=0] = 4/10 = 0.4
	// E[Y|D=1,Z=1] = 28/40 = 0.7
	// E[Y|do(D=1)] = 0.4*0.5 + 0.7*0.5 = 0.55
	//
	// E[Y|do(D=0)] = sum_z E[Y|D=0,Z=z]*P(Z=z)
	// E[Y|D=0,Z=0] = 4/40 = 0.1
	// E[Y|D=0,Z=1] = 3/10 = 0.3
	// E[Y|do(D=0)] = 0.1*0.5 + 0.3*0.5 = 0.2
	//
	// ATE = 0.55 - 0.2 = 0.35
	assertNear(t, ate, 0.35, 1e-6, "BackdoorAdjustment with Z")
}

func TestBackdoorAdjustment_NilData(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.BackdoorAdjustment("D", "G", nil, nil)
	if err == nil {
		t.Fatal("expected error for nil data")
	}
}

func TestBackdoorAdjustment_InvalidBackdoorSet(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	data := tabgo.NewDataFrameFromRows(
		[]string{"D", "G", "L"},
		[][]any{{0, 1, 0}},
	)

	// G is a descendant of D, so {G} is not a valid backdoor.
	_, err = ci.BackdoorAdjustment("D", "L", []string{"G"}, data)
	if err == nil {
		t.Fatal("expected error for invalid backdoor set")
	}
}

// ---------------------------------------------------------------------------
// Verify interventional != observational for confounded variables
// ---------------------------------------------------------------------------

func TestDoVsObserve_GradeAndIntelligence(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(D | G=0) [observational] vs P(D | do(G=0)) [interventional]
	// Observational: observing a good grade tells us something about difficulty.
	// Interventional: forcing the grade tells us nothing about difficulty.
	obsResult, err := ci.Query([]string{"D"}, nil, map[string]int{"G": 0})
	if err != nil {
		t.Fatal(err)
	}

	doResult, err := ci.Query([]string{"D"}, map[string]int{"G": 0}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// do(G=0) should give P(D) = [0.6, 0.4] (prior, since D is independent
	// of G once G's parents are severed).
	assertNear(t, doResult.GetValue(map[string]int{"D": 0}), 0.6, 1e-6, "P(D=0|do(G=0))")

	// Observational P(D=0|G=0) should NOT be 0.6.
	obsD0 := obsResult.GetValue(map[string]int{"D": 0})
	if math.Abs(obsD0-0.6) < 1e-3 {
		t.Errorf("P(D=0|G=0) should differ from prior 0.6, got %f", obsD0)
	}
}

// ---------------------------------------------------------------------------
// Edge case: multiple do-variables
// ---------------------------------------------------------------------------

func TestQuery_MultipleDoVariables(t *testing.T) {
	bn := buildStudentBN(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	// P(G | do(D=1), do(I=0)): Both parents of G are intervened on.
	// This should give P(G | D=1, I=0) = [0.05, 0.25, 0.7] directly
	// from the CPD.
	result, err := ci.Query([]string{"G"}, map[string]int{"D": 1, "I": 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.05, 1e-6, "P(G=0|do(D=1),do(I=0))")
	assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.25, 1e-6, "P(G=1|do(D=1),do(I=0))")
	assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.70, 1e-6, "P(G=2|do(D=1),do(I=0))")
}
