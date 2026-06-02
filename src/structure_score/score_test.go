//go:build unit

package structure_score

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeTestData creates a simple DataFrame for testing.
// X has values 0,1; Y has values 0,1; Z has values 0,1.
// Y depends on X (Y=X), Z is random.
func makeTestData() *tabgo.DataFrame {
	// 100 rows: X determines Y perfectly, Z is independent
	rows := make([][]any, 0, 100)
	for i := 0; i < 50; i++ {
		rows = append(rows, []any{0, 0, i % 2})
	}
	for i := 0; i < 50; i++ {
		rows = append(rows, []any{1, 1, i % 2})
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y", "Z"}, rows)
}

// makeSmallData creates a small dataset for verifying exact values.
func makeSmallData() *tabgo.DataFrame {
	rows := [][]any{
		{0, 0},
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1, 1},
	}
	return tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// ============================================================
// BIC Tests
// ============================================================

func TestBIC_NewBIC(t *testing.T) {
	bic := NewBIC()
	if bic == nil {
		t.Fatal("NewBIC returned nil")
	}
}

func TestBIC_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	bic := NewBIC()

	// A has values {0, 1}, counts: 0->3, 1->3, N=6
	score := bic.LocalScore("A", nil, data)

	// LL = 3*ln(3/6) + 3*ln(3/6) = 6*ln(0.5) = -4.1588...
	// k = (2-1)*1 = 1
	// penalty = 0.5 * 1 * ln(6) = 0.8959...
	// BIC = -4.1588 - 0.8959 = -5.0547...
	expectedLL := 6 * math.Log(0.5)
	expectedPenalty := 0.5 * 1 * math.Log(6)
	expected := expectedLL - expectedPenalty

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("BIC LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestBIC_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	bic := NewBIC()

	// B given A:
	// A=0: B counts: 0->2, 1->1, N_j=3; LL_j = 2*ln(2/3) + 1*ln(1/3)
	// A=1: B counts: 0->1, 1->2, N_j=3; LL_j = 1*ln(1/3) + 2*ln(2/3)
	score := bic.LocalScore("B", []string{"A"}, data)

	llJ0 := 2*math.Log(2.0/3.0) + 1*math.Log(1.0/3.0)
	llJ1 := 1*math.Log(1.0/3.0) + 2*math.Log(2.0/3.0)
	ll := llJ0 + llJ1
	// k = (2-1) * 2 = 2 (card=2, numParentConfigs=2)
	penalty := 0.5 * 2 * math.Log(6)
	expected := ll - penalty

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("BIC LocalScore(B, [A]): got %f, want %f", score, expected)
	}
}

func TestBIC_PenalizesComplexity(t *testing.T) {
	data := makeTestData()
	bic := NewBIC()

	// Z is independent of X. Adding X as parent of Z should not improve score.
	scoreNoParent := bic.LocalScore("Z", nil, data)
	scoreWithParent := bic.LocalScore("Z", []string{"X"}, data)

	// Since Z is uniformly distributed regardless of X, the extra parameters
	// should cause BIC to penalize the more complex model.
	if scoreWithParent > scoreNoParent {
		t.Errorf("BIC should penalize unnecessary parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestBIC_RewardsRelevantParent(t *testing.T) {
	data := makeTestData()
	bic := NewBIC()

	// Y = X, so X is a relevant parent of Y.
	scoreNoParent := bic.LocalScore("Y", nil, data)
	scoreWithParent := bic.LocalScore("Y", []string{"X"}, data)

	// With X as parent, LL should be much better (perfect prediction).
	if scoreWithParent <= scoreNoParent {
		t.Errorf("BIC should reward relevant parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestBIC_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	bic := NewBIC()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := bic.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += bic.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("BIC Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestBIC_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	bic := NewBIC()
	score := bic.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("BIC LocalScore on empty data: got %f, want 0", score)
	}
}

// ============================================================
// BDeu Tests
// ============================================================

func TestBDeu_NewBDeu(t *testing.T) {
	bdeu := NewBDeu(1.0)
	if bdeu == nil {
		t.Fatal("NewBDeu returned nil")
	}
	if bdeu.equivalentSampleSize != 1.0 {
		t.Errorf("ESS: got %f, want 1.0", bdeu.equivalentSampleSize)
	}
}

func TestBDeu_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	bdeu := NewBDeu(1.0)

	score := bdeu.LocalScore("A", nil, data)

	// A: states={0,1}, card=2, numParentConfigs=1
	// alpha_jk = 1.0 / (1*2) = 0.5
	// alpha_j = 1.0 / 1 = 1.0
	// N_j = 6, N_j0 = 3, N_j1 = 3
	// score = lgamma(1) - lgamma(1+6)
	//       + lgamma(0.5+3) - lgamma(0.5)
	//       + lgamma(0.5+3) - lgamma(0.5)
	expected := scigo.Gammaln(1.0) - scigo.Gammaln(7.0) +
		scigo.Gammaln(3.5) - scigo.Gammaln(0.5) +
		scigo.Gammaln(3.5) - scigo.Gammaln(0.5)

	if !approxEqual(score, expected, 1e-10) {
		t.Errorf("BDeu LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestBDeu_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	bdeu := NewBDeu(1.0)

	score := bdeu.LocalScore("B", []string{"A"}, data)

	// B given A:
	// card(B) = 2, numParentConfigs = 2 (A has 2 states)
	// alpha_jk = 1.0 / (2*2) = 0.25
	// alpha_j = 1.0 / 2 = 0.5
	//
	// A=0: N_j=3, B=0->2, B=1->1
	// A=1: N_j=3, B=0->1, B=1->2
	alphaJK := 0.25
	alphaJ := 0.5

	expected := 0.0
	// A=0
	expected += scigo.Gammaln(alphaJ) - scigo.Gammaln(alphaJ+3)
	expected += scigo.Gammaln(alphaJK+2) - scigo.Gammaln(alphaJK) // B=0
	expected += scigo.Gammaln(alphaJK+1) - scigo.Gammaln(alphaJK) // B=1
	// A=1
	expected += scigo.Gammaln(alphaJ) - scigo.Gammaln(alphaJ+3)
	expected += scigo.Gammaln(alphaJK+1) - scigo.Gammaln(alphaJK) // B=0
	expected += scigo.Gammaln(alphaJK+2) - scigo.Gammaln(alphaJK) // B=1

	if !approxEqual(score, expected, 1e-10) {
		t.Errorf("BDeu LocalScore(B, [A]): got %f, want %f", score, expected)
	}
}

func TestBDeu_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	bdeu := NewBDeu(1.0)

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := bdeu.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += bdeu.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("BDeu Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestBDeu_DifferentESS(t *testing.T) {
	data := makeSmallData()

	bdeu1 := NewBDeu(1.0)
	bdeu10 := NewBDeu(10.0)

	score1 := bdeu1.LocalScore("A", nil, data)
	score10 := bdeu10.LocalScore("A", nil, data)

	// Different ESS should produce different scores.
	if approxEqual(score1, score10, 1e-10) {
		t.Errorf("BDeu scores should differ with different ESS: ess=1 got %f, ess=10 got %f",
			score1, score10)
	}
}

// ============================================================
// K2 Tests
// ============================================================

func TestK2_NewK2(t *testing.T) {
	k2 := NewK2()
	if k2 == nil {
		t.Fatal("NewK2 returned nil")
	}
}

func TestK2_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	k2 := NewK2()

	score := k2.LocalScore("A", nil, data)

	// A: states={0,1}, card=2, numParentConfigs=1
	// N_j = 6, N_j0 = 3, N_j1 = 3
	// score = lgamma(2) - lgamma(6+2) + lgamma(3+1) + lgamma(3+1)
	expected := scigo.Gammaln(2) - scigo.Gammaln(8) +
		scigo.Gammaln(4) + scigo.Gammaln(4)

	if !approxEqual(score, expected, 1e-10) {
		t.Errorf("K2 LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestK2_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	k2 := NewK2()

	score := k2.LocalScore("B", []string{"A"}, data)

	// B given A:
	// card(B) = 2
	//
	// A=0: N_j=3, B=0->2, B=1->1
	//   lgamma(2) - lgamma(3+2) + lgamma(2+1) + lgamma(1+1)
	// A=1: N_j=3, B=0->1, B=1->2
	//   lgamma(2) - lgamma(3+2) + lgamma(1+1) + lgamma(2+1)
	expected := 0.0
	// A=0
	expected += scigo.Gammaln(2) - scigo.Gammaln(5) + scigo.Gammaln(3) + scigo.Gammaln(2)
	// A=1
	expected += scigo.Gammaln(2) - scigo.Gammaln(5) + scigo.Gammaln(2) + scigo.Gammaln(3)

	if !approxEqual(score, expected, 1e-10) {
		t.Errorf("K2 LocalScore(B, [A]): got %f, want %f", score, expected)
	}
}

func TestK2_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	k2 := NewK2()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := k2.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += k2.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("K2 Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

// ============================================================
// Interface compliance tests
// ============================================================

func TestInterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewBIC()
	var _ StructureScore = NewBDeu(1.0)
	var _ StructureScore = NewK2()
}

// ============================================================
// Cross-scorer consistency tests
// ============================================================

func TestAllScorers_PreferRelevantParent(t *testing.T) {
	data := makeTestData()

	scorers := map[string]StructureScore{
		"BIC":  NewBIC(),
		"BDeu": NewBDeu(1.0),
		"K2":   NewK2(),
	}

	for name, scorer := range scorers {
		// Y = X, so Y|X should score better than Y|{} for all scorers.
		noParent := scorer.LocalScore("Y", nil, data)
		withParent := scorer.LocalScore("Y", []string{"X"}, data)

		if withParent <= noParent {
			t.Errorf("%s: should prefer relevant parent for Y: no_parent=%f, with_parent=%f",
				name, noParent, withParent)
		}
	}
}

func TestAllScorers_NegativeScores(t *testing.T) {
	// Structure scores should generally be negative (log-based).
	data := makeTestData()
	scorers := map[string]StructureScore{
		"BIC":  NewBIC(),
		"BDeu": NewBDeu(1.0),
		"K2":   NewK2(),
	}

	for name, scorer := range scorers {
		score := scorer.LocalScore("X", nil, data)
		if score > 0 {
			t.Errorf("%s: expected negative score for X, got %f", name, score)
		}
	}
}

// ============================================================
// Helper function tests
// ============================================================

func TestCountTable(t *testing.T) {
	data := makeSmallData()
	counts, parentCounts, card, states := countTable("A", nil, data)

	if card != 2 {
		t.Errorf("card: got %d, want 2", card)
	}
	if len(states) != 2 {
		t.Errorf("states length: got %d, want 2", len(states))
	}
	// No parents: single config ""
	if parentCounts[""] != 6 {
		t.Errorf("parentCounts[\"\"]: got %d, want 6", parentCounts[""])
	}
	if counts[""][0] != 3 || counts[""][1] != 3 {
		t.Errorf("counts: got %v, want {0:3, 1:3}", counts[""])
	}
}

func TestCountTable_WithParents(t *testing.T) {
	data := makeSmallData()
	counts, parentCounts, card, _ := countTable("B", []string{"A"}, data)

	if card != 2 {
		t.Errorf("card: got %d, want 2", card)
	}
	// A=0: 3 rows; A=1: 3 rows
	if parentCounts["A=0"] != 3 {
		t.Errorf("parentCounts[A=0]: got %d, want 3", parentCounts["A=0"])
	}
	if parentCounts["A=1"] != 3 {
		t.Errorf("parentCounts[A=1]: got %d, want 3", parentCounts["A=1"])
	}
	// A=0: B=0->2, B=1->1
	if counts["A=0"][0] != 2 {
		t.Errorf("counts[A=0][0]: got %d, want 2", counts["A=0"][0])
	}
	if counts["A=0"][1] != 1 {
		t.Errorf("counts[A=0][1]: got %d, want 1", counts["A=0"][1])
	}
}

func TestNumParentConfigurations(t *testing.T) {
	data := makeSmallData()

	n := numParentConfigurations(nil, data)
	if n != 1 {
		t.Errorf("no parents: got %d, want 1", n)
	}

	n = numParentConfigurations([]string{"A"}, data)
	if n != 2 {
		t.Errorf("[A] parent: got %d, want 2", n)
	}
}

// ============================================================
// Multiple parents test
// ============================================================

func TestBIC_MultipleParents(t *testing.T) {
	// Create data where C depends on both A and B.
	rows := make([][]any, 0, 80)
	for i := 0; i < 20; i++ {
		rows = append(rows, []any{0, 0, 0})
	}
	for i := 0; i < 20; i++ {
		rows = append(rows, []any{0, 1, 1})
	}
	for i := 0; i < 20; i++ {
		rows = append(rows, []any{1, 0, 1})
	}
	for i := 0; i < 20; i++ {
		rows = append(rows, []any{1, 1, 0})
	}
	data := tabgo.NewDataFrameFromRows([]string{"A", "B", "C"}, rows)

	bic := NewBIC()

	// C depends on A XOR B (deterministic).
	scoreNone := bic.LocalScore("C", nil, data)
	scoreA := bic.LocalScore("C", []string{"A"}, data)
	scoreAB := bic.LocalScore("C", []string{"A", "B"}, data)

	// With both parents should be much better than none or just one.
	if scoreAB <= scoreA {
		t.Errorf("BIC: C|{A,B} (%f) should be better than C|{A} (%f)", scoreAB, scoreA)
	}
	if scoreAB <= scoreNone {
		t.Errorf("BIC: C|{A,B} (%f) should be better than C|{} (%f)", scoreAB, scoreNone)
	}
}
