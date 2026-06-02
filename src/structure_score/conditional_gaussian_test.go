//go:build unit

package structure_score

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeMixedData creates a DataFrame with both discrete and continuous columns.
// D is discrete (0 or 1), X is continuous, Y = 3*X + 5*D + noise.
func makeMixedData() *tabgo.DataFrame {
	n := 200
	rng := rand.New(rand.NewSource(42))
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		d := rng.Intn(2)
		x := rng.NormFloat64()
		y := 3*x + 5*float64(d) + rng.NormFloat64()*0.1
		rows[i] = []any{d, x, y}
	}
	return tabgo.NewDataFrameFromRows([]string{"D", "X", "Y"}, rows)
}

// makePureContinuousMixedData creates a DataFrame with only continuous columns
// to test that the conditional Gaussian falls back to standard Gaussian.
func makePureContinuousMixedData() *tabgo.DataFrame {
	n := 100
	rng := rand.New(rand.NewSource(42))
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		y := 2*x + rng.NormFloat64()*0.5
		rows[i] = []any{x, y}
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y"}, rows)
}

// ============================================================
// BICCondGauss Tests
// ============================================================

func TestBICCondGauss_NewBICCondGauss(t *testing.T) {
	bg := NewBICCondGauss()
	if bg == nil {
		t.Fatal("NewBICCondGauss returned nil")
	}
}

func TestBICCondGauss_LocalScore_NoParents(t *testing.T) {
	data := makeMixedData()
	bg := NewBICCondGauss()

	score := bg.LocalScore("Y", nil, data)

	// Should be a finite negative score.
	if math.IsNaN(score) || math.IsInf(score, 0) {
		t.Errorf("BICCondGauss LocalScore(Y, nil) should be finite, got %f", score)
	}
}

func TestBICCondGauss_RewardsRelevantParents(t *testing.T) {
	data := makeMixedData()
	bg := NewBICCondGauss()

	// Y depends on both D and X.
	scoreNoParent := bg.LocalScore("Y", nil, data)
	scoreDiscreteOnly := bg.LocalScore("Y", []string{"D"}, data)
	scoreBothParents := bg.LocalScore("Y", []string{"D", "X"}, data)

	if scoreDiscreteOnly <= scoreNoParent {
		t.Errorf("BICCondGauss: D is a relevant parent: no_parent=%f, D_only=%f",
			scoreNoParent, scoreDiscreteOnly)
	}
	if scoreBothParents <= scoreDiscreteOnly {
		t.Errorf("BICCondGauss: X is also relevant: D_only=%f, both=%f",
			scoreDiscreteOnly, scoreBothParents)
	}
}

func TestBICCondGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeMixedData()
	bg := NewBICCondGauss()

	variables := []string{"D", "X", "Y"}
	parentMap := map[string][]string{
		"D": {},
		"X": {},
		"Y": {"D", "X"},
	}

	totalScore := bg.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += bg.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("BICCondGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestBICCondGauss_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	bg := NewBICCondGauss()
	score := bg.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("BICCondGauss LocalScore on empty data: got %f, want 0", score)
	}
}

// ============================================================
// AICCondGauss Tests
// ============================================================

func TestAICCondGauss_NewAICCondGauss(t *testing.T) {
	ag := NewAICCondGauss()
	if ag == nil {
		t.Fatal("NewAICCondGauss returned nil")
	}
}

func TestAICCondGauss_RewardsRelevantParents(t *testing.T) {
	data := makeMixedData()
	ag := NewAICCondGauss()

	scoreNoParent := ag.LocalScore("Y", nil, data)
	scoreBothParents := ag.LocalScore("Y", []string{"D", "X"}, data)

	if scoreBothParents <= scoreNoParent {
		t.Errorf("AICCondGauss: both parents should improve score: no_parent=%f, both=%f",
			scoreNoParent, scoreBothParents)
	}
}

func TestAICCondGauss_LessPenaltyThanBICCondGauss(t *testing.T) {
	data := makeMixedData()
	ag := NewAICCondGauss()
	bg := NewBICCondGauss()

	scoreAIC := ag.LocalScore("Y", []string{"D", "X"}, data)
	scoreBIC := bg.LocalScore("Y", []string{"D", "X"}, data)

	// AIC should have less penalty than BIC for N=200.
	if scoreAIC < scoreBIC {
		t.Errorf("AICCondGauss should have less penalty: AIC=%f, BIC=%f", scoreAIC, scoreBIC)
	}
}

func TestAICCondGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeMixedData()
	ag := NewAICCondGauss()

	variables := []string{"D", "X", "Y"}
	parentMap := map[string][]string{
		"D": {},
		"X": {},
		"Y": {"D", "X"},
	}

	totalScore := ag.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += ag.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("AICCondGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

// ============================================================
// LogLikelihoodCondGauss Tests
// ============================================================

func TestLogLikelihoodCondGauss_NewLogLikelihoodCondGauss(t *testing.T) {
	lg := NewLogLikelihoodCondGauss()
	if lg == nil {
		t.Fatal("NewLogLikelihoodCondGauss returned nil")
	}
}

func TestLogLikelihoodCondGauss_GreaterThanOrEqualBICCondGauss(t *testing.T) {
	data := makeMixedData()
	lg := NewLogLikelihoodCondGauss()
	bg := NewBICCondGauss()

	for _, parents := range [][]string{nil, {"D", "X"}} {
		scoreLL := lg.LocalScore("Y", parents, data)
		scoreBIC := bg.LocalScore("Y", parents, data)
		if scoreLL < scoreBIC {
			t.Errorf("LogLikelihoodCondGauss should be >= BICCondGauss: LL=%f, BIC=%f (parents=%v)",
				scoreLL, scoreBIC, parents)
		}
	}
}

func TestLogLikelihoodCondGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeMixedData()
	lg := NewLogLikelihoodCondGauss()

	variables := []string{"D", "X", "Y"}
	parentMap := map[string][]string{
		"D": {},
		"X": {},
		"Y": {"D", "X"},
	}

	totalScore := lg.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += lg.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("LogLikelihoodCondGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestLogLikelihoodCondGauss_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	lg := NewLogLikelihoodCondGauss()
	score := lg.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("LogLikelihoodCondGauss LocalScore on empty data: got %f, want 0", score)
	}
}

// ============================================================
// CondGauss Scorer Ordering
// ============================================================

func TestCondGaussScorers_ScoreOrdering(t *testing.T) {
	// LogLikelihoodCondGauss >= AICCondGauss >= BICCondGauss for large N.
	data := makeMixedData()

	lg := NewLogLikelihoodCondGauss()
	ag := NewAICCondGauss()
	bg := NewBICCondGauss()

	for _, parents := range [][]string{nil, {"D", "X"}} {
		sLL := lg.LocalScore("Y", parents, data)
		sAIC := ag.LocalScore("Y", parents, data)
		sBIC := bg.LocalScore("Y", parents, data)

		if sLL < sAIC {
			t.Errorf("LL should be >= AIC: LL=%f, AIC=%f (parents=%v)", sLL, sAIC, parents)
		}
		if sAIC < sBIC {
			t.Errorf("AIC should be >= BIC: AIC=%f, BIC=%f (parents=%v)", sAIC, sBIC, parents)
		}
	}
}

// ============================================================
// Interface compliance tests
// ============================================================

func TestCondGaussian_InterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewBICCondGauss()
	var _ StructureScore = NewAICCondGauss()
	var _ StructureScore = NewLogLikelihoodCondGauss()
}

// ============================================================
// Helper function tests
// ============================================================

func TestIsDiscreteColumn(t *testing.T) {
	// Integer column should be discrete.
	rows := [][]any{{1, 1.5}, {2, 2.7}, {3, 3.9}}
	data := tabgo.NewDataFrameFromRows([]string{"D", "C"}, rows)

	if !isDiscreteColumn(data, "D") {
		t.Error("integer column D should be detected as discrete")
	}
	if isDiscreteColumn(data, "C") {
		t.Error("float column C should be detected as continuous")
	}
}

func TestSplitParents(t *testing.T) {
	rows := [][]any{{1, 1.5, "a"}, {2, 2.7, "b"}}
	data := tabgo.NewDataFrameFromRows([]string{"D", "C", "S"}, rows)

	discrete, continuous := splitParents([]string{"D", "C", "S"}, data)

	// D (int) and S (string) are discrete, C (float) is continuous.
	if len(discrete) != 2 {
		t.Errorf("expected 2 discrete parents, got %d: %v", len(discrete), discrete)
	}
	if len(continuous) != 1 {
		t.Errorf("expected 1 continuous parent, got %d: %v", len(continuous), continuous)
	}
}

func TestConditionalGaussianLL_NoDiscreteParents(t *testing.T) {
	// With no discrete parents, should fall back to standard Gaussian LL.
	data := makePureContinuousMixedData()

	llCond, paramsCond := conditionalGaussianLL("Y", nil, []string{"X"}, data)
	llGauss, paramsGauss := gaussianLL("Y", []string{"X"}, data)

	if !approxEqual(llCond, llGauss, 1e-10) {
		t.Errorf("conditionalGaussianLL with no discrete parents should match gaussianLL: %f vs %f",
			llCond, llGauss)
	}
	if paramsCond != paramsGauss {
		t.Errorf("numParams should match: %d vs %d", paramsCond, paramsGauss)
	}
}

func TestConditionalGaussianLL_DiscreteOnly(t *testing.T) {
	data := makeMixedData()

	ll, numParams := conditionalGaussianLL("Y", []string{"D"}, nil, data)

	// Should be a finite value.
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("conditionalGaussianLL with discrete-only parents should be finite, got %f", ll)
	}
	// With 2 strata (D=0, D=1) and no continuous parents:
	// params per stratum = 0 + 2 = 2, total = 2*2 = 4
	if numParams != 4 {
		t.Errorf("numParams: got %d, want 4", numParams)
	}
	_ = ll
}

func TestConditionalGaussianLL_MixedParents(t *testing.T) {
	data := makeMixedData()

	ll, numParams := conditionalGaussianLL("Y", []string{"D"}, []string{"X"}, data)

	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("conditionalGaussianLL with mixed parents should be finite, got %f", ll)
	}
	// With 2 strata and 1 continuous parent:
	// params per stratum = 1 + 2 = 3, total = 2*3 = 6
	if numParams != 6 {
		t.Errorf("numParams: got %d, want 6", numParams)
	}
	_ = ll
}

// ============================================================
// Cross-scorer consistency tests
// ============================================================

func TestCondGaussScorers_PreferRelevantParent(t *testing.T) {
	data := makeMixedData()

	scorers := map[string]StructureScore{
		"BICCondGauss":           NewBICCondGauss(),
		"AICCondGauss":           NewAICCondGauss(),
		"LogLikelihoodCondGauss": NewLogLikelihoodCondGauss(),
	}

	for name, scorer := range scorers {
		noParent := scorer.LocalScore("Y", nil, data)
		withBothParents := scorer.LocalScore("Y", []string{"D", "X"}, data)

		if withBothParents <= noParent {
			t.Errorf("%s: should prefer relevant parents for Y: no_parent=%f, both_parents=%f",
				name, noParent, withBothParents)
		}
	}
}
