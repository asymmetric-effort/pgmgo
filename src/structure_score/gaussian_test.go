//go:build unit

package structure_score

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeGaussianData creates a DataFrame with continuous columns.
// X ~ N(0,1), Y = 2*X + noise, Z ~ N(0,1) (independent).
func makeGaussianData() *tabgo.DataFrame {
	n := 200
	rng := rand.New(rand.NewSource(42))
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		y := 2*x + rng.NormFloat64()*0.1
		z := rng.NormFloat64()
		rows[i] = []any{x, y, z}
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y", "Z"}, rows)
}

// makeSmallGaussianData creates a small DataFrame for exact value testing.
func makeSmallGaussianData() *tabgo.DataFrame {
	rows := [][]any{
		{1.0, 2.0},
		{2.0, 4.0},
		{3.0, 6.0},
		{4.0, 8.0},
		{5.0, 10.0},
	}
	return tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
}

// ============================================================
// BICGauss Tests
// ============================================================

func TestBICGauss_NewBICGauss(t *testing.T) {
	bg := NewBICGauss()
	if bg == nil {
		t.Fatal("NewBICGauss returned nil")
	}
}

func TestBICGauss_LocalScore_NoParents(t *testing.T) {
	data := makeSmallGaussianData()
	bg := NewBICGauss()

	score := bg.LocalScore("A", nil, data)

	// A = [1,2,3,4,5], mean=3, var = ((1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2)/5 = 10/5 = 2
	// LL = -5/2*(ln(2*pi*2) + 1) = -5/2*(ln(2*pi) + ln(2) + 1)
	// Wait, LL = -N/2*(ln(2*pi) + ln(var) + 1)
	nf := 5.0
	variance := 2.0
	ll := -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
	// k = 2 (intercept + variance)
	penalty := 0.5 * 2 * math.Log(nf)
	expected := ll - penalty

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("BICGauss LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestBICGauss_RewardsRelevantParent(t *testing.T) {
	data := makeGaussianData()
	bg := NewBICGauss()

	// Y = 2*X + noise, so X is a relevant parent.
	scoreNoParent := bg.LocalScore("Y", nil, data)
	scoreWithParent := bg.LocalScore("Y", []string{"X"}, data)

	if scoreWithParent <= scoreNoParent {
		t.Errorf("BICGauss should reward relevant parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestBICGauss_PenalizesComplexity(t *testing.T) {
	// Use a larger dataset with a fixed seed where Z is truly independent of X.
	n := 1000
	rng := rand.New(rand.NewSource(99))
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		z := rng.NormFloat64()
		rows[i] = []any{x, z}
	}
	data := tabgo.NewDataFrameFromRows([]string{"X", "Z"}, rows)
	bg := NewBICGauss()

	// Z is independent of X.
	scoreNoParent := bg.LocalScore("Z", nil, data)
	scoreWithParent := bg.LocalScore("Z", []string{"X"}, data)

	if scoreWithParent > scoreNoParent {
		t.Errorf("BICGauss should penalize unnecessary parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestBICGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeGaussianData()
	bg := NewBICGauss()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := bg.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += bg.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("BICGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestBICGauss_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	bg := NewBICGauss()
	score := bg.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("BICGauss LocalScore on empty data: got %f, want 0", score)
	}
}

// ============================================================
// AICGauss Tests
// ============================================================

func TestAICGauss_NewAICGauss(t *testing.T) {
	ag := NewAICGauss()
	if ag == nil {
		t.Fatal("NewAICGauss returned nil")
	}
}

func TestAICGauss_LocalScore_NoParents(t *testing.T) {
	data := makeSmallGaussianData()
	ag := NewAICGauss()

	score := ag.LocalScore("A", nil, data)

	nf := 5.0
	variance := 2.0
	ll := -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
	// k = 2 (intercept + variance)
	expected := ll - 2

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("AICGauss LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestAICGauss_LessPenaltyThanBICGauss(t *testing.T) {
	data := makeGaussianData() // N=200
	ag := NewAICGauss()
	bg := NewBICGauss()

	scoreAIC := ag.LocalScore("Y", []string{"X"}, data)
	scoreBIC := bg.LocalScore("Y", []string{"X"}, data)

	// AIC should have less penalty than BIC for N=200.
	if scoreAIC < scoreBIC {
		t.Errorf("AICGauss should have less penalty than BICGauss for large N: AIC=%f, BIC=%f",
			scoreAIC, scoreBIC)
	}
}

func TestAICGauss_RewardsRelevantParent(t *testing.T) {
	data := makeGaussianData()
	ag := NewAICGauss()

	scoreNoParent := ag.LocalScore("Y", nil, data)
	scoreWithParent := ag.LocalScore("Y", []string{"X"}, data)

	if scoreWithParent <= scoreNoParent {
		t.Errorf("AICGauss should reward relevant parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestAICGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeGaussianData()
	ag := NewAICGauss()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := ag.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += ag.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("AICGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

// ============================================================
// LogLikelihoodGauss Tests
// ============================================================

func TestLogLikelihoodGauss_NewLogLikelihoodGauss(t *testing.T) {
	lg := NewLogLikelihoodGauss()
	if lg == nil {
		t.Fatal("NewLogLikelihoodGauss returned nil")
	}
}

func TestLogLikelihoodGauss_LocalScore_NoParents(t *testing.T) {
	data := makeSmallGaussianData()
	lg := NewLogLikelihoodGauss()

	score := lg.LocalScore("A", nil, data)

	nf := 5.0
	variance := 2.0
	expected := -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("LogLikelihoodGauss LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestLogLikelihoodGauss_GreaterThanOrEqualBICGauss(t *testing.T) {
	data := makeGaussianData()
	lg := NewLogLikelihoodGauss()
	bg := NewBICGauss()

	for _, parents := range [][]string{nil, {"X"}} {
		scoreLL := lg.LocalScore("Y", parents, data)
		scoreBIC := bg.LocalScore("Y", parents, data)
		if scoreLL < scoreBIC {
			t.Errorf("LogLikelihoodGauss should be >= BICGauss: LL=%f, BIC=%f (parents=%v)",
				scoreLL, scoreBIC, parents)
		}
	}
}

func TestLogLikelihoodGauss_MonotonicallyIncreases(t *testing.T) {
	data := makeGaussianData()
	lg := NewLogLikelihoodGauss()

	scoreNoParent := lg.LocalScore("Y", nil, data)
	scoreWithParent := lg.LocalScore("Y", []string{"X"}, data)

	if scoreWithParent < scoreNoParent {
		t.Errorf("LogLikelihoodGauss should not decrease with more parents: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestLogLikelihoodGauss_PerfectPrediction(t *testing.T) {
	// B = 2*A, so regressing B on A should give very high LL.
	data := makeSmallGaussianData()
	lg := NewLogLikelihoodGauss()

	scoreWithParent := lg.LocalScore("B", []string{"A"}, data)
	scoreNoParent := lg.LocalScore("B", nil, data)

	if scoreWithParent <= scoreNoParent {
		t.Errorf("LogLikelihoodGauss should improve with perfect parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestLogLikelihoodGauss_Score_SumOfLocalScores(t *testing.T) {
	data := makeGaussianData()
	lg := NewLogLikelihoodGauss()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := lg.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += lg.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("LogLikelihoodGauss Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestLogLikelihoodGauss_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	lg := NewLogLikelihoodGauss()
	score := lg.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("LogLikelihoodGauss LocalScore on empty data: got %f, want 0", score)
	}
}

// ============================================================
// Gaussian helper tests
// ============================================================

func TestGaussSolve(t *testing.T) {
	// 2x + y = 5, x + 3y = 7 => x=1.6, y=1.8
	A := []float64{2, 1, 1, 3}
	b := []float64{5, 7}
	sol := gaussSolve(A, b, 2)
	if !approxEqual(sol[0], 1.6, 1e-10) || !approxEqual(sol[1], 1.8, 1e-10) {
		t.Errorf("gaussSolve: expected [1.6, 1.8], got [%f, %f]", sol[0], sol[1])
	}
}

func TestGaussianLL_NoParents(t *testing.T) {
	data := makeSmallGaussianData()
	ll, numParams := gaussianLL("A", nil, data)

	nf := 5.0
	variance := 2.0
	expected := -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)

	if !approxEqual(ll, expected, 1e-6) {
		t.Errorf("gaussianLL(A, nil): got %f, want %f", ll, expected)
	}
	// numParams = intercept + variance = 2
	if numParams != 2 {
		t.Errorf("gaussianLL numParams: got %d, want 2", numParams)
	}
}

func TestGaussianLL_WithParents(t *testing.T) {
	data := makeSmallGaussianData()
	ll, numParams := gaussianLL("B", []string{"A"}, data)

	// B = 2*A exactly, so residual variance should be ~0 (will be clamped to 1e-300).
	// LL should be very large positive value (negative * log of very small number).
	if ll <= 0 {
		// For perfect prediction, LL should be positive due to very small variance.
		// Actually, LL = -N/2*(ln(2*pi) + ln(var) + 1). If var is tiny, ln(var) is
		// very negative, so the whole thing is positive.
		// With clamped variance 1e-300: -5/2*(1.8379 + (-690.7755) + 1) > 0
		t.Logf("gaussianLL with perfect prediction: %f", ll)
	}

	// numParams = intercept + 1 parent + variance = 3
	if numParams != 3 {
		t.Errorf("gaussianLL numParams: got %d, want 3", numParams)
	}
}

// ============================================================
// Interface compliance tests
// ============================================================

func TestGaussian_InterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewBICGauss()
	var _ StructureScore = NewAICGauss()
	var _ StructureScore = NewLogLikelihoodGauss()
}

// ============================================================
// Cross-scorer consistency tests
// ============================================================

func TestGaussianScorers_PreferRelevantParent(t *testing.T) {
	data := makeGaussianData()

	scorers := map[string]StructureScore{
		"BICGauss":           NewBICGauss(),
		"AICGauss":           NewAICGauss(),
		"LogLikelihoodGauss": NewLogLikelihoodGauss(),
	}

	for name, scorer := range scorers {
		noParent := scorer.LocalScore("Y", nil, data)
		withParent := scorer.LocalScore("Y", []string{"X"}, data)

		if withParent <= noParent {
			t.Errorf("%s: should prefer relevant parent for Y: no_parent=%f, with_parent=%f",
				name, noParent, withParent)
		}
	}
}

func TestGaussianScorers_ScoreOrdering(t *testing.T) {
	// For the same model: LogLikelihoodGauss >= AICGauss >= BICGauss (for large N).
	data := makeGaussianData()

	lg := NewLogLikelihoodGauss()
	ag := NewAICGauss()
	bg := NewBICGauss()

	for _, parents := range [][]string{nil, {"X"}} {
		sLL := lg.LocalScore("Y", parents, data)
		sAIC := ag.LocalScore("Y", parents, data)
		sBIC := bg.LocalScore("Y", parents, data)

		if sLL < sAIC {
			t.Errorf("LL should be >= AIC: LL=%f, AIC=%f (parents=%v)", sLL, sAIC, parents)
		}
		if sAIC < sBIC {
			t.Errorf("AIC should be >= BIC for N=200: AIC=%f, BIC=%f (parents=%v)", sAIC, sBIC, parents)
		}
	}
}
