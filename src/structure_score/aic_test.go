//go:build unit

package structure_score

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestAIC_NewAIC(t *testing.T) {
	aic := NewAIC()
	if aic == nil {
		t.Fatal("NewAIC returned nil")
	}
}

func TestAIC_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	aic := NewAIC()

	score := aic.LocalScore("A", nil, data)

	// LL = 3*ln(3/6) + 3*ln(3/6) = 6*ln(0.5)
	// k = (2-1)*1 = 1
	// AIC = LL - k = 6*ln(0.5) - 1
	expectedLL := 6 * math.Log(0.5)
	expected := expectedLL - 1

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("AIC LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestAIC_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	aic := NewAIC()

	score := aic.LocalScore("B", []string{"A"}, data)

	llJ0 := 2*math.Log(2.0/3.0) + 1*math.Log(1.0/3.0)
	llJ1 := 1*math.Log(1.0/3.0) + 2*math.Log(2.0/3.0)
	ll := llJ0 + llJ1
	// k = (2-1) * 2 = 2
	expected := ll - 2

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("AIC LocalScore(B, [A]): got %f, want %f", score, expected)
	}
}

func TestAIC_LessPenaltyThanBIC(t *testing.T) {
	// For large N, BIC penalizes more than AIC.
	// BIC penalty = 0.5*k*ln(N), AIC penalty = k.
	// For N > e^2 (approximately 7.389), BIC penalizes more.
	data := makeTestData() // N=100
	aic := NewAIC()
	bic := NewBIC()

	scoreAIC := aic.LocalScore("Y", []string{"X"}, data)
	scoreBIC := bic.LocalScore("Y", []string{"X"}, data)

	// AIC should be >= BIC since AIC has smaller penalty for N=100.
	if scoreAIC < scoreBIC {
		t.Errorf("AIC should have less penalty than BIC for N=100: AIC=%f, BIC=%f", scoreAIC, scoreBIC)
	}
}

func TestAIC_PenalizesComplexity(t *testing.T) {
	data := makeTestData()
	aic := NewAIC()

	scoreNoParent := aic.LocalScore("Z", nil, data)
	scoreWithParent := aic.LocalScore("Z", []string{"X"}, data)

	if scoreWithParent > scoreNoParent {
		t.Errorf("AIC should penalize unnecessary parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestAIC_RewardsRelevantParent(t *testing.T) {
	data := makeTestData()
	aic := NewAIC()

	scoreNoParent := aic.LocalScore("Y", nil, data)
	scoreWithParent := aic.LocalScore("Y", []string{"X"}, data)

	if scoreWithParent <= scoreNoParent {
		t.Errorf("AIC should reward relevant parent: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestAIC_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	aic := NewAIC()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := aic.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += aic.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("AIC Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestAIC_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	aic := NewAIC()
	score := aic.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("AIC LocalScore on empty data: got %f, want 0", score)
	}
}

func TestAIC_InterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewAIC()
}
