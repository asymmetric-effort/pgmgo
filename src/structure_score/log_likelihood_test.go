//go:build unit

package structure_score

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestLogLikelihood_NewLogLikelihood(t *testing.T) {
	ll := NewLogLikelihood()
	if ll == nil {
		t.Fatal("NewLogLikelihood returned nil")
	}
}

func TestLogLikelihood_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	ll := NewLogLikelihood()

	score := ll.LocalScore("A", nil, data)

	// LL = 3*ln(3/6) + 3*ln(3/6) = 6*ln(0.5)
	expected := 6 * math.Log(0.5)

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("LogLikelihood LocalScore(A, nil): got %f, want %f", score, expected)
	}
}

func TestLogLikelihood_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	ll := NewLogLikelihood()

	score := ll.LocalScore("B", []string{"A"}, data)

	llJ0 := 2*math.Log(2.0/3.0) + 1*math.Log(1.0/3.0)
	llJ1 := 1*math.Log(1.0/3.0) + 2*math.Log(2.0/3.0)
	expected := llJ0 + llJ1

	if !approxEqual(score, expected, 1e-6) {
		t.Errorf("LogLikelihood LocalScore(B, [A]): got %f, want %f", score, expected)
	}
}

func TestLogLikelihood_GreaterThanOrEqualBIC(t *testing.T) {
	// LL has no penalty, so it should always be >= BIC.
	data := makeTestData()
	ll := NewLogLikelihood()
	bic := NewBIC()

	for _, parents := range [][]string{nil, {"X"}} {
		scoreLL := ll.LocalScore("Y", parents, data)
		scoreBIC := bic.LocalScore("Y", parents, data)
		if scoreLL < scoreBIC {
			t.Errorf("LogLikelihood should be >= BIC: LL=%f, BIC=%f (parents=%v)",
				scoreLL, scoreBIC, parents)
		}
	}
}

func TestLogLikelihood_GreaterThanOrEqualAIC(t *testing.T) {
	data := makeTestData()
	ll := NewLogLikelihood()
	aic := NewAIC()

	for _, parents := range [][]string{nil, {"X"}} {
		scoreLL := ll.LocalScore("Y", parents, data)
		scoreAIC := aic.LocalScore("Y", parents, data)
		if scoreLL < scoreAIC {
			t.Errorf("LogLikelihood should be >= AIC: LL=%f, AIC=%f (parents=%v)",
				scoreLL, scoreAIC, parents)
		}
	}
}

func TestLogLikelihood_MonotonicallyIncreases(t *testing.T) {
	// Adding more parents should never decrease the log-likelihood.
	data := makeTestData()
	ll := NewLogLikelihood()

	scoreNoParent := ll.LocalScore("Y", nil, data)
	scoreWithParent := ll.LocalScore("Y", []string{"X"}, data)

	if scoreWithParent < scoreNoParent {
		t.Errorf("LogLikelihood should not decrease with more parents: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestLogLikelihood_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	ll := NewLogLikelihood()

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := ll.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += ll.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("LogLikelihood Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestLogLikelihood_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	ll := NewLogLikelihood()
	score := ll.LocalScore("A", nil, data)
	if score != 0 {
		t.Errorf("LogLikelihood LocalScore on empty data: got %f, want 0", score)
	}
}

func TestLogLikelihood_InterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewLogLikelihood()
}
