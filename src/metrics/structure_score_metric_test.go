//go:build unit

package metrics

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// mockScorer is a simple scorer for testing that returns a fixed score per variable.
type mockScorer struct {
	scores map[string]float64
}

func (m *mockScorer) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	if s, ok := m.scores[variable]; ok {
		return s
	}
	return 0.0
}

func TestStructureScoreMetric_Basic(t *testing.T) {
	df := tabgo.NewDataFrameFromRows(
		[]string{"A", "B", "C"},
		[][]any{
			{0, 0, 0},
			{1, 1, 1},
		},
	)

	scorer := &mockScorer{
		scores: map[string]float64{
			"A": -1.0,
			"B": -2.0,
			"C": -3.0,
		},
	}

	variables := []string{"A", "B", "C"}
	parentMap := map[string][]string{
		"A": {},
		"B": {"A"},
		"C": {"A", "B"},
	}

	got := StructureScoreMetric(variables, parentMap, df, scorer)
	want := -6.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("StructureScoreMetric = %f, want %f", got, want)
	}
}

func TestStructureScoreMetric_Empty(t *testing.T) {
	df := tabgo.NewDataFrameFromRows([]string{"X"}, [][]any{{0}})
	scorer := &mockScorer{scores: map[string]float64{}}

	got := StructureScoreMetric(nil, map[string][]string{}, df, scorer)
	if got != 0.0 {
		t.Errorf("StructureScoreMetric with no variables = %f, want 0", got)
	}
}

func TestStructureScoreMetric_SingleVariable(t *testing.T) {
	df := tabgo.NewDataFrameFromRows([]string{"X"}, [][]any{{1}, {2}})

	scorer := &mockScorer{
		scores: map[string]float64{"X": -5.5},
	}

	got := StructureScoreMetric([]string{"X"}, map[string][]string{"X": {}}, df, scorer)
	if math.Abs(got-(-5.5)) > 1e-12 {
		t.Errorf("StructureScoreMetric = %f, want -5.5", got)
	}
}

// parentAwareScorer returns different scores based on the number of parents.
type parentAwareScorer struct{}

func (p *parentAwareScorer) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	return -float64(len(parents) + 1)
}

func TestStructureScoreMetric_ParentAware(t *testing.T) {
	df := tabgo.NewDataFrameFromRows(
		[]string{"A", "B", "C"},
		[][]any{{0, 0, 0}},
	)

	scorer := &parentAwareScorer{}
	variables := []string{"A", "B", "C"}
	parentMap := map[string][]string{
		"A": {},         // score: -1
		"B": {"A"},      // score: -2
		"C": {"A", "B"}, // score: -3
	}

	got := StructureScoreMetric(variables, parentMap, df, scorer)
	want := -6.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("StructureScoreMetric = %f, want %f", got, want)
	}
}
