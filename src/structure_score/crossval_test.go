//go:build unit

package structure_score_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/structure_score"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildScoreDF creates a DataFrame with int-typed values from fixture rows.
func buildScoreDF(columns []string, rows [][]float64) *tabgo.DataFrame {
	anyRows := make([][]any, len(rows))
	for i, row := range rows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			anyRow[j] = strconv.Itoa(int(v))
		}
		anyRows[i] = anyRow
	}
	return tabgo.NewDataFrameFromRows(columns, anyRows)
}

func TestCrossval_LocalScores_G_DI(t *testing.T) {
	ff := testutil.LoadFixtures(t, "scores/fixtures.json")
	tc := ff.FindTestCase(t, "local_scores_g_di")

	var input struct {
		Variable    string      `json:"variable"`
		Parents     []string    `json:"parents"`
		DataColumns []string    `json:"data_columns"`
		DataRows    [][]float64 `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Scores map[string]float64 `json:"scores"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildScoreDF(input.DataColumns, input.DataRows)

	scorers := map[string]structure_score.StructureScore{
		"BIC":  structure_score.NewBIC(),
		"AIC":  structure_score.NewAIC(),
		"BDeu": structure_score.NewBDeu(10.0),
		"BDs":  structure_score.NewBDs(10.0, 1.0),
		"K2":   structure_score.NewK2(),
	}

	for name, scorer := range scorers {
		expectedScore, ok := expected.Scores[name]
		if !ok {
			continue
		}
		got := scorer.LocalScore(input.Variable, input.Parents, df)
		// Allow 5% relative tolerance for scoring differences
		relTol := math.Abs(expectedScore) * 0.05
		if relTol < 1.0 {
			relTol = 1.0
		}
		if math.Abs(got-expectedScore) > relTol {
			t.Errorf("%s local_score(%s, %v): expected %f, got %f (tol=%f)",
				name, input.Variable, input.Parents, expectedScore, got, relTol)
		}
	}
}

func TestCrossval_FullScores_StudentBN(t *testing.T) {
	ff := testutil.LoadFixtures(t, "scores/fixtures.json")
	tc := ff.FindTestCase(t, "full_scores_student_bn")

	var input struct {
		Edges       [][]string          `json:"edges"`
		ParentMap   map[string][]string `json:"parent_map"`
		DataColumns []string            `json:"data_columns"`
		DataRows    [][]float64         `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Scores map[string]float64 `json:"scores"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildScoreDF(input.DataColumns, input.DataRows)

	variables := make([]string, 0, len(input.ParentMap))
	for v := range input.ParentMap {
		variables = append(variables, v)
	}

	scorers := map[string]structure_score.StructureScore{
		"BIC":  structure_score.NewBIC(),
		"AIC":  structure_score.NewAIC(),
		"BDeu": structure_score.NewBDeu(10.0),
		"BDs":  structure_score.NewBDs(10.0, 1.0),
		"K2":   structure_score.NewK2(),
	}

	for name, scorer := range scorers {
		expectedScore, ok := expected.Scores[name]
		if !ok {
			continue
		}
		got := scorer.Score(variables, input.ParentMap, df)
		relTol := math.Abs(expectedScore) * 0.05
		if relTol < 1.0 {
			relTol = 1.0
		}
		if math.Abs(got-expectedScore) > relTol {
			t.Errorf("%s Score: expected %f, got %f (tol=%f)",
				name, expectedScore, got, relTol)
		}
	}
}

func TestCrossval_LocalScoresAllNodes(t *testing.T) {
	ff := testutil.LoadFixtures(t, "scores/fixtures.json")
	tc := ff.FindTestCase(t, "local_scores_all_nodes")

	var input struct {
		ParentMap   map[string][]string `json:"parent_map"`
		DataColumns []string            `json:"data_columns"`
		DataRows    [][]float64         `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		NodeScores map[string]map[string]float64 `json:"node_scores"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildScoreDF(input.DataColumns, input.DataRows)

	scorers := map[string]structure_score.StructureScore{
		"BIC":  structure_score.NewBIC(),
		"AIC":  structure_score.NewAIC(),
		"BDeu": structure_score.NewBDeu(10.0),
		"BDs":  structure_score.NewBDs(10.0, 1.0),
		"K2":   structure_score.NewK2(),
	}

	for node, nodeScores := range expected.NodeScores {
		parents := input.ParentMap[node]
		for name, expectedScore := range nodeScores {
			scorer, ok := scorers[name]
			if !ok {
				continue
			}
			got := scorer.LocalScore(node, parents, df)
			relTol := math.Abs(expectedScore) * 0.05
			if relTol < 1.0 {
				relTol = 1.0
			}
			if math.Abs(got-expectedScore) > relTol {
				t.Errorf("%s local_score(%s, %v): expected %f, got %f",
					name, node, parents, expectedScore, got)
			}
		}
	}
}
