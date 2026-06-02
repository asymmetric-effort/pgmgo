//go:build unit

package learning_test

import (
	"sort"
	"strconv"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/learning"
	"github.com/asymmetric-effort/pgmgo/src/structure_score"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// structureInput holds the input section of a structure learning test case.
type structureInput struct {
	Variables   []string       `json:"variables"`
	NodeCards   map[string]int `json:"node_cards"`
	DataColumns []string       `json:"data_columns"`
	DataRows    [][]float64    `json:"data_rows"`
}

// hillClimbExpected holds the expected output for HillClimb tests.
type hillClimbExpected struct {
	LearnedEdges  [][]string `json:"learned_edges"`
	SkeletonEdges [][]string `json:"skeleton_edges"`
}

// buildDataFrameStrFromRows creates a DataFrame with string-typed values.
func buildDataFrameStrFromRows(columns []string, rows [][]float64) *tabgo.DataFrame {
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

// edgeToSortedPair returns a sorted [2]string from an edge for skeleton comparison.
func edgeToSortedPair(e [2]string) [2]string {
	if e[0] > e[1] {
		return [2]string{e[1], e[0]}
	}
	return e
}

func TestCrossval_HillClimbBIC(t *testing.T) {
	ff := testutil.LoadFixtures(t, "structure_learning/fixtures.json")
	tc := ff.FindTestCase(t, "hill_climb_bic")

	var input structureInput
	tc.UnmarshalInput(t, &input)

	var expected hillClimbExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildDataFrameStrFromRows(input.DataColumns, input.DataRows)

	bic := structure_score.NewBIC()
	scoreFn := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		return bic.LocalScore(variable, parents, data)
	}

	hc := learning.NewHillClimbSearch(df, scoreFn)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("HillClimbSearch.Estimate(): %v", err)
	}

	// Extract learned skeleton edges (sorted pairs) for comparison.
	// We compare skeletons because direction within a Markov equivalence class
	// may differ between implementations.
	gotEdges := bn.Edges()
	gotSkeleton := make([][2]string, len(gotEdges))
	for i, e := range gotEdges {
		gotSkeleton[i] = edgeToSortedPair(e)
	}
	sort.Slice(gotSkeleton, func(i, j int) bool {
		if gotSkeleton[i][0] != gotSkeleton[j][0] {
			return gotSkeleton[i][0] < gotSkeleton[j][0]
		}
		return gotSkeleton[i][1] < gotSkeleton[j][1]
	})

	// Build expected skeleton from fixture.
	expectedSkeleton := make([][2]string, len(expected.SkeletonEdges))
	for i, e := range expected.SkeletonEdges {
		expectedSkeleton[i] = [2]string{e[0], e[1]}
	}
	sort.Slice(expectedSkeleton, func(i, j int) bool {
		if expectedSkeleton[i][0] != expectedSkeleton[j][0] {
			return expectedSkeleton[i][0] < expectedSkeleton[j][0]
		}
		return expectedSkeleton[i][1] < expectedSkeleton[j][1]
	})

	if len(gotSkeleton) != len(expectedSkeleton) {
		t.Fatalf("skeleton edge count: expected %d, got %d\nexpected: %v\ngot: %v",
			len(expectedSkeleton), len(gotSkeleton), expectedSkeleton, gotSkeleton)
	}

	for i := range expectedSkeleton {
		if gotSkeleton[i] != expectedSkeleton[i] {
			t.Errorf("skeleton edge[%d]: expected %v, got %v", i, expectedSkeleton[i], gotSkeleton[i])
		}
	}
}
