//go:build unit

package learning

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestExhaustiveSearch_ThreeVariables(t *testing.T) {
	data := syntheticData() // A, B, C with A=B=C
	es := NewExhaustiveSearch(data, countScore)
	bn, err := es.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}

	edges := bn.Edges()
	t.Logf("Best DAG edges: %v", edges)

	// With perfect correlation, the best DAG should have edges connecting all
	// three variables.
	if len(edges) == 0 {
		t.Error("expected at least one edge in best DAG")
	}
}

func TestExhaustiveSearch_TwoVariables(t *testing.T) {
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		xVals[i] = i % 2
		yVals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	es := NewExhaustiveSearch(data, countScore)
	bn, err := es.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	t.Logf("Best DAG edges: %v", edges)

	// Should have exactly 1 edge (X->Y or Y->X).
	if len(edges) != 1 {
		t.Errorf("expected 1 edge for 2 perfectly correlated variables, got %d: %v", len(edges), edges)
	}
}

func TestExhaustiveSearch_TooManyVariables(t *testing.T) {
	vals := make([]any, 10)
	for i := range vals {
		vals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", vals),
		"B": tabgo.NewSeries("B", vals),
		"C": tabgo.NewSeries("C", vals),
		"D": tabgo.NewSeries("D", vals),
		"E": tabgo.NewSeries("E", vals),
	})

	es := NewExhaustiveSearch(data, countScore)
	_, err := es.Estimate()
	if err == nil {
		t.Fatal("expected error for >4 variables, got nil")
	}
}

func TestExhaustiveSearch_SingleVariable(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1, 2, 3}),
	})
	es := NewExhaustiveSearch(data, countScore)
	bn, err := es.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}
	edges := bn.Edges()
	if len(edges) != 0 {
		t.Errorf("expected no edges for single variable, got %v", edges)
	}
}

func TestExhaustiveSearch_AllScores(t *testing.T) {
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	for i := 0; i < n; i++ {
		xVals[i] = i % 2
		yVals[i] = i % 2
		zVals[i] = (i + 1) % 2 // Z is anti-correlated with X and Y
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
		"Z": tabgo.NewSeries("Z", zVals),
	})

	es := NewExhaustiveSearch(data, countScore)
	scores, err := es.AllScores()
	if err != nil {
		t.Fatalf("AllScores() error: %v", err)
	}

	if len(scores) == 0 {
		t.Fatal("expected non-empty scores map")
	}

	// The empty graph should be one of the entries.
	emptyKey := "[]"
	if _, ok := scores[emptyKey]; !ok {
		t.Error("expected empty graph to be scored")
	}

	// The empty graph score should be 0 (no parents = 0 for countScore).
	if scores[emptyKey] != 0.0 {
		t.Errorf("expected empty graph score to be 0, got %f", scores[emptyKey])
	}

	t.Logf("Total DAGs scored: %d", len(scores))

	// For 3 variables there are 25 possible DAGs.
	if len(scores) != 25 {
		t.Errorf("expected 25 DAGs for 3 variables, got %d", len(scores))
	}
}

func TestExhaustiveSearch_NoColumns(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{})
	es := NewExhaustiveSearch(data, countScore)
	_, err := es.Estimate()
	if err == nil {
		t.Fatal("expected error for empty DataFrame, got nil")
	}
}

func TestExhaustiveSearch_FourVariables(t *testing.T) {
	// Should work for exactly 4 variables.
	n := 50
	vals := make([]any, n)
	for i := range vals {
		vals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", vals),
		"B": tabgo.NewSeries("B", vals),
		"C": tabgo.NewSeries("C", vals),
		"D": tabgo.NewSeries("D", vals),
	})

	es := NewExhaustiveSearch(data, countScore)
	bn, err := es.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(nodes))
	}

	// Verify the result is a DAG (no cycles).
	edges := bn.Edges()
	adj := make(map[string][]string)
	inDeg := make(map[string]int)
	for _, node := range nodes {
		adj[node] = nil
		inDeg[node] = 0
	}
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], e[1])
		inDeg[e[1]]++
	}
	queue := make([]string, 0)
	for n, d := range inDeg {
		if d == 0 {
			queue = append(queue, n)
		}
	}
	visited := 0
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		visited++
		for _, s := range adj[node] {
			inDeg[s]--
			if inDeg[s] == 0 {
				queue = append(queue, s)
			}
		}
	}
	if visited != len(nodes) {
		t.Error("learned graph contains a cycle")
	}
}
