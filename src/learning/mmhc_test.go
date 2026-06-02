//go:build unit

package learning

import (
	"fmt"
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// simpleCITest is a basic CI test for testing. It tests conditional independence
// by checking if the mutual information between x and y (given z) is below a
// threshold. Returns a chi-squared-like statistic as the first return value.
func simpleCITest(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	if n == 0 {
		return 0, 1, true
	}

	xVals := data.Column(x).Values()
	yVals := data.Column(y).Values()

	// Compute conditional counts.
	if len(z) == 0 {
		// Marginal test: compute chi-squared-like statistic.
		type pair struct{ a, b interface{} }
		joint := make(map[pair]int)
		xMarg := make(map[interface{}]int)
		yMarg := make(map[interface{}]int)

		for i := 0; i < n; i++ {
			joint[pair{xVals[i], yVals[i]}]++
			xMarg[xVals[i]]++
			yMarg[yVals[i]]++
		}

		nf := float64(n)
		stat := 0.0
		for p, obs := range joint {
			expected := float64(xMarg[p.a]) * float64(yMarg[p.b]) / nf
			if expected > 0 {
				diff := float64(obs) - expected
				stat += diff * diff / expected
			}
		}

		// Use a simple threshold.
		threshold := 3.841 * (1.0 / significance) * 0.05 // rough approximation
		pval := math.Exp(-stat / 2.0)
		return stat, pval, stat < threshold
	}

	// Conditional test: stratify by z values.
	// Build conditioning key for each row.
	zVals := make([][]interface{}, len(z))
	for i, zVar := range z {
		vals := data.Column(zVar).Values()
		zVals[i] = make([]interface{}, n)
		for j := 0; j < n; j++ {
			zVals[i][j] = vals[j]
		}
	}

	type pair struct{ a, b interface{} }

	totalStat := 0.0
	// Group by z values.
	groups := make(map[string][]int)
	for i := 0; i < n; i++ {
		key := ""
		for _, zv := range zVals {
			key += fmt.Sprintf("%v,", zv[i])
		}
		groups[key] = append(groups[key], i)
	}

	for _, indices := range groups {
		joint := make(map[pair]int)
		xMarg := make(map[interface{}]int)
		yMarg := make(map[interface{}]int)
		gn := len(indices)
		if gn == 0 {
			continue
		}

		for _, i := range indices {
			joint[pair{xVals[i], yVals[i]}]++
			xMarg[xVals[i]]++
			yMarg[yVals[i]]++
		}

		gnf := float64(gn)
		for p, obs := range joint {
			expected := float64(xMarg[p.a]) * float64(yMarg[p.b]) / gnf
			if expected > 0 {
				diff := float64(obs) - expected
				totalStat += diff * diff / expected
			}
		}
	}

	threshold := 3.841 * (1.0 / significance) * 0.05
	pval := math.Exp(-totalStat / 2.0)
	return totalStat, pval, totalStat < threshold
}

func TestMMHC_BasicStructure(t *testing.T) {
	data := syntheticData() // A=B=C
	mmhc := NewMMHC(data, countScore, simpleCITest, 0.05)
	bn, err := mmhc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}

	edges := bn.Edges()
	t.Logf("MMHC edges: %v", edges)

	// Should learn some edges connecting A, B, C.
	if len(edges) == 0 {
		t.Error("expected at least one edge")
	}
}

func TestMMHC_TwoVariables(t *testing.T) {
	n := 200
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

	mmhc := NewMMHC(data, countScore, simpleCITest, 0.05)
	bn, err := mmhc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	t.Logf("MMHC edges: %v", edges)

	es := edgeSet(edges)
	hasXY := es["X->Y"] || es["Y->X"]
	if !hasXY {
		t.Error("expected edge between X and Y")
	}
}

func TestMMHC_IndependentVariables(t *testing.T) {
	// X and Y are independent. The CI test should detect this and the MMPC
	// phase should produce an empty candidate set.
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		xVals[i] = i % 2
		yVals[i] = (i / 2) % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	// Use a CI test that always reports independence for these variables.
	alwaysIndep := func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
		return 0, 1, true
	}

	mmhc := NewMMHC(data, countScore, alwaysIndep, 0.05)
	bn, err := mmhc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 0 {
		t.Errorf("expected no edges for independent variables, got %v", edges)
	}
}

func TestMMHC_TooFewVariables(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1, 2, 3}),
	})
	mmhc := NewMMHC(data, countScore, simpleCITest, 0.05)
	_, err := mmhc.Estimate()
	if err == nil {
		t.Fatal("expected error for single variable, got nil")
	}
}

func TestMMHC_CandidateRestriction(t *testing.T) {
	// Create data where A->B is strong, A->C is strong, but B-C is independent.
	n := 200
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	for i := 0; i < n; i++ {
		a := i % 2
		b := a // B depends on A
		c := a // C depends on A
		aVals[i] = a
		bVals[i] = b
		cVals[i] = c
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
	})

	mmhc := NewMMHC(data, countScore, simpleCITest, 0.05)
	bn, err := mmhc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}

	edges := bn.Edges()
	t.Logf("MMHC edges for collider-like data: %v", edges)

	// Should have some edges.
	if len(edges) == 0 {
		t.Error("expected at least one edge")
	}

	// The result should be a DAG.
	adj := make(map[string][]string)
	inDeg := make(map[string]int)
	for _, nd := range nodes {
		adj[nd] = nil
		inDeg[nd] = 0
	}
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], e[1])
		inDeg[e[1]]++
	}
	queue := make([]string, 0)
	for nd, d := range inDeg {
		if d == 0 {
			queue = append(queue, nd)
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
		t.Error("MMHC result contains a cycle")
	}
}
