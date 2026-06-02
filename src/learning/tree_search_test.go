//go:build unit

package learning

import (
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// treeTestData creates a dataset where X determines Y and Y determines Z.
func treeTestData() *tabgo.DataFrame {
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	for i := 0; i < n; i++ {
		x := i % 3
		y := x // Y depends on X
		z := y // Z depends on Y
		xVals[i] = x
		yVals[i] = y
		zVals[i] = z
	}
	return tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
		"Z": tabgo.NewSeries("Z", zVals),
	})
}

func TestTreeSearch_ChowLiuBasic(t *testing.T) {
	data := treeTestData()
	ts := NewTreeSearch(data)
	bn, err := ts.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}

	edges := bn.Edges()
	t.Logf("Chow-Liu edges: %v", edges)

	// A tree on 3 nodes has exactly 2 edges.
	if len(edges) != 2 {
		t.Errorf("expected 2 edges (tree), got %d: %v", len(edges), edges)
	}

	// Verify it's a DAG (no cycles).
	adj := make(map[string][]string)
	inDeg := make(map[string]int)
	for _, n := range nodes {
		adj[n] = nil
		inDeg[n] = 0
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
		t.Error("tree structure contains a cycle")
	}
}

func TestTreeSearch_WithRoot(t *testing.T) {
	data := treeTestData()
	ts := NewTreeSearch(data, WithRoot("Z"))
	bn, err := ts.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	t.Logf("Chow-Liu edges (root=Z): %v", edges)

	// All edges should be oriented away from Z.
	// Z should have no parents.
	parents := bn.Parents("Z")
	if len(parents) != 0 {
		t.Errorf("expected root Z to have no parents, got %v", parents)
	}

	// Z should be an ancestor of all other nodes (directly or indirectly).
	children := bn.Children("Z")
	if len(children) == 0 {
		t.Error("expected root Z to have at least one child")
	}
}

func TestTreeSearch_TAN(t *testing.T) {
	// Create data with class variable C and features X, Y, Z.
	n := 200
	cVals := make([]any, n)
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	for i := 0; i < n; i++ {
		c := i % 2
		x := c           // X depends on C
		y := (c + 1) % 2 // Y anti-correlated with C
		z := c           // Z depends on C
		cVals[i] = c
		xVals[i] = x
		yVals[i] = y
		zVals[i] = z
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"C": tabgo.NewSeries("C", cVals),
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
		"Z": tabgo.NewSeries("Z", zVals),
	})

	ts := NewTreeSearch(data, WithClassVariable("C"))
	bn, err := ts.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 4 {
		t.Fatalf("expected 4 nodes, got %d", len(nodes))
	}

	edges := bn.Edges()
	t.Logf("TAN edges: %v", edges)

	// C should be parent of all features X, Y, Z.
	cChildren := bn.Children("C")
	sort.Strings(cChildren)
	if len(cChildren) < 3 {
		t.Errorf("expected C to be parent of X, Y, Z; children: %v", cChildren)
	}

	// Features should form a tree among themselves (2 edges).
	featureEdges := 0
	for _, e := range edges {
		if e[0] != "C" && e[1] != "C" {
			featureEdges++
		}
	}
	if featureEdges != 2 {
		t.Errorf("expected 2 feature-to-feature edges (tree), got %d", featureEdges)
	}
}

func TestTreeSearch_InvalidClassVar(t *testing.T) {
	data := treeTestData()
	ts := NewTreeSearch(data, WithClassVariable("MISSING"))
	_, err := ts.Estimate()
	if err == nil {
		t.Fatal("expected error for missing class variable, got nil")
	}
}

func TestTreeSearch_InvalidRoot(t *testing.T) {
	data := treeTestData()
	ts := NewTreeSearch(data, WithRoot("MISSING"))
	_, err := ts.Estimate()
	if err == nil {
		t.Fatal("expected error for missing root variable, got nil")
	}
}

func TestTreeSearch_TooFewVariables(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1, 2, 3}),
	})
	ts := NewTreeSearch(data)
	_, err := ts.Estimate()
	if err == nil {
		t.Fatal("expected error for single variable, got nil")
	}
}

func TestTreeSearch_IndependentVariables(t *testing.T) {
	// Two variables with no correlation.
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

	ts := NewTreeSearch(data)
	bn, err := ts.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	// Even for independent variables, Chow-Liu produces a tree (1 edge for 2 nodes).
	edges := bn.Edges()
	if len(edges) != 1 {
		t.Errorf("expected 1 edge for 2 variables, got %d", len(edges))
	}
}

func TestTreeSearch_MutualInformation(t *testing.T) {
	// Verify that highly correlated pairs get higher MI than uncorrelated.
	n := 200
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	for i := 0; i < n; i++ {
		aVals[i] = i % 2
		bVals[i] = i % 2       // B = A (perfect correlation)
		cVals[i] = (i / 2) % 2 // C roughly independent of A
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
	})

	ts := NewTreeSearch(data)

	miAB := ts.mutualInformation("A", "B", n)
	miAC := ts.mutualInformation("A", "C", n)

	t.Logf("MI(A,B) = %f, MI(A,C) = %f", miAB, miAC)

	if miAB <= miAC {
		t.Errorf("expected MI(A,B) > MI(A,C) since B=A; got MI(A,B)=%f, MI(A,C)=%f", miAB, miAC)
	}
}
