//go:build unit

package learning

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// syntheticData generates a DataFrame where column B depends on A, and C
// depends on B. The "score" function we use rewards edges that align with
// co-occurrence counts, so the search should discover A→B→C.
func syntheticData() *tabgo.DataFrame {
	// 100 rows: A determines B, B determines C.
	n := 200
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	for i := 0; i < n; i++ {
		a := i % 2 // 0 or 1
		b := a     // B copies A
		c := b     // C copies B
		aVals[i] = a
		bVals[i] = b
		cVals[i] = c
	}
	return tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
	})
}

// countScore is a simple scoring function for testing. It measures mutual
// information-like co-occurrence: for each parent, count how often
// (parent_value, variable_value) pairs agree, normalised by N.
// More parents that agree → higher score. No parents → baseline of 0.
func countScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	if len(parents) == 0 {
		return 0.0
	}
	n := data.Len()
	if n == 0 {
		return 0.0
	}

	varVals := data.Column(variable).Values()
	score := 0.0
	for _, p := range parents {
		pVals := data.Column(p).Values()
		matches := 0
		for i := 0; i < n; i++ {
			if fmt.Sprintf("%v", varVals[i]) == fmt.Sprintf("%v", pVals[i]) {
				matches++
			}
		}
		// Reward when parent values match variable values.
		score += float64(matches) / float64(n)
	}
	// Penalise complexity slightly so we don't add useless edges.
	score -= 0.05 * float64(len(parents))
	return score
}

func edgeSet(edges [][2]string) map[string]bool {
	m := make(map[string]bool, len(edges))
	for _, e := range edges {
		m[e[0]+"->"+e[1]] = true
	}
	return m
}

func TestHillClimbSearch_BasicStructure(t *testing.T) {
	data := syntheticData()
	hc := NewHillClimbSearch(data, countScore)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := bn.Nodes()
	sort.Strings(nodes)
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}
	if nodes[0] != "A" || nodes[1] != "B" || nodes[2] != "C" {
		t.Fatalf("expected nodes [A B C], got %v", nodes)
	}

	edges := bn.Edges()
	if len(edges) == 0 {
		t.Fatal("expected at least one edge, got none")
	}

	// With perfect correlation A=B=C, the algorithm should learn edges
	// connecting all three. The exact direction may vary, but we expect
	// connectivity.
	es := edgeSet(edges)
	t.Logf("Learned edges: %v", edges)

	// At least A→B or B→A, and B→C or C→B should exist (or transitive).
	hasAB := es["A->B"] || es["B->A"]
	hasBC := es["B->C"] || es["C->B"]
	hasAC := es["A->C"] || es["C->A"]
	if !hasAB && !hasBC && !hasAC {
		t.Errorf("expected meaningful edges among A,B,C; got %v", edges)
	}
}

func TestHillClimbSearch_EmptyData(t *testing.T) {
	// Single column — no possible edges.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1, 2, 3}),
	})
	hc := NewHillClimbSearch(data, countScore)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}
	edges := bn.Edges()
	if len(edges) != 0 {
		t.Errorf("expected no edges for single-column data, got %v", edges)
	}
}

func TestHillClimbSearch_NoColumns(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{})
	hc := NewHillClimbSearch(data, countScore)
	_, err := hc.Estimate()
	if err == nil {
		t.Fatal("expected error for empty DataFrame, got nil")
	}
}

func TestHillClimbSearch_WhiteList(t *testing.T) {
	data := syntheticData()
	// Force edge C→A (which the algorithm might not learn on its own).
	hc := NewHillClimbSearch(data, countScore,
		WithWhiteList([][2]string{{"C", "A"}}),
	)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	es := edgeSet(edges)
	t.Logf("Learned edges (with whitelist C->A): %v", edges)
	if !es["C->A"] {
		t.Errorf("expected whitelist edge C->A to be present; edges: %v", edges)
	}
}

func TestHillClimbSearch_BlackList(t *testing.T) {
	data := syntheticData()
	// Blacklist both directions between A and B.
	hc := NewHillClimbSearch(data, countScore,
		WithBlackList([][2]string{{"A", "B"}, {"B", "A"}}),
	)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	es := edgeSet(edges)
	t.Logf("Learned edges (with blacklist A<->B): %v", edges)
	if es["A->B"] || es["B->A"] {
		t.Errorf("blacklisted edge A<->B should not be present; edges: %v", edges)
	}
}

func TestHillClimbSearch_MaxIndegree(t *testing.T) {
	// Create data with 4 columns where D depends on A, B, C.
	n := 100
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	dVals := make([]any, n)
	for i := 0; i < n; i++ {
		a := i % 2
		b := i % 2
		c := i % 2
		d := i % 2
		aVals[i] = a
		bVals[i] = b
		cVals[i] = c
		dVals[i] = d
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
		"D": tabgo.NewSeries("D", dVals),
	})

	hc := NewHillClimbSearch(data, countScore, WithMaxIndegree(1))
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	// Verify no node has more than 1 parent.
	for _, node := range bn.Nodes() {
		parents := bn.Parents(node)
		if len(parents) > 1 {
			t.Errorf("node %s has %d parents (max 1): %v", node, len(parents), parents)
		}
	}
}

func TestHillClimbSearch_TabuSize(t *testing.T) {
	data := syntheticData()
	// Tiny tabu list should still work (just may explore differently).
	hc := NewHillClimbSearch(data, countScore, WithTabuSize(2))
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}
	// Just verify it terminates and produces a valid structure.
	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(nodes))
	}
}

func TestHillClimbSearch_NoImprovementStops(t *testing.T) {
	// Score function that never rewards any edges — search should return
	// an empty graph immediately.
	zeroScore := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		return 0.0
	}
	data := syntheticData()
	hc := NewHillClimbSearch(data, zeroScore)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}
	edges := bn.Edges()
	if len(edges) != 0 {
		t.Errorf("expected no edges when score never improves, got %v", edges)
	}
}

func TestHillClimbSearch_KnownChain(t *testing.T) {
	// Build data where the ONLY beneficial edge is X→Y (Y = X+1).
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		xVals[i] = i % 3
		yVals[i] = (i + 1) % 3 // different from X
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	// Score function: only reward parent when parent!=variable has some
	// non-trivial relationship (use a fixed bonus for any parent).
	directedScore := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		if len(parents) == 0 {
			return 0.0
		}
		// Give a fixed bonus per parent.
		return 0.5 * float64(len(parents))
	}
	hc := NewHillClimbSearch(data, directedScore)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}
	edges := bn.Edges()
	if len(edges) == 0 {
		t.Fatal("expected at least one edge")
	}
	// Should have exactly one edge (either X→Y or Y→X).
	if len(edges) != 1 {
		t.Errorf("expected 1 edge, got %d: %v", len(edges), edges)
	}
}

func TestHillClimbSearch_AcyclicityEnforced(t *testing.T) {
	// Score function that strongly wants A→B, B→C, AND C→A (a cycle).
	// The search must not produce a cycle.
	cycleScore := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		score := 0.0
		for _, p := range parents {
			// Reward all edges strongly.
			score += 10.0
			_ = p
		}
		return score
	}

	n := 50
	vals := make([]any, n)
	for i := range vals {
		vals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", vals),
		"B": tabgo.NewSeries("B", vals),
		"C": tabgo.NewSeries("C", vals),
	})

	hc := NewHillClimbSearch(data, cycleScore)
	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	// Verify the result is a DAG: no node should be its own ancestor.
	edges := bn.Edges()
	t.Logf("Learned edges: %v", edges)

	// Build adjacency and check no cycles via topological sort attempt.
	adj := make(map[string][]string)
	inDeg := make(map[string]int)
	for _, node := range bn.Nodes() {
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
	if visited != len(bn.Nodes()) {
		t.Errorf("learned graph contains a cycle! edges: %v", edges)
	}
}

func TestHillClimbSearch_WhiteListConflictCycle(t *testing.T) {
	// Whitelist edges that form a cycle: should return an error.
	data := syntheticData()
	hc := NewHillClimbSearch(data, countScore,
		WithWhiteList([][2]string{{"A", "B"}, {"B", "C"}, {"C", "A"}}),
	)
	_, err := hc.Estimate()
	if err == nil {
		t.Fatal("expected error for cyclic whitelist, got nil")
	}
	if !strings.Contains(err.Error(), "cycle") {
		t.Errorf("expected cycle error, got: %v", err)
	}
}

func TestHillClimbSearch_OptionsApplied(t *testing.T) {
	data := syntheticData()
	hc := NewHillClimbSearch(data, countScore,
		WithMaxIndegree(2),
		WithTabuSize(50),
		WithBlackList([][2]string{{"A", "C"}}),
		WithWhiteList([][2]string{{"A", "B"}}),
	)
	if hc.maxIndegree != 2 {
		t.Errorf("maxIndegree: got %d, want 2", hc.maxIndegree)
	}
	if hc.tabuSize != 50 {
		t.Errorf("tabuSize: got %d, want 50", hc.tabuSize)
	}
	if !hc.blackList[[2]string{"A", "C"}] {
		t.Error("blackList should contain A->C")
	}
	if !hc.whiteList[[2]string{"A", "B"}] {
		t.Error("whiteList should contain A->B")
	}
}
