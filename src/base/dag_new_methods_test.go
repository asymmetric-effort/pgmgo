//go:build unit

package base

import (
	"strings"
	"testing"
)

func buildStudentDAG(t *testing.T) *DAG {
	t.Helper()
	d := NewDAG()
	for _, n := range []string{"D", "G", "I", "L", "S"} {
		if err := d.AddNode(n); err != nil {
			t.Fatal(err)
		}
	}
	for _, e := range [][2]string{{"D", "G"}, {"I", "G"}, {"G", "L"}, {"I", "S"}} {
		if err := d.AddEdge(e[0], e[1]); err != nil {
			t.Fatal(err)
		}
	}
	return d
}

func TestAddEdgesFrom(t *testing.T) {
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C")
	err := d.AddEdgesFrom([][2]string{{"A", "B"}, {"B", "C"}})
	if err != nil {
		t.Fatalf("AddEdgesFrom failed: %v", err)
	}
	if !d.HasEdge("A", "B") || !d.HasEdge("B", "C") {
		t.Error("edges not added")
	}
}

func TestAddEdgesFromCycle(t *testing.T) {
	d := NewDAG()
	_ = d.AddNodes("A", "B")
	_ = d.AddEdge("A", "B")
	err := d.AddEdgesFrom([][2]string{{"B", "A"}})
	if err == nil {
		t.Error("expected error for cycle")
	}
}

func TestMoralize(t *testing.T) {
	d := buildStudentDAG(t)
	moral := d.Moralize()
	if moral == nil {
		t.Fatal("Moralize returned nil")
	}
}

func TestGetIndependencies(t *testing.T) {
	d := buildStudentDAG(t)
	indeps := d.GetIndependencies()
	if len(indeps) == 0 {
		t.Error("expected some independence assertions")
	}
	// D and I should be independent (no parents for either, not adjacent).
	found := false
	for _, ind := range indeps {
		if (ind[0][0] == "D" && ind[1][0] == "I") || (ind[0][0] == "I" && ind[1][0] == "D") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected D _|_ I to be found")
	}
}

func TestLocalIndependencies(t *testing.T) {
	d := buildStudentDAG(t)
	indeps := d.LocalIndependencies("G")
	// G depends on D and I; given {D, I}, G should be independent of non-descendants (S).
	if len(indeps) == 0 {
		t.Error("expected local independence assertions for G")
	}
}

func TestLocalIndependenciesRoot(t *testing.T) {
	d := buildStudentDAG(t)
	indeps := d.LocalIndependencies("D")
	// D is a root. Independent of {I, S} given {} (no parents).
	if len(indeps) == 0 {
		t.Error("expected local independence for root D")
	}
}

func TestIsIEquivalent(t *testing.T) {
	d1 := buildStudentDAG(t)
	d2 := d1.Copy()
	if !d1.IsIEquivalent(d2) {
		t.Error("copy should be I-equivalent")
	}
}

func TestIsIEquivalentDifferent(t *testing.T) {
	d1 := buildStudentDAG(t)
	d2 := NewDAG()
	_ = d2.AddNodes("D", "G", "I", "L", "S")
	// Different structure: D -> G -> L, I -> G, I -> S, but also D -> I (extra edge).
	_ = d2.AddEdge("D", "G")
	_ = d2.AddEdge("I", "G")
	_ = d2.AddEdge("G", "L")
	_ = d2.AddEdge("I", "S")
	_ = d2.AddEdge("D", "I")
	if d1.IsIEquivalent(d2) {
		t.Error("different skeleton should not be I-equivalent")
	}
}

func TestGetImmoralities(t *testing.T) {
	d := buildStudentDAG(t)
	imm := d.GetImmoralities()
	// D -> G <- I is a v-structure.
	found := false
	for _, im := range imm {
		if im[1] == "G" && ((im[0] == "D" && im[2] == "I") || (im[0] == "I" && im[2] == "D")) {
			found = true
		}
	}
	if !found {
		t.Error("expected D -> G <- I v-structure")
	}
}

func TestIsDConnected(t *testing.T) {
	d := buildStudentDAG(t)
	// D and I are d-separated marginally (no common ancestor or path).
	if d.IsDConnected("D", "I", nil) {
		t.Error("D and I should not be d-connected marginally")
	}
	// D and I become d-connected given G (explaining away).
	if !d.IsDConnected("D", "I", []string{"G"}) {
		t.Error("D and I should be d-connected given G")
	}
}

func TestMinimalDSeparator(t *testing.T) {
	d := buildStudentDAG(t)
	sep, ok := d.MinimalDSeparator("D", "S")
	if !ok {
		t.Fatal("expected a d-separator for D and S")
	}
	// Any valid separator should actually d-separate.
	xSet := map[string]bool{"D": true}
	ySet := map[string]bool{"S": true}
	zSet := make(map[string]bool, len(sep))
	for _, v := range sep {
		zSet[v] = true
	}
	// Check using d.IsDConnected inverse.
	if d.IsDConnected("D", "S", sep) {
		t.Error("minimal d-separator should d-separate D and S")
	}
	_ = xSet
	_ = ySet
	_ = zSet
}

func TestGetMarkovBlanket(t *testing.T) {
	d := buildStudentDAG(t)
	mb := d.GetMarkovBlanket("G")
	// G's Markov blanket: parents {D, I}, children {L}, co-parents of children = {}.
	expected := map[string]bool{"D": true, "I": true, "L": true}
	if len(mb) != len(expected) {
		t.Errorf("expected %d nodes in Markov blanket, got %d: %v", len(expected), len(mb), mb)
	}
	for _, v := range mb {
		if !expected[v] {
			t.Errorf("unexpected node %q in Markov blanket", v)
		}
	}
}

func TestActiveTrailNodes(t *testing.T) {
	d := buildStudentDAG(t)
	active := d.ActiveTrailNodes("D", nil)
	// From D, active trail reaches G, L (through G), and possibly I through G if observed.
	if len(active) == 0 {
		t.Error("expected some active trail nodes from D")
	}
}

func TestGetAncestors(t *testing.T) {
	d := buildStudentDAG(t)
	anc := d.GetAncestors("L")
	// L's ancestors: G, D, I.
	expected := map[string]bool{"G": true, "D": true, "I": true}
	if len(anc) != len(expected) {
		t.Errorf("expected %d ancestors, got %d: %v", len(expected), len(anc), anc)
	}
}

func TestGetDescendants(t *testing.T) {
	d := buildStudentDAG(t)
	desc := d.GetDescendants("I")
	// I's descendants: G, L, S.
	expected := map[string]bool{"G": true, "L": true, "S": true}
	if len(desc) != len(expected) {
		t.Errorf("expected %d descendants, got %d: %v", len(expected), len(desc), desc)
	}
}

func TestToPDAG(t *testing.T) {
	d := buildStudentDAG(t)
	pdag := d.ToPDAG()
	if pdag == nil {
		t.Fatal("ToPDAG returned nil")
	}
	nodes := pdag.Nodes()
	if len(nodes) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(nodes))
	}
	// The v-structure D -> G <- I should have directed edges preserved.
	if !pdag.HasDirectedEdge("D", "G") {
		t.Error("expected directed edge D -> G (compelled by v-structure)")
	}
	if !pdag.HasDirectedEdge("I", "G") {
		t.Error("expected directed edge I -> G (compelled by v-structure)")
	}
}

func TestDo(t *testing.T) {
	d := buildStudentDAG(t)
	doG := d.Do("G")
	// After do(G), G should have no parents.
	parents := doG.Parents("G")
	if len(parents) != 0 {
		t.Errorf("expected 0 parents after do(G), got %d: %v", len(parents), parents)
	}
	// G should still have children.
	children := doG.Children("G")
	if len(children) != 1 || children[0] != "L" {
		t.Errorf("expected children [L] after do(G), got %v", children)
	}
}

func TestGetAncestralGraph(t *testing.T) {
	d := buildStudentDAG(t)
	ag := d.GetAncestralGraph([]string{"L"})
	// Ancestral graph of {L} includes L, G, D, I.
	nodes := ag.Nodes()
	if len(nodes) != 4 {
		t.Errorf("expected 4 nodes in ancestral graph, got %d: %v", len(nodes), nodes)
	}
}

func TestFromLavaan_Parsed(t *testing.T) {
	syntax := "Y ~ X1 + X2\nZ ~ Y"
	d, err := FromLavaan(syntax)
	if err != nil {
		t.Fatalf("FromLavaan: %v", err)
	}
	if !d.HasEdge("X1", "Y") || !d.HasEdge("X2", "Y") || !d.HasEdge("Y", "Z") {
		t.Error("expected all edges from lavaan syntax")
	}
}

func TestFromLavaan_Empty(t *testing.T) {
	_, err := FromLavaan("")
	if err == nil {
		t.Error("expected error for empty lavaan syntax")
	}
}

func TestFromLavaan_NoValidLines(t *testing.T) {
	_, err := FromLavaan("no tilde here\njust text")
	if err == nil {
		t.Error("expected error for no valid lavaan lines")
	}
}

func TestFromDagitty_Parsed(t *testing.T) {
	syntax := "dag { X -> Y; Y -> Z }"
	d, err := FromDagitty(syntax)
	if err != nil {
		t.Fatalf("FromDagitty: %v", err)
	}
	if !d.HasEdge("X", "Y") || !d.HasEdge("Y", "Z") {
		t.Error("expected edges from dagitty syntax")
	}
}

func TestFromDagitty_Empty(t *testing.T) {
	_, err := FromDagitty("")
	if err == nil {
		t.Error("expected error for empty dagitty syntax")
	}
}

func TestFromDagitty_Multiline(t *testing.T) {
	syntax := "dag {\n  A -> B\n  B -> C\n}"
	d, err := FromDagitty(syntax)
	if err != nil {
		t.Fatalf("FromDagitty multiline: %v", err)
	}
	if len(d.Edges()) != 2 {
		t.Errorf("expected 2 edges, got %d", len(d.Edges()))
	}
}

func TestOutDegreeIter(t *testing.T) {
	d := buildStudentDAG(t)
	out := d.OutDegreeIter()
	// D has children: G
	if out["D"] != 1 {
		t.Errorf("expected D out-degree 1, got %d", out["D"])
	}
	// I has children: G, S
	if out["I"] != 2 {
		t.Errorf("expected I out-degree 2, got %d", out["I"])
	}
	// L is a leaf
	if out["L"] != 0 {
		t.Errorf("expected L out-degree 0, got %d", out["L"])
	}
}

func TestInDegreeIter(t *testing.T) {
	d := buildStudentDAG(t)
	in := d.InDegreeIter()
	// D is a root
	if in["D"] != 0 {
		t.Errorf("expected D in-degree 0, got %d", in["D"])
	}
	// G has parents: D, I
	if in["G"] != 2 {
		t.Errorf("expected G in-degree 2, got %d", in["G"])
	}
}

func TestGetRandomDAG_Basic(t *testing.T) {
	d, err := GetRandomDAG(5, 4, 42)
	if err != nil {
		t.Fatalf("GetRandomDAG: %v", err)
	}
	if len(d.Nodes()) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(d.Nodes()))
	}
	if len(d.Edges()) != 4 {
		t.Errorf("expected 4 edges, got %d", len(d.Edges()))
	}
	_, err = d.TopologicalOrder()
	if err != nil {
		t.Errorf("random DAG is not a valid DAG: %v", err)
	}
}

func TestGetRandomDAG_Deterministic(t *testing.T) {
	d1, _ := GetRandomDAG(6, 5, 123)
	d2, _ := GetRandomDAG(6, 5, 123)
	edges1 := d1.Edges()
	edges2 := d2.Edges()
	if len(edges1) != len(edges2) {
		t.Fatal("same seed should produce same DAG")
	}
	for i := range edges1 {
		if edges1[i].Src != edges2[i].Src || edges1[i].Dst != edges2[i].Dst {
			t.Errorf("edge %d differs", i)
		}
	}
}

func TestGetRandomDAG_InvalidArgs(t *testing.T) {
	if _, err := GetRandomDAG(0, 0, 1); err == nil {
		t.Error("expected error for nNodes=0")
	}
	if _, err := GetRandomDAG(3, 10, 1); err == nil {
		t.Error("expected error for nEdges > maxEdges")
	}
}

func TestToGraphviz(t *testing.T) {
	d := buildStudentDAG(t)
	dot := d.ToGraphviz()
	if !strings.Contains(dot, "digraph") {
		t.Error("expected 'digraph' in output")
	}
	if !strings.Contains(dot, "->") {
		t.Error("expected '->' in output")
	}
}

func TestToLavaan(t *testing.T) {
	d := buildStudentDAG(t)
	lav := d.ToLavaan()
	if !strings.Contains(lav, "~") {
		t.Error("expected '~' in Lavaan output")
	}
}

func TestToDagitty(t *testing.T) {
	d := buildStudentDAG(t)
	dag := d.ToDagitty()
	if !strings.Contains(dag, "dag {") {
		t.Error("expected 'dag {' in Dagitty output")
	}
}

func TestToDaft(t *testing.T) {
	d := buildStudentDAG(t)
	daft := d.ToDaft()
	if !strings.Contains(daft, "daft") {
		t.Error("expected 'daft' in Daft output")
	}
}

func TestEdgeStrength(t *testing.T) {
	d := buildStudentDAG(t)
	strengths := d.EdgeStrength()
	if len(strengths) == 0 {
		t.Error("expected some edge strengths")
	}
	for key, val := range strengths {
		if val < 0 {
			t.Errorf("negative strength for edge %v: %f", key, val)
		}
	}
}

func TestGetStats(t *testing.T) {
	d := buildStudentDAG(t)
	stats := d.GetStats()
	if stats.NumNodes != 5 {
		t.Errorf("expected 5 nodes, got %d", stats.NumNodes)
	}
	if stats.NumEdges != 4 {
		t.Errorf("expected 4 edges, got %d", stats.NumEdges)
	}
	if stats.NumRoots != 2 {
		t.Errorf("expected 2 roots, got %d", stats.NumRoots)
	}
	if stats.NumLeaves != 2 {
		t.Errorf("expected 2 leaves, got %d", stats.NumLeaves)
	}
}

func TestGetRandom(t *testing.T) {
	d := GetRandom([]string{"A", "B", "C", "D"}, 0.5, 42)
	if d == nil {
		t.Fatal("GetRandom returned nil")
	}
	nodes := d.Nodes()
	if len(nodes) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(nodes))
	}
	// Should be a valid DAG.
	_, err := d.TopologicalOrder()
	if err != nil {
		t.Errorf("random DAG is not a valid DAG: %v", err)
	}
}

func TestDiGraph(t *testing.T) {
	d := buildStudentDAG(t)
	g := d.DiGraph()
	if g == nil {
		t.Fatal("DiGraph returned nil")
	}
	if g.NumberOfNodes() != 5 {
		t.Errorf("expected 5 nodes, got %d", g.NumberOfNodes())
	}
}
