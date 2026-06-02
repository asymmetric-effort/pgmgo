//go:build unit

package graphgo

import (
	"testing"
)

// Helper: check that a PDAG has exactly the expected directed and undirected edges.
func assertPDAGEdges(t *testing.T, p *PDAG, wantDirected [][2]string, wantUndirected [][2]string) {
	t.Helper()

	gotDirected := p.DirectedEdges()
	if len(gotDirected) != len(wantDirected) {
		t.Fatalf("directed edges: got %d, want %d\ngot:  %v\nwant: %v",
			len(gotDirected), len(wantDirected), gotDirected, wantDirected)
	}
	dirSet := make(map[[2]string]bool)
	for _, e := range gotDirected {
		dirSet[e] = true
	}
	for _, e := range wantDirected {
		if !dirSet[e] {
			t.Fatalf("missing directed edge %v->%v\ngot: %v", e[0], e[1], gotDirected)
		}
	}

	gotUndirected := p.UndirectedEdges()
	if len(gotUndirected) != len(wantUndirected) {
		t.Fatalf("undirected edges: got %d, want %d\ngot:  %v\nwant: %v",
			len(gotUndirected), len(wantUndirected), gotUndirected, wantUndirected)
	}
	undirSet := make(map[[2]string]bool)
	for _, e := range gotUndirected {
		undirSet[e] = true
	}
	for _, e := range wantUndirected {
		// Canonicalize.
		key := e
		if key[0] > key[1] {
			key[0], key[1] = key[1], key[0]
		}
		if !undirSet[key] {
			t.Fatalf("missing undirected edge %v-%v\ngot: %v", e[0], e[1], gotUndirected)
		}
	}
}

// TestDAGToPDAGChain: A→B→C is a Markov equivalence class with all edges undirected.
func TestDAGToPDAGChain(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	p := DAGToPDAG(g)

	// A chain A→B→C has no v-structure, so all edges should be undirected in CPDAG.
	assertPDAGEdges(t, p,
		nil, // no directed edges
		[][2]string{{"A", "B"}, {"B", "C"}},
	)
}

// TestDAGToPDAGVStructure: A→C←B (v-structure) should keep both edges directed.
func TestDAGToPDAGVStructure(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "C")
	g.AddEdge("B", "C")

	p := DAGToPDAG(g)

	// v-structure: A→C←B, both edges must be directed.
	assertPDAGEdges(t, p,
		[][2]string{{"A", "C"}, {"B", "C"}},
		nil, // no undirected edges
	)
}

// TestDAGToPDAGDiamond: A→B, A→C, B→D, C→D.
// v-structures at D? B and C are not parents of D that are non-adjacent...
// Actually B and C are both parents of D and they are not adjacent, so B→D←C is a v-structure.
// A→B and A→C: A is parent of both B and C, but B-C not adjacent... wait, A→B, A→C
// doesn't create v-structure at B or C because there's only one parent each.
func TestDAGToPDAGDiamond(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	p := DAGToPDAG(g)

	// B→D←C is a v-structure. B→D and C→D are compelled.
	// A→B and A→C: by Meek R1, since B→D and B not adj to... let's check:
	// Actually, A→B: after v-structure B→D, C→D are oriented.
	// Meek R1 on A—B: is there w→A where w not adj B? No directed edges into A.
	// Meek R1 on A—C: similarly no.
	// Meek R2 on A—B: is there A→w→B? A's children in directed: none yet (A-B, A-C undirected).
	// So A-B and A-C should remain undirected.
	assertPDAGEdges(t, p,
		[][2]string{{"B", "D"}, {"C", "D"}},
		[][2]string{{"A", "B"}, {"A", "C"}},
	)
}

// TestDAGToPDAGSingleNode.
func TestDAGToPDAGSingleNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")

	p := DAGToPDAG(g)
	if len(p.Nodes()) != 1 {
		t.Fatalf("expected 1 node, got %d", len(p.Nodes()))
	}
	assertPDAGEdges(t, p, nil, nil)
}

// TestDAGToPDAGDisconnected.
func TestDAGToPDAGDisconnected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddNode("C")

	p := DAGToPDAG(g)
	if len(p.Nodes()) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(p.Nodes()))
	}
	assertPDAGEdges(t, p,
		nil,
		[][2]string{{"A", "B"}},
	)
}

// TestMeekR1Direct: Specific test for Meek Rule 1.
// Setup: w→u—v, w not adjacent to v. Should orient u→v.
func TestMeekR1Direct(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("W", "U", "V")
	p.AddDirectedEdge("W", "U")   // w→u
	p.AddUndirectedEdge("U", "V") // u—v
	// w not adjacent to v.

	changed := ApplyMeekRules(p)
	if !changed {
		t.Fatal("expected changes")
	}
	if !p.HasDirectedEdge("U", "V") {
		t.Fatal("expected U→V after R1")
	}
	if p.HasUndirectedEdge("U", "V") {
		t.Fatal("U—V should no longer be undirected")
	}
}

// TestMeekR1NoChange: w→u—v but w adjacent to v. R1 should not fire.
func TestMeekR1NoChange(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("W", "U", "V")
	p.AddDirectedEdge("W", "U")
	p.AddUndirectedEdge("U", "V")
	p.AddUndirectedEdge("W", "V") // w adjacent to v.

	changed := ApplyMeekRules(p)
	if changed {
		t.Fatal("expected no changes since w is adjacent to v")
	}
	if !p.HasUndirectedEdge("U", "V") {
		t.Fatal("U—V should remain undirected")
	}
}

// TestMeekR2Direct: u→w→v and u—v. Should orient u→v.
func TestMeekR2Direct(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("U", "W", "V")
	p.AddDirectedEdge("U", "W")
	p.AddDirectedEdge("W", "V")
	p.AddUndirectedEdge("U", "V")

	changed := ApplyMeekRules(p)
	if !changed {
		t.Fatal("expected changes")
	}
	if !p.HasDirectedEdge("U", "V") {
		t.Fatal("expected U→V after R2")
	}
}

// TestMeekR3Direct: w1—u, w2—u, w1→v, w2→v, w1 not adj w2, u—v.
// Should orient u→v.
func TestMeekR3Direct(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("U", "V", "W1", "W2")
	p.AddUndirectedEdge("W1", "U")
	p.AddUndirectedEdge("W2", "U")
	p.AddDirectedEdge("W1", "V")
	p.AddDirectedEdge("W2", "V")
	p.AddUndirectedEdge("U", "V")
	// W1 and W2 are NOT adjacent.

	changed := ApplyMeekRules(p)
	if !changed {
		t.Fatal("expected changes")
	}
	if !p.HasDirectedEdge("U", "V") {
		t.Fatal("expected U→V after R3")
	}
}

// TestMeekR3NoChange: Same as R3 but w1 adj w2. Should not fire.
func TestMeekR3NoChange(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("U", "V", "W1", "W2")
	p.AddUndirectedEdge("W1", "U")
	p.AddUndirectedEdge("W2", "U")
	p.AddDirectedEdge("W1", "V")
	p.AddDirectedEdge("W2", "V")
	p.AddUndirectedEdge("U", "V")
	p.AddUndirectedEdge("W1", "W2") // w1 adj w2 — R3 should not fire.

	// R1 might fire though: w1→v and w1 not adj ... let's check.
	// For U—V: R1 needs w→U where w not adj V. W1→? No, W1 is undirected to U, not directed.
	// No directed edges into U. R1 won't fire on U—V.
	// For W1—U: R1 needs w→W1 where w not adj U. W2→? No directed edges into W1.
	// Same for W2—U, W1—W2.
	// R2: need U→w→V. No directed from U. Nope.
	// So no rules should fire.
	changed := ApplyMeekRules(p)
	if changed {
		t.Fatal("expected no changes when w1 adj w2")
	}
}

// TestMeekR4Direct: w—u, w→x→v, u—v. Should orient u→v.
// We need to ensure R1 does not fire first in the wrong direction.
// R1 would fire on U—V if there exists z→V where z not adj U. So we make
// X adjacent to U to block R1 from orienting V→U.
func TestMeekR4Direct(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("U", "V", "W", "X")
	p.AddUndirectedEdge("W", "U") // w—u
	p.AddDirectedEdge("W", "X")   // w→x
	p.AddDirectedEdge("X", "V")   // x→v
	p.AddUndirectedEdge("U", "V") // u—v
	p.AddUndirectedEdge("X", "U") // make X adj U so R1 doesn't orient V→U via X→V

	changed := ApplyMeekRules(p)
	if !changed {
		t.Fatal("expected changes")
	}
	if !p.HasDirectedEdge("U", "V") {
		t.Fatal("expected U→V after R4")
	}
}

// TestApplyMeekRulesNoChanges: graph with no undirected edges.
func TestApplyMeekRulesNoChanges(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddDirectedEdge("B", "C")

	changed := ApplyMeekRules(p)
	if changed {
		t.Fatal("expected no changes on fully directed graph")
	}
}

// TestApplyMeekRulesEmptyGraph.
func TestApplyMeekRulesEmptyGraph(t *testing.T) {
	p := NewPDAG()
	changed := ApplyMeekRules(p)
	if changed {
		t.Fatal("expected no changes on empty graph")
	}
}

// TestDAGToPDAGFork: A←B→C (fork). No v-structure. All edges should be undirected.
func TestDAGToPDAGFork(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("B", "A")
	g.AddEdge("B", "C")

	p := DAGToPDAG(g)
	assertPDAGEdges(t, p,
		nil,
		[][2]string{{"A", "B"}, {"B", "C"}},
	)
}

// TestDAGToPDAGTriangle: A→B, A→C, B→C. No v-structure (all pairs adjacent).
func TestDAGToPDAGTriangle(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "C")

	p := DAGToPDAG(g)

	// In triangle A→B→C, A→C: parents of C are A and B, but A adj B, so no v-structure.
	// Parents of B: only A. No v-structures anywhere.
	// All edges should be undirected.
	assertPDAGEdges(t, p,
		nil,
		[][2]string{{"A", "B"}, {"A", "C"}, {"B", "C"}},
	)
}

// TestDAGToPDAGVStructureWithMeekCascade:
// A→C←B, C→D. The v-structure orients A→C and B→C.
// Then C→D stays. Meek R1 applies: C→D, so D's undirected neighbors...
// Actually C→D is already directed from the original. Let's set up:
// DAG: A→C, B→C, C→D. v-structure at C (A,B not adjacent).
// After v-structure: A→C, B→C are directed. C—D starts undirected.
// Meek R1 on C—D: w→C where w not adj D? A→C and A not adj D? If A not adj D, yes.
// So orient C→D.
func TestDAGToPDAGVStructureWithMeekR1(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "C")
	g.AddEdge("B", "C")
	g.AddEdge("C", "D")

	p := DAGToPDAG(g)

	// A→C and B→C are v-structure. C—D should be oriented to C→D by R1
	// (A→C, A not adj D).
	assertPDAGEdges(t, p,
		[][2]string{{"A", "C"}, {"B", "C"}, {"C", "D"}},
		nil,
	)
}

// TestMeekRulesIterative: Rules may need multiple passes.
// A→B—C—D where A not adj C, A not adj D.
// R1 first pass: A→B, A not adj C → orient B→C.
// R1 second pass: B→C, B not adj D → orient C→D.
func TestMeekRulesIterative(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("A", "B", "C", "D")
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("B", "C")
	p.AddUndirectedEdge("C", "D")

	changed := ApplyMeekRules(p)
	if !changed {
		t.Fatal("expected changes")
	}
	if !p.HasDirectedEdge("B", "C") {
		t.Fatal("expected B→C")
	}
	if !p.HasDirectedEdge("C", "D") {
		t.Fatal("expected C→D")
	}
	if len(p.UndirectedEdges()) != 0 {
		t.Fatalf("expected no undirected edges, got %v", p.UndirectedEdges())
	}
}

// TestDAGToPDAGPreservesNodes: all nodes from the DAG appear in the PDAG.
func TestDAGToPDAGPreservesNodes(t *testing.T) {
	g := NewDiGraph()
	g.AddNodes("X", "Y", "Z")
	g.AddEdge("X", "Y")

	p := DAGToPDAG(g)
	nodes := p.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}
}

// TestOrientHelper tests the orient function directly.
func TestOrientHelper(t *testing.T) {
	p := NewPDAG()
	p.AddUndirectedEdge("A", "B")

	ok := orient(p, "A", "B")
	if !ok {
		t.Fatal("orient should return true")
	}
	if !p.HasDirectedEdge("A", "B") {
		t.Fatal("should have A→B")
	}
	if p.HasUndirectedEdge("A", "B") {
		t.Fatal("should not have A—B")
	}

	// Orienting a non-existent undirected edge should return false.
	ok = orient(p, "A", "B")
	if ok {
		t.Fatal("orient should return false for already-oriented edge")
	}
}
