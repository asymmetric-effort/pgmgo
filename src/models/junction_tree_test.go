//go:build unit

package models

import (
	"sort"
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func TestNewJunctionTreeFromBN(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}
	if jt == nil {
		t.Fatal("NewJunctionTreeFromBN returned nil")
	}

	cliques := jt.Cliques()
	if len(cliques) == 0 {
		t.Fatal("expected at least one clique")
	}

	// The student network has 5 variables. Every variable must appear in at
	// least one clique.
	allVars := make(map[string]bool)
	for _, c := range cliques {
		for _, v := range c {
			allVars[v] = true
		}
	}
	for _, v := range []string{"D", "G", "I", "L", "S"} {
		if !allVars[v] {
			t.Errorf("variable %q not found in any clique", v)
		}
	}
}

func TestJunctionTreeCliquesContainFactorScopes(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	// Every factor's scope must be a subset of some clique.
	markovFactors, _ := bn.ToMarkovFactors()
	cliques := jt.Cliques()

	for _, f := range markovFactors {
		vars := f.Variables()
		found := false
		for _, c := range cliques {
			cSet := make(map[string]bool, len(c))
			for _, v := range c {
				cSet[v] = true
			}
			allIn := true
			for _, v := range vars {
				if !cSet[v] {
					allIn = false
					break
				}
			}
			if allIn {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("factor with variables %v is not contained in any clique", vars)
		}
	}
}

func TestJunctionTreeCheckModel(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	// The running intersection property should hold.
	if err := jt.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestJunctionTreeSeparatorSets(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	seps := jt.SeparatorSets()
	cliques := jt.Cliques()

	// Number of edges in a tree = number of nodes - 1.
	if len(cliques) > 1 && len(seps) != len(cliques)-1 {
		t.Errorf("expected %d separator sets, got %d", len(cliques)-1, len(seps))
	}

	// Each separator set must be non-empty (for a connected tree with >1 cliques).
	for k, sep := range seps {
		if len(sep) == 0 {
			t.Errorf("separator set for edge %q is empty", k)
		}
	}
}

func TestJunctionTreeGetCliqueFactors(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	// Every factor must be assigned to exactly one clique.
	totalFactors := 0
	for _, c := range jt.Cliques() {
		fs := jt.GetCliqueFactors(c)
		totalFactors += len(fs)
	}
	if totalFactors != 5 {
		t.Errorf("expected 5 total assigned factors, got %d", totalFactors)
	}
}

func TestJunctionTreeGetCliqueFactorsNonexistent(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	// A clique that doesn't exist should return nil.
	fs := jt.GetCliqueFactors([]string{"X", "Y", "Z"})
	if fs != nil {
		t.Errorf("expected nil for nonexistent clique, got %v", fs)
	}
}

func TestJunctionTreeFromInvalidBN(t *testing.T) {
	bn := buildStudentNetwork(t)
	bn.RemoveCPD("G") // Make the model invalid.

	_, err := NewJunctionTreeFromBN(bn)
	if err == nil {
		t.Error("expected error from NewJunctionTreeFromBN with invalid BN")
	}
}

func TestJunctionTreeEmptyBN(t *testing.T) {
	bn := NewBayesianNetwork()
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN on empty BN: %v", err)
	}
	if len(jt.Cliques()) != 0 {
		t.Errorf("expected 0 cliques for empty BN, got %d", len(jt.Cliques()))
	}
	if err := jt.CheckModel(); err != nil {
		t.Fatalf("CheckModel on empty JT: %v", err)
	}
}

func TestJunctionTreeSingleNode(t *testing.T) {
	bn := NewBayesianNetwork()
	_ = bn.AddNode("X")
	cpd, _ := buildSingleNodeCPD("X", 2, []float64{0.4, 0.6})
	_ = bn.AddCPD(cpd)

	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	cliques := jt.Cliques()
	if len(cliques) != 1 {
		t.Fatalf("expected 1 clique, got %d", len(cliques))
	}
	if len(cliques[0]) != 1 || cliques[0][0] != "X" {
		t.Errorf("expected clique [X], got %v", cliques[0])
	}
	if err := jt.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestJunctionTreeCliquesAreSorted(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	for i, c := range jt.Cliques() {
		if !sort.StringsAreSorted(c) {
			t.Errorf("clique %d is not sorted: %v", i, c)
		}
	}
}

func TestJunctionTreeGetCliqueFactorsOrderIndependent(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}

	// Get a clique and reverse it; GetCliqueFactors should still work.
	cliques := jt.Cliques()
	if len(cliques) == 0 {
		t.Skip("no cliques to test")
	}
	original := cliques[0]
	reversed := make([]string, len(original))
	for i, v := range original {
		reversed[len(original)-1-i] = v
	}

	fs1 := jt.GetCliqueFactors(original)
	fs2 := jt.GetCliqueFactors(reversed)
	if len(fs1) != len(fs2) {
		t.Errorf("different factor counts for same clique in different orders: %d vs %d", len(fs1), len(fs2))
	}
}

// buildSingleNodeCPD is a helper for creating a CPD for a node with no parents.
func buildSingleNodeCPD(variable string, card int, probs []float64) (*factors.TabularCPD, error) {
	rows := make([][]float64, card)
	for i := 0; i < card; i++ {
		rows[i] = []float64{probs[i]}
	}
	return factors.NewTabularCPD(variable, card, rows, nil, nil)
}

func TestJunctionTreeCheckModelSingleClique(t *testing.T) {
	// A single-clique junction tree trivially satisfies RIP.
	bn := NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	cpdA, _ := buildSingleNodeCPD("A", 2, []float64{0.5, 0.5})
	_ = bn.AddCPD(cpdA)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{
		{0.3, 0.7},
		{0.7, 0.3},
	}, []string{"A"}, []int{2})
	_ = bn.AddCPD(cpdB)

	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}
	if err := jt.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestMinDegreeOrder(t *testing.T) {
	// Test that minDegreeOrder returns all nodes exactly once.
	bn := buildStudentNetwork(t)
	dg := graphgo.NewDiGraph()
	for _, node := range bn.Nodes() {
		dg.AddNode(node)
	}
	for _, e := range bn.Edges() {
		dg.AddEdge(e[0], e[1])
	}
	moral := graphgo.Moralize(dg)

	order := minDegreeOrder(moral)
	nodes := moral.Nodes()
	sort.Strings(nodes)
	sort.Strings(order)

	if strings.Join(order, ",") != strings.Join(nodes, ",") {
		t.Errorf("minDegreeOrder returned %v, expected all nodes %v", order, nodes)
	}
}
