//go:build unit

package graphgo

import (
	"sort"
	"testing"
)

func sortComponents(comps [][]string) {
	for _, c := range comps {
		sort.Strings(c)
	}
	sort.Slice(comps, func(i, j int) bool {
		if len(comps[i]) != len(comps[j]) {
			return len(comps[i]) < len(comps[j])
		}
		return comps[i][0] < comps[j][0]
	})
}

func TestWeaklyConnectedComponents_Connected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	comps := WeaklyConnectedComponents(g)
	if len(comps) != 1 {
		t.Errorf("expected 1 component, got %d", len(comps))
	}
	sort.Strings(comps[0])
	expected := []string{"A", "B", "C"}
	for i, n := range expected {
		if comps[0][i] != n {
			t.Errorf("component[%d] = %q, want %q", i, comps[0][i], n)
		}
	}
}

func TestWeaklyConnectedComponents_Disconnected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddNode("C")
	comps := WeaklyConnectedComponents(g)
	if len(comps) != 2 {
		t.Errorf("expected 2 components, got %d", len(comps))
	}
}

func TestStronglyConnectedComponents_Cycle(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	comps := StronglyConnectedComponents(g)
	if len(comps) != 1 {
		t.Errorf("expected 1 SCC, got %d", len(comps))
	}
	sort.Strings(comps[0])
	expected := []string{"A", "B", "C"}
	for i, n := range expected {
		if comps[0][i] != n {
			t.Errorf("SCC[%d] = %q, want %q", i, comps[0][i], n)
		}
	}
}

func TestStronglyConnectedComponents_DAG(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	comps := StronglyConnectedComponents(g)
	if len(comps) != 3 {
		t.Errorf("expected 3 SCCs, got %d", len(comps))
	}
}

func TestIsWeaklyConnected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	if !IsWeaklyConnected(g) {
		t.Error("expected weakly connected")
	}
	g.AddNode("D")
	if IsWeaklyConnected(g) {
		t.Error("expected not weakly connected")
	}
}

func TestIsWeaklyConnected_Empty(t *testing.T) {
	g := NewDiGraph()
	if IsWeaklyConnected(g) {
		t.Error("empty graph should not be weakly connected")
	}
}

func TestIsStronglyConnected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	if !IsStronglyConnected(g) {
		t.Error("expected strongly connected")
	}
}

func TestIsStronglyConnected_False(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	if IsStronglyConnected(g) {
		t.Error("expected not strongly connected")
	}
}

func TestIsStronglyConnected_Empty(t *testing.T) {
	g := NewDiGraph()
	if IsStronglyConnected(g) {
		t.Error("empty graph should not be strongly connected")
	}
}

func TestGraphIsConnected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	if !g.IsConnected() {
		t.Error("expected connected")
	}
	g.AddNode("D")
	if g.IsConnected() {
		t.Error("expected not connected")
	}
}

func TestGraphIsConnected_Empty(t *testing.T) {
	g := NewGraph()
	if g.IsConnected() {
		t.Error("empty graph should not be connected")
	}
}

func TestGraphConnectedComponents(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("C", "D")
	comps := g.ConnectedComponents()
	if len(comps) != 2 {
		t.Errorf("expected 2 components, got %d", len(comps))
	}
}

func TestGraphConnectedComponents_Single(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	comps := g.ConnectedComponents()
	if len(comps) != 1 {
		t.Errorf("expected 1 component, got %d", len(comps))
	}
}
