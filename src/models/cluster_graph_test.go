//go:build unit

package models

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func TestNewClusterGraph(t *testing.T) {
	cg := NewClusterGraph()
	if cg == nil {
		t.Fatal("NewClusterGraph returned nil")
	}
	if len(cg.Clusters()) != 0 {
		t.Errorf("expected 0 clusters, got %d", len(cg.Clusters()))
	}
	if len(cg.Edges()) != 0 {
		t.Errorf("expected 0 edges, got %d", len(cg.Edges()))
	}
}

func TestClusterGraphAddCluster(t *testing.T) {
	cg := NewClusterGraph()

	f1, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})

	idx := cg.AddCluster([]string{"B", "A"}, []*factors.DiscreteFactor{f1})
	if idx != 0 {
		t.Errorf("expected cluster index 0, got %d", idx)
	}

	clusters := cg.Clusters()
	if len(clusters) != 1 {
		t.Fatalf("expected 1 cluster, got %d", len(clusters))
	}

	// Variables should be sorted.
	if clusters[0].Variables[0] != "A" || clusters[0].Variables[1] != "B" {
		t.Errorf("expected sorted variables [A B], got %v", clusters[0].Variables)
	}
	if len(clusters[0].Factors) != 1 {
		t.Errorf("expected 1 factor, got %d", len(clusters[0].Factors))
	}
}

func TestClusterGraphAddMultipleClusters(t *testing.T) {
	cg := NewClusterGraph()

	idx0 := cg.AddCluster([]string{"A", "B"}, nil)
	idx1 := cg.AddCluster([]string{"B", "C"}, nil)
	idx2 := cg.AddCluster([]string{"C", "D"}, nil)

	if idx0 != 0 || idx1 != 1 || idx2 != 2 {
		t.Errorf("expected indices 0, 1, 2; got %d, %d, %d", idx0, idx1, idx2)
	}

	if len(cg.Clusters()) != 3 {
		t.Errorf("expected 3 clusters, got %d", len(cg.Clusters()))
	}
}

func TestClusterGraphAddEdge(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)

	err := cg.AddEdge(0, 1, []string{"B"})
	if err != nil {
		t.Fatalf("AddEdge: %v", err)
	}

	edges := cg.Edges()
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].Cluster1 != 0 || edges[0].Cluster2 != 1 {
		t.Errorf("expected edge (0, 1), got (%d, %d)", edges[0].Cluster1, edges[0].Cluster2)
	}
	if len(edges[0].SepSet) != 1 || edges[0].SepSet[0] != "B" {
		t.Errorf("expected sep set [B], got %v", edges[0].SepSet)
	}
}

func TestClusterGraphAddEdgeErrors(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)

	// Invalid cluster indices.
	if err := cg.AddEdge(-1, 1, nil); err == nil {
		t.Error("expected error for negative cluster index")
	}
	if err := cg.AddEdge(0, 5, nil); err == nil {
		t.Error("expected error for out-of-range cluster index")
	}

	// Self-loop.
	if err := cg.AddEdge(0, 0, nil); err == nil {
		t.Error("expected error for self-loop")
	}
}

func TestClusterGraphCheckModel(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)
	_ = cg.AddEdge(0, 1, []string{"B"})

	if err := cg.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestClusterGraphCheckModelNoClusters(t *testing.T) {
	cg := NewClusterGraph()
	if err := cg.CheckModel(); err == nil {
		t.Error("expected error for empty cluster graph")
	}
}

func TestClusterGraphCheckModelInvalidSepSet(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"C", "D"}, nil)

	// Sep set contains "X" which is not in the intersection of the two clusters.
	_ = cg.AddEdge(0, 1, []string{"X"})

	if err := cg.CheckModel(); err == nil {
		t.Error("expected error for invalid sep set")
	}
}

func TestClusterGraphCheckModelSepSetNotInIntersection(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)

	// "A" is in cluster 0 but not cluster 1 => not in intersection.
	_ = cg.AddEdge(0, 1, []string{"A"})

	if err := cg.CheckModel(); err == nil {
		t.Error("expected error for sep set variable not in intersection")
	}
}

func TestClusterGraphCheckModelDuplicateEdge(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)

	_ = cg.AddEdge(0, 1, []string{"B"})
	_ = cg.AddEdge(1, 0, []string{"B"}) // Same edge, reversed order.

	if err := cg.CheckModel(); err == nil {
		t.Error("expected error for duplicate edge")
	}
}

func TestClusterGraphCheckModelEmptySepSet(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)

	// Empty sep set is valid (it's a subset of any set).
	_ = cg.AddEdge(0, 1, nil)

	if err := cg.CheckModel(); err != nil {
		t.Fatalf("CheckModel with empty sep set: %v", err)
	}
}

func TestClusterGraphClustersCopy(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)

	clusters := cg.Clusters()
	clusters[0].Variables[0] = "MODIFIED"

	// Original should be unaffected.
	clusters2 := cg.Clusters()
	if clusters2[0].Variables[0] != "A" {
		t.Error("Clusters() did not return a deep copy")
	}
}

func TestClusterGraphEdgesCopy(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A", "B"}, nil)
	cg.AddCluster([]string{"B", "C"}, nil)
	_ = cg.AddEdge(0, 1, []string{"B"})

	edges := cg.Edges()
	edges[0].SepSet[0] = "MODIFIED"

	// Original should be unaffected.
	edges2 := cg.Edges()
	if edges2[0].SepSet[0] != "B" {
		t.Error("Edges() did not return a deep copy")
	}
}

func TestClusterGraphWithFactors(t *testing.T) {
	cg := NewClusterGraph()

	f1, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	f2, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 3}, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})

	cg.AddCluster([]string{"A", "B"}, []*factors.DiscreteFactor{f1})
	cg.AddCluster([]string{"B", "C"}, []*factors.DiscreteFactor{f2})
	_ = cg.AddEdge(0, 1, []string{"B"})

	if err := cg.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}

	clusters := cg.Clusters()
	if len(clusters[0].Factors) != 1 {
		t.Errorf("cluster 0: expected 1 factor, got %d", len(clusters[0].Factors))
	}
	if len(clusters[1].Factors) != 1 {
		t.Errorf("cluster 1: expected 1 factor, got %d", len(clusters[1].Factors))
	}
}

func TestClusterGraphNoEdges(t *testing.T) {
	cg := NewClusterGraph()
	cg.AddCluster([]string{"A"}, nil)
	cg.AddCluster([]string{"B"}, nil)

	// No edges is valid.
	if err := cg.CheckModel(); err != nil {
		t.Fatalf("CheckModel with no edges: %v", err)
	}
}
