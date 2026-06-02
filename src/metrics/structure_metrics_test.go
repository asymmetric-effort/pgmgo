//go:build unit

package metrics

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

func TestSHD_IdenticalGraphs(t *testing.T) {
	g := graphgo.NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	got := SHD(g, g)
	if got != 0 {
		t.Errorf("SHD(identical) = %d, want 0", got)
	}
}

func TestSHD_OneExtraEdge(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")
	trueG.AddEdge("B", "C")

	est := graphgo.NewDiGraph()
	est.AddEdge("A", "B")
	est.AddEdge("B", "C")
	est.AddEdge("A", "C") // extra

	got := SHD(trueG, est)
	if got != 1 {
		t.Errorf("SHD(one extra edge) = %d, want 1", got)
	}
}

func TestSHD_OneMissingEdge(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")
	trueG.AddEdge("B", "C")
	trueG.AddEdge("A", "C")

	est := graphgo.NewDiGraph()
	est.AddEdge("A", "B")
	est.AddEdge("B", "C")
	// missing A->C

	got := SHD(trueG, est)
	if got != 1 {
		t.Errorf("SHD(one missing edge) = %d, want 1", got)
	}
}

func TestSHD_OneReversedEdge(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")
	trueG.AddEdge("B", "C")

	est := graphgo.NewDiGraph()
	est.AddEdge("B", "A") // reversed
	est.AddEdge("B", "C")

	got := SHD(trueG, est)
	if got != 1 {
		t.Errorf("SHD(one reversed edge) = %d, want 1", got)
	}
}

func TestSHD_EmptyGraphs(t *testing.T) {
	g1 := graphgo.NewDiGraph()
	g2 := graphgo.NewDiGraph()
	got := SHD(g1, g2)
	if got != 0 {
		t.Errorf("SHD(empty, empty) = %d, want 0", got)
	}
}

func TestSHD_OneExtraAndOneReversed(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")
	trueG.AddEdge("B", "C")

	est := graphgo.NewDiGraph()
	est.AddEdge("B", "A") // reversed
	est.AddEdge("B", "C")
	est.AddEdge("C", "D") // extra (new node too)

	got := SHD(trueG, est)
	if got != 2 {
		t.Errorf("SHD(one reversed + one extra) = %d, want 2", got)
	}
}

func TestAdjacencyConfusionMatrix_Identical(t *testing.T) {
	g := graphgo.NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	tp, fp, tn, fn := AdjacencyConfusionMatrix(g, g)
	if tp != 2 {
		t.Errorf("TP = %d, want 2", tp)
	}
	if fp != 0 {
		t.Errorf("FP = %d, want 0", fp)
	}
	if fn != 0 {
		t.Errorf("FN = %d, want 0", fn)
	}
	// 3 nodes = 3 pairs, 2 are edges, 1 is TN
	if tn != 1 {
		t.Errorf("TN = %d, want 1", tn)
	}
}

func TestAdjacencyConfusionMatrix_ExtraEdge(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")
	trueG.AddNodes("C")

	est := graphgo.NewDiGraph()
	est.AddEdge("A", "B")
	est.AddEdge("A", "C")

	tp, fp, tn, fn := AdjacencyConfusionMatrix(trueG, est)
	if tp != 1 {
		t.Errorf("TP = %d, want 1", tp)
	}
	if fp != 1 {
		t.Errorf("FP = %d, want 1", fp)
	}
	if tn != 1 {
		t.Errorf("TN = %d, want 1", tn)
	}
	if fn != 0 {
		t.Errorf("FN = %d, want 0", fn)
	}
}

func TestAdjacencyConfusionMatrix_ReversedEdgeIsTP(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")

	est := graphgo.NewDiGraph()
	est.AddEdge("B", "A") // reversed, but undirected adjacency is the same

	tp, fp, _, fn := AdjacencyConfusionMatrix(trueG, est)
	if tp != 1 {
		t.Errorf("TP = %d, want 1 (reversed edge should be adjacency TP)", tp)
	}
	if fp != 0 {
		t.Errorf("FP = %d, want 0", fp)
	}
	if fn != 0 {
		t.Errorf("FN = %d, want 0", fn)
	}
}

func TestOrientationConfusionMatrix_Identical(t *testing.T) {
	g := graphgo.NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	tp, fp, tn, fn := OrientationConfusionMatrix(g, g)
	// 2 adjacency TPs, each has 2 directions: A->B present both, B->A absent both, etc.
	if tp != 2 {
		t.Errorf("TP = %d, want 2", tp)
	}
	if fp != 0 {
		t.Errorf("FP = %d, want 0", fp)
	}
	if fn != 0 {
		t.Errorf("FN = %d, want 0", fn)
	}
	if tn != 2 {
		t.Errorf("TN = %d, want 2", tn)
	}
}

func TestOrientationConfusionMatrix_Reversed(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")

	est := graphgo.NewDiGraph()
	est.AddEdge("B", "A")

	tp, fp, tn, fn := OrientationConfusionMatrix(trueG, est)
	// 1 adjacency TP pair {A,B}. Check A->B: true=yes, est=no => FN.
	// Check B->A: true=no, est=yes => FP.
	if tp != 0 {
		t.Errorf("TP = %d, want 0", tp)
	}
	if fp != 1 {
		t.Errorf("FP = %d, want 1", fp)
	}
	if fn != 1 {
		t.Errorf("FN = %d, want 1", fn)
	}
	if tn != 0 {
		t.Errorf("TN = %d, want 0", tn)
	}
}

func TestOrientationConfusionMatrix_NoAdjacencyOverlap(t *testing.T) {
	trueG := graphgo.NewDiGraph()
	trueG.AddEdge("A", "B")

	est := graphgo.NewDiGraph()
	est.AddEdge("C", "D")

	tp, fp, tn, fn := OrientationConfusionMatrix(trueG, est)
	// No adjacency TPs, so all should be 0.
	if tp != 0 || fp != 0 || tn != 0 || fn != 0 {
		t.Errorf("Expected all zeros, got tp=%d fp=%d tn=%d fn=%d", tp, fp, tn, fn)
	}
}
