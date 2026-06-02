//go:build unit

package metrics_test

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/src/metrics"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildDigraph constructs a DiGraph from node and edge lists in fixture data.
func buildDigraph(t *testing.T, nodes []string, edges [][]string) *graphgo.DiGraph {
	t.Helper()
	g := graphgo.NewDiGraph()
	for _, n := range nodes {
		g.AddNode(n)
	}
	for _, e := range edges {
		if len(e) != 2 {
			t.Fatalf("expected edge pair, got %v", e)
		}
		g.AddEdge(e[0], e[1])
	}
	return g
}

func TestCrossval_SHD_OneExtraEdge(t *testing.T) {
	ff := testutil.LoadFixtures(t, "metrics/fixtures.json")
	tc := ff.FindTestCase(t, "shd_one_extra_edge")

	var input struct {
		TrueEdges      [][]string `json:"true_edges"`
		EstimatedEdges [][]string `json:"estimated_edges"`
		TrueNodes      []string   `json:"true_nodes"`
		EstimatedNodes []string   `json:"estimated_nodes"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		SHD int `json:"shd"`
	}
	tc.UnmarshalExpected(t, &expected)

	trueG := buildDigraph(t, input.TrueNodes, input.TrueEdges)
	estG := buildDigraph(t, input.EstimatedNodes, input.EstimatedEdges)

	got := metrics.SHD(trueG, estG)
	if got != expected.SHD {
		t.Errorf("SHD: expected %d, got %d", expected.SHD, got)
	}
}

func TestCrossval_SHD_Reversal(t *testing.T) {
	ff := testutil.LoadFixtures(t, "metrics/fixtures.json")
	tc := ff.FindTestCase(t, "shd_reversal")

	var input struct {
		TrueEdges      [][]string `json:"true_edges"`
		EstimatedEdges [][]string `json:"estimated_edges"`
		TrueNodes      []string   `json:"true_nodes"`
		EstimatedNodes []string   `json:"estimated_nodes"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		SHD int `json:"shd"`
	}
	tc.UnmarshalExpected(t, &expected)

	trueG := buildDigraph(t, input.TrueNodes, input.TrueEdges)
	estG := buildDigraph(t, input.EstimatedNodes, input.EstimatedEdges)

	got := metrics.SHD(trueG, estG)
	if got != expected.SHD {
		t.Errorf("SHD: expected %d, got %d", expected.SHD, got)
	}
}

func TestCrossval_AdjacencyConfusion(t *testing.T) {
	ff := testutil.LoadFixtures(t, "metrics/fixtures.json")
	tc := ff.FindTestCase(t, "adjacency_confusion")

	var input struct {
		TrueEdges      [][]string `json:"true_edges"`
		EstimatedEdges [][]string `json:"estimated_edges"`
		TrueNodes      []string   `json:"true_nodes"`
		EstimatedNodes []string   `json:"estimated_nodes"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		TP int `json:"tp"`
		FP int `json:"fp"`
		TN int `json:"tn"`
		FN int `json:"fn"`
	}
	tc.UnmarshalExpected(t, &expected)

	trueG := buildDigraph(t, input.TrueNodes, input.TrueEdges)
	estG := buildDigraph(t, input.EstimatedNodes, input.EstimatedEdges)

	tp, fp, tn, fn := metrics.AdjacencyConfusionMatrix(trueG, estG)
	if tp != expected.TP {
		t.Errorf("TP: expected %d, got %d", expected.TP, tp)
	}
	if fp != expected.FP {
		t.Errorf("FP: expected %d, got %d", expected.FP, fp)
	}
	if tn != expected.TN {
		t.Errorf("TN: expected %d, got %d", expected.TN, tn)
	}
	if fn != expected.FN {
		t.Errorf("FN: expected %d, got %d", expected.FN, fn)
	}
}

func TestCrossval_OrientationConfusion(t *testing.T) {
	ff := testutil.LoadFixtures(t, "metrics/fixtures.json")
	tc := ff.FindTestCase(t, "orientation_confusion")

	var input struct {
		TrueEdges      [][]string `json:"true_edges"`
		EstimatedEdges [][]string `json:"estimated_edges"`
		TrueNodes      []string   `json:"true_nodes"`
		EstimatedNodes []string   `json:"estimated_nodes"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		TP int `json:"tp"`
		FP int `json:"fp"`
		TN int `json:"tn"`
		FN int `json:"fn"`
	}
	tc.UnmarshalExpected(t, &expected)

	trueG := buildDigraph(t, input.TrueNodes, input.TrueEdges)
	estG := buildDigraph(t, input.EstimatedNodes, input.EstimatedEdges)

	tp, fp, tn, fn := metrics.OrientationConfusionMatrix(trueG, estG)
	if tp != expected.TP {
		t.Errorf("TP: expected %d, got %d", expected.TP, tp)
	}
	if fp != expected.FP {
		t.Errorf("FP: expected %d, got %d", expected.FP, fp)
	}
	if tn != expected.TN {
		t.Errorf("TN: expected %d, got %d", expected.TN, tn)
	}
	if fn != expected.FN {
		t.Errorf("FN: expected %d, got %d", expected.FN, fn)
	}
}
