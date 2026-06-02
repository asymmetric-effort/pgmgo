//go:build unit

package models_test

import (
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

func TestCrossval_BayesianNetworkStructure(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models/fixtures.json")
	tc := ff.FindTestCase(t, "bayesian_network_structure")

	var input struct {
		Edges [][]string `json:"edges"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Nodes    []string            `json:"nodes"`
		Edges    [][]string          `json:"edges"`
		NumNodes int                 `json:"num_nodes"`
		NumEdges int                 `json:"num_edges"`
		Parents  map[string][]string `json:"parents"`
		Children map[string][]string `json:"children"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := models.NewBayesianNetwork()

	// Collect all unique nodes from edges and add them.
	nodeSet := make(map[string]bool)
	for _, edge := range input.Edges {
		nodeSet[edge[0]] = true
		nodeSet[edge[1]] = true
	}
	for node := range nodeSet {
		if err := bn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q) failed: %v", node, err)
		}
	}

	for _, edge := range input.Edges {
		if err := bn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q) failed: %v", edge[0], edge[1], err)
		}
	}

	// Verify nodes.
	gotNodes := bn.Nodes()
	sort.Strings(gotNodes)
	expectedNodes := make([]string, len(expected.Nodes))
	copy(expectedNodes, expected.Nodes)
	sort.Strings(expectedNodes)

	if len(gotNodes) != len(expectedNodes) {
		t.Fatalf("nodes: expected %v, got %v", expectedNodes, gotNodes)
	}
	for i := range expectedNodes {
		if gotNodes[i] != expectedNodes[i] {
			t.Errorf("nodes[%d]: expected %q, got %q", i, expectedNodes[i], gotNodes[i])
		}
	}

	if len(gotNodes) != expected.NumNodes {
		t.Errorf("num_nodes: expected %d, got %d", expected.NumNodes, len(gotNodes))
	}

	// Verify edges.
	gotEdges := bn.Edges()
	if len(gotEdges) != expected.NumEdges {
		t.Errorf("num_edges: expected %d, got %d", expected.NumEdges, len(gotEdges))
	}

	// Verify parents for each node.
	for node, expectedParents := range expected.Parents {
		gotParents := bn.Parents(node)
		sort.Strings(gotParents)
		ep := make([]string, len(expectedParents))
		copy(ep, expectedParents)
		sort.Strings(ep)

		if len(gotParents) != len(ep) {
			t.Errorf("parents(%q): expected %v, got %v", node, ep, gotParents)
			continue
		}
		for i := range ep {
			if gotParents[i] != ep[i] {
				t.Errorf("parents(%q)[%d]: expected %q, got %q", node, i, ep[i], gotParents[i])
			}
		}
	}

	// Verify children for each node.
	for node, expectedChildren := range expected.Children {
		gotChildren := bn.Children(node)
		sort.Strings(gotChildren)
		ec := make([]string, len(expectedChildren))
		copy(ec, expectedChildren)
		sort.Strings(ec)

		if len(gotChildren) != len(ec) {
			t.Errorf("children(%q): expected %v, got %v", node, ec, gotChildren)
			continue
		}
		for i := range ec {
			if gotChildren[i] != ec[i] {
				t.Errorf("children(%q)[%d]: expected %q, got %q", node, i, ec[i], gotChildren[i])
			}
		}
	}
}

func TestCrossval_BayesianNetworkCPDs(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models/fixtures.json")
	tc := ff.FindTestCase(t, "bayesian_network_cpds")

	var input struct {
		Edges [][]string `json:"edges"`
		CPDs  map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		IsValid bool     `json:"is_valid"`
		Nodes   []string `json:"nodes"`
		NumCPDs int      `json:"num_cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := models.NewBayesianNetwork()

	// Add all nodes from edges.
	nodeSet := make(map[string]bool)
	for _, edge := range input.Edges {
		nodeSet[edge[0]] = true
		nodeSet[edge[1]] = true
	}
	for node := range nodeSet {
		if err := bn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q) failed: %v", node, err)
		}
	}

	// Add edges.
	for _, edge := range input.Edges {
		if err := bn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q) failed: %v", edge[0], edge[1], err)
		}
	}

	// Add CPDs.
	for varName, cpdData := range input.CPDs {
		evidence := cpdData.Evidence
		if evidence == nil {
			evidence = []string{}
		}
		evidenceCard := cpdData.EvidenceCard
		if evidenceCard == nil {
			evidenceCard = []int{}
		}

		cpd, err := factors.NewTabularCPD(varName, cpdData.VariableCard, cpdData.Values,
			evidence, evidenceCard)
		if err != nil {
			t.Fatalf("NewTabularCPD(%q) failed: %v", varName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q) failed: %v", varName, err)
		}
	}

	// Verify check_model passes.
	err := bn.CheckModel()
	if expected.IsValid {
		if err != nil {
			t.Errorf("CheckModel: expected valid, got error: %v", err)
		}
	} else {
		if err == nil {
			t.Errorf("CheckModel: expected invalid, got nil error")
		}
	}

	// Verify node count.
	gotNodes := bn.Nodes()
	sort.Strings(gotNodes)
	expectedNodes := make([]string, len(expected.Nodes))
	copy(expectedNodes, expected.Nodes)
	sort.Strings(expectedNodes)

	if len(gotNodes) != len(expectedNodes) {
		t.Fatalf("nodes: expected %v, got %v", expectedNodes, gotNodes)
	}
	for i := range expectedNodes {
		if gotNodes[i] != expectedNodes[i] {
			t.Errorf("nodes[%d]: expected %q, got %q", i, expectedNodes[i], gotNodes[i])
		}
	}

	// Verify CPD count.
	gotCPDs := bn.GetCPDs()
	if len(gotCPDs) != expected.NumCPDs {
		t.Errorf("num_cpds: expected %d, got %d", expected.NumCPDs, len(gotCPDs))
	}
}
