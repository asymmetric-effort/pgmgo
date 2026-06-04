//go:build unit

package readwrite_test

import (
	"bytes"
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/readwrite"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

func TestCrossval_BIFRoundtrip(t *testing.T) {
	ff := testutil.LoadFixtures(t, "readwrite/fixtures.json")
	tc := ff.FindTestCase(t, "bif_roundtrip")

	var input struct {
		BIFContent    string     `json:"bif_content"`
		OriginalNodes []string   `json:"original_nodes"`
		OriginalEdges [][]string `json:"original_edges"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Nodes    []string   `json:"nodes"`
		Edges    [][]string `json:"edges"`
		NumNodes int        `json:"num_nodes"`
		NumEdges int        `json:"num_edges"`
		NumCPDs  int        `json:"num_cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Read from the BIF content provided by pgmpy
	r := bytes.NewBufferString(input.BIFContent)
	bn, err := readwrite.ReadBIF(r)
	if err != nil {
		t.Fatalf("ReadBIF: %v", err)
	}

	// Verify node count
	gotNodes := bn.Nodes()
	sort.Strings(gotNodes)
	if len(gotNodes) != expected.NumNodes {
		t.Errorf("num_nodes: expected %d, got %d", expected.NumNodes, len(gotNodes))
	}

	// Verify nodes match
	for i := range expected.Nodes {
		if i >= len(gotNodes) {
			t.Errorf("missing node %q", expected.Nodes[i])
			continue
		}
		if gotNodes[i] != expected.Nodes[i] {
			t.Errorf("node[%d]: expected %q, got %q", i, expected.Nodes[i], gotNodes[i])
		}
	}

	// Verify edge count
	gotEdges := bn.Edges()
	if len(gotEdges) != expected.NumEdges {
		t.Errorf("num_edges: expected %d, got %d", expected.NumEdges, len(gotEdges))
	}

	// Write back to BIF and read again to verify roundtrip
	var buf bytes.Buffer
	if err := readwrite.WriteBIF(&buf, bn); err != nil {
		t.Fatalf("WriteBIF: %v", err)
	}

	bn2, err := readwrite.ReadBIF(&buf)
	if err != nil {
		t.Fatalf("ReadBIF roundtrip: %v", err)
	}

	gotNodes2 := bn2.Nodes()
	sort.Strings(gotNodes2)
	if len(gotNodes2) != expected.NumNodes {
		t.Errorf("roundtrip num_nodes: expected %d, got %d", expected.NumNodes, len(gotNodes2))
	}
}

func TestCrossval_XMLBIFRoundtrip(t *testing.T) {
	ff := testutil.LoadFixtures(t, "readwrite/fixtures.json")
	tc := ff.FindTestCase(t, "xmlbif_roundtrip")

	var input struct {
		XMLBIFContent string     `json:"xmlbif_content"`
		OriginalNodes []string   `json:"original_nodes"`
		OriginalEdges [][]string `json:"original_edges"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Nodes    []string   `json:"nodes"`
		Edges    [][]string `json:"edges"`
		NumNodes int        `json:"num_nodes"`
		NumEdges int        `json:"num_edges"`
		NumCPDs  int        `json:"num_cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Read from the XMLBIF content provided by pgmpy
	r := bytes.NewBufferString(input.XMLBIFContent)
	bn, err := readwrite.ReadXMLBIF(r)
	if err != nil {
		t.Fatalf("ReadXMLBIF: %v", err)
	}

	// Verify node count
	gotNodes := bn.Nodes()
	sort.Strings(gotNodes)
	if len(gotNodes) != expected.NumNodes {
		t.Errorf("num_nodes: expected %d, got %d", expected.NumNodes, len(gotNodes))
	}

	// Verify nodes match
	for i := range expected.Nodes {
		if i >= len(gotNodes) {
			t.Errorf("missing node %q", expected.Nodes[i])
			continue
		}
		if gotNodes[i] != expected.Nodes[i] {
			t.Errorf("node[%d]: expected %q, got %q", i, expected.Nodes[i], gotNodes[i])
		}
	}

	// Verify edge count
	gotEdges := bn.Edges()
	if len(gotEdges) != expected.NumEdges {
		t.Errorf("num_edges: expected %d, got %d", expected.NumEdges, len(gotEdges))
	}

	// Write back and read again
	var buf bytes.Buffer
	if err := readwrite.WriteXMLBIF(&buf, bn); err != nil {
		t.Fatalf("WriteXMLBIF: %v", err)
	}

	bn2, err := readwrite.ReadXMLBIF(&buf)
	if err != nil {
		t.Fatalf("ReadXMLBIF roundtrip: %v", err)
	}

	gotNodes2 := bn2.Nodes()
	sort.Strings(gotNodes2)
	if len(gotNodes2) != expected.NumNodes {
		t.Errorf("roundtrip num_nodes: expected %d, got %d", expected.NumNodes, len(gotNodes2))
	}
}
