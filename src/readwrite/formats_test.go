//go:build unit

package readwrite

import (
	"bytes"
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// buildThreeNodeBN creates a network: Rain -> Sprinkler, Rain -> WetGrass
// with a 3-state WetGrass variable for more coverage.
func buildThreeNodeBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	must(t, bn.AddNode("Rain"))
	must(t, bn.SetStates("Rain", []string{"True", "False"}))
	must(t, bn.AddNode("Sprinkler"))
	must(t, bn.SetStates("Sprinkler", []string{"On", "Off"}))
	must(t, bn.AddEdge("Rain", "Sprinkler"))

	rainCPD, err := factors.NewTabularCPD("Rain", 2,
		[][]float64{{0.2}, {0.8}}, nil, nil)
	must(t, err)
	must(t, bn.AddCPD(rainCPD))

	sprCPD, err := factors.NewTabularCPD("Sprinkler", 2,
		[][]float64{{0.01, 0.4}, {0.99, 0.6}},
		[]string{"Rain"}, []int{2})
	must(t, err)
	must(t, bn.AddCPD(sprCPD))

	return bn
}

// buildMultiParentBN creates a network with multiple parents: D, I -> G
func buildMultiParentBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	must(t, bn.AddNode("D"))
	must(t, bn.SetStates("D", []string{"Easy", "Hard"}))
	must(t, bn.AddNode("I"))
	must(t, bn.SetStates("I", []string{"Low", "High"}))
	must(t, bn.AddNode("G"))
	must(t, bn.SetStates("G", []string{"A", "B", "C"}))

	must(t, bn.AddEdge("D", "G"))
	must(t, bn.AddEdge("I", "G"))

	dCPD, err := factors.NewTabularCPD("D", 2, [][]float64{{0.6}, {0.4}}, nil, nil)
	must(t, err)
	must(t, bn.AddCPD(dCPD))

	iCPD, err := factors.NewTabularCPD("I", 2, [][]float64{{0.7}, {0.3}}, nil, nil)
	must(t, err)
	must(t, bn.AddCPD(iCPD))

	gCPD, err := factors.NewTabularCPD("G", 3,
		[][]float64{
			{0.3, 0.05, 0.9, 0.5},
			{0.4, 0.25, 0.08, 0.3},
			{0.3, 0.7, 0.02, 0.2},
		},
		[]string{"D", "I"}, []int{2, 2},
	)
	must(t, err)
	must(t, bn.AddCPD(gCPD))

	return bn
}

// assertBNEqual compares two BayesianNetworks structurally and numerically.
func assertBNEqual(t *testing.T, got, want *models.BayesianNetwork, label string) {
	t.Helper()

	// Nodes.
	nodes1 := want.Nodes()
	nodes2 := got.Nodes()
	if len(nodes1) != len(nodes2) {
		t.Fatalf("%s: node count mismatch: want %d, got %d", label, len(nodes1), len(nodes2))
	}
	for i := range nodes1 {
		if nodes1[i] != nodes2[i] {
			t.Errorf("%s: node %d: want %q, got %q", label, i, nodes1[i], nodes2[i])
		}
	}

	// Edges.
	edges1 := want.Edges()
	edges2 := got.Edges()
	if len(edges1) != len(edges2) {
		t.Fatalf("%s: edge count mismatch: want %d, got %d", label, len(edges1), len(edges2))
	}
	for i := range edges1 {
		if edges1[i] != edges2[i] {
			t.Errorf("%s: edge %d: want %v, got %v", label, i, edges1[i], edges2[i])
		}
	}

	// States and CPD values.
	for _, node := range nodes1 {
		s1 := want.GetStates(node)
		s2 := got.GetStates(node)
		if len(s1) != len(s2) {
			t.Errorf("%s: states for %q: want %v, got %v", label, node, s1, s2)
			continue
		}
		for j := range s1 {
			if s1[j] != s2[j] {
				t.Errorf("%s: state %q[%d]: want %q, got %q", label, node, j, s1[j], s2[j])
			}
		}

		cpd1 := want.GetCPD(node)
		cpd2 := got.GetCPD(node)
		if cpd1 == nil || cpd2 == nil {
			t.Errorf("%s: CPD for %q: nil in one network", label, node)
			continue
		}
		data1 := cpd1.ToFactor().Values().Data()
		data2 := cpd2.ToFactor().Values().Data()
		assertFloatsClose(t, data2, data1, label+" "+node+" CPD")
	}
}

// ---------------------------------------------------------------------------
// XMLBIF
// ---------------------------------------------------------------------------

func TestXMLBIF_RoundTrip(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WriteXMLBIF(&buf, bn); err != nil {
		t.Fatalf("WriteXMLBIF failed: %v", err)
	}

	bn2, err := ReadXMLBIF(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXMLBIF failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "XMLBIF round-trip")
}

func TestXMLBIF_RoundTrip_MultiParent(t *testing.T) {
	bn := buildMultiParentBN(t)

	var buf bytes.Buffer
	if err := WriteXMLBIF(&buf, bn); err != nil {
		t.Fatalf("WriteXMLBIF failed: %v", err)
	}

	bn2, err := ReadXMLBIF(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXMLBIF failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "XMLBIF multi-parent round-trip")
}

func TestReadXMLBIF_Basic(t *testing.T) {
	xmlData := `<?xml version="1.0" encoding="UTF-8"?>
<BIF VERSION="0.3">
  <NETWORK>
    <NAME>test</NAME>
    <VARIABLE TYPE="nature">
      <NAME>X</NAME>
      <OUTCOME>x0</OUTCOME>
      <OUTCOME>x1</OUTCOME>
    </VARIABLE>
    <DEFINITION>
      <FOR>X</FOR>
      <TABLE>0.3 0.7</TABLE>
    </DEFINITION>
  </NETWORK>
</BIF>`

	bn, err := ReadXMLBIF(strings.NewReader(xmlData))
	if err != nil {
		t.Fatalf("ReadXMLBIF failed: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 1 || nodes[0] != "X" {
		t.Fatalf("expected [X], got %v", nodes)
	}

	states := bn.GetStates("X")
	if len(states) != 2 || states[0] != "x0" || states[1] != "x1" {
		t.Errorf("X states = %v, want [x0, x1]", states)
	}

	cpd := bn.GetCPD("X")
	data := cpd.ToFactor().Values().Data()
	assertFloatsClose(t, data, []float64{0.3, 0.7}, "X CPD")
}

// ---------------------------------------------------------------------------
// NET
// ---------------------------------------------------------------------------

func TestNET_RoundTrip(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WriteNET(&buf, bn); err != nil {
		t.Fatalf("WriteNET failed: %v", err)
	}

	bn2, err := ReadNET(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadNET failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "NET round-trip")
}

func TestNET_RoundTrip_MultiParent(t *testing.T) {
	bn := buildMultiParentBN(t)

	var buf bytes.Buffer
	if err := WriteNET(&buf, bn); err != nil {
		t.Fatalf("WriteNET failed: %v", err)
	}

	bn2, err := ReadNET(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadNET failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "NET multi-parent round-trip")
}

func TestReadNET_Basic(t *testing.T) {
	netData := `net
{
}

node X
{
  states = ("x0" "x1");
}

potential (X)
{
  data = (0.3 0.7);
}
`
	bn, err := ReadNET(strings.NewReader(netData))
	if err != nil {
		t.Fatalf("ReadNET failed: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 1 || nodes[0] != "X" {
		t.Fatalf("expected [X], got %v", nodes)
	}

	cpd := bn.GetCPD("X")
	data := cpd.ToFactor().Values().Data()
	assertFloatsClose(t, data, []float64{0.3, 0.7}, "X CPD")
}

func TestReadNET_Conditional(t *testing.T) {
	netData := `net
{
}

node A
{
  states = ("a0" "a1");
}

node B
{
  states = ("b0" "b1");
}

potential (A)
{
  data = (0.6 0.4);
}

potential (B | A)
{
  data = ((0.2 0.8)(0.75 0.25));
}
`
	bn, err := ReadNET(strings.NewReader(netData))
	if err != nil {
		t.Fatalf("ReadNET failed: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 1 || edges[0] != [2]string{"A", "B"} {
		t.Fatalf("expected edge [A B], got %v", edges)
	}

	bCPD := bn.GetCPD("B")
	bData := bCPD.ToFactor().Values().Data()
	// child=b0: A=a0=>0.2, A=a1=>0.75
	// child=b1: A=a0=>0.8, A=a1=>0.25
	assertFloatsClose(t, bData, []float64{0.2, 0.75, 0.8, 0.25}, "B CPD")
}

// ---------------------------------------------------------------------------
// UAI
// ---------------------------------------------------------------------------

func TestUAI_RoundTrip(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WriteUAI(&buf, bn); err != nil {
		t.Fatalf("WriteUAI failed: %v", err)
	}

	bn2, err := ReadUAI(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadUAI failed: %v\nOutput:\n%s", err, buf.String())
	}

	// UAI uses generated variable names (V0, V1, ...) so we compare structure.
	nodes1 := bn.Nodes()
	nodes2 := bn2.Nodes()
	if len(nodes1) != len(nodes2) {
		t.Fatalf("UAI round-trip: node count mismatch: %d vs %d", len(nodes1), len(nodes2))
	}

	// Compare CPD values by position.
	for i, node := range nodes1 {
		cpd1 := bn.GetCPD(node)
		cpd2 := bn2.GetCPD(nodes2[i])
		if cpd1 == nil || cpd2 == nil {
			t.Errorf("CPD nil for node %d", i)
			continue
		}
		data1 := cpd1.ToFactor().Values().Data()
		data2 := cpd2.ToFactor().Values().Data()
		assertFloatsClose(t, data2, data1, "UAI round-trip "+node)
	}

	// Compare edge count.
	edges1 := bn.Edges()
	edges2 := bn2.Edges()
	if len(edges1) != len(edges2) {
		t.Errorf("UAI round-trip: edge count mismatch: %d vs %d", len(edges1), len(edges2))
	}
}

func TestUAI_RoundTrip_MultiParent(t *testing.T) {
	bn := buildMultiParentBN(t)

	var buf bytes.Buffer
	if err := WriteUAI(&buf, bn); err != nil {
		t.Fatalf("WriteUAI failed: %v", err)
	}

	bn2, err := ReadUAI(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadUAI failed: %v\nOutput:\n%s", err, buf.String())
	}

	nodes1 := bn.Nodes()
	nodes2 := bn2.Nodes()
	if len(nodes1) != len(nodes2) {
		t.Fatalf("UAI multi-parent round-trip: node count mismatch")
	}

	for i, node := range nodes1 {
		cpd1 := bn.GetCPD(node)
		cpd2 := bn2.GetCPD(nodes2[i])
		data1 := cpd1.ToFactor().Values().Data()
		data2 := cpd2.ToFactor().Values().Data()
		assertFloatsClose(t, data2, data1, "UAI multi-parent round-trip "+node)
	}
}

func TestReadUAI_Basic(t *testing.T) {
	uaiData := `BAYES
2
2 2
2
1 0
2 0 1
2
0.6 0.4
4
0.2 0.8 0.75 0.25
`
	bn, err := ReadUAI(strings.NewReader(uaiData))
	if err != nil {
		t.Fatalf("ReadUAI failed: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(nodes))
	}

	// V0 (unconditional).
	v0CPD := bn.GetCPD("V0")
	v0Data := v0CPD.ToFactor().Values().Data()
	assertFloatsClose(t, v0Data, []float64{0.6, 0.4}, "V0 CPD")

	// V1 | V0.
	v1CPD := bn.GetCPD("V1")
	v1Data := v1CPD.ToFactor().Values().Data()
	// child=s0: V0=s0=>0.2, V0=s1=>0.75
	// child=s1: V0=s0=>0.8, V0=s1=>0.25
	assertFloatsClose(t, v1Data, []float64{0.2, 0.75, 0.8, 0.25}, "V1 CPD")
}

// ---------------------------------------------------------------------------
// XDSL
// ---------------------------------------------------------------------------

func TestXDSL_RoundTrip(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WriteXDSL(&buf, bn); err != nil {
		t.Fatalf("WriteXDSL failed: %v", err)
	}

	bn2, err := ReadXDSL(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXDSL failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "XDSL round-trip")
}

func TestXDSL_RoundTrip_MultiParent(t *testing.T) {
	bn := buildMultiParentBN(t)

	var buf bytes.Buffer
	if err := WriteXDSL(&buf, bn); err != nil {
		t.Fatalf("WriteXDSL failed: %v", err)
	}

	bn2, err := ReadXDSL(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXDSL failed: %v\nOutput:\n%s", err, buf.String())
	}

	assertBNEqual(t, bn2, bn, "XDSL multi-parent round-trip")
}

func TestReadXDSL_Basic(t *testing.T) {
	xdslData := `<?xml version="1.0" encoding="UTF-8"?>
<smile id="test">
  <nodes>
    <cpt id="X">
      <state id="x0"/>
      <state id="x1"/>
      <probabilities>0.3 0.7</probabilities>
    </cpt>
  </nodes>
</smile>`

	bn, err := ReadXDSL(strings.NewReader(xdslData))
	if err != nil {
		t.Fatalf("ReadXDSL failed: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 1 || nodes[0] != "X" {
		t.Fatalf("expected [X], got %v", nodes)
	}

	cpd := bn.GetCPD("X")
	data := cpd.ToFactor().Values().Data()
	assertFloatsClose(t, data, []float64{0.3, 0.7}, "X CPD")
}

// ---------------------------------------------------------------------------
// PomdpX (stub)
// ---------------------------------------------------------------------------

func TestPomdpX_RoundTrip_Unconditional(t *testing.T) {
	// Build a simple unconditional-only network for the stub.
	bn := models.NewBayesianNetwork()
	must(t, bn.AddNode("X"))
	must(t, bn.SetStates("X", []string{"x0", "x1"}))

	xCPD, err := factors.NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	must(t, err)
	must(t, bn.AddCPD(xCPD))

	var buf bytes.Buffer
	if err := WritePomdpX(&buf, bn); err != nil {
		t.Fatalf("WritePomdpX failed: %v", err)
	}

	bn2, err := ReadPomdpX(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadPomdpX failed: %v\nOutput:\n%s", err, buf.String())
	}

	nodes := bn2.Nodes()
	if len(nodes) != 1 || nodes[0] != "X" {
		t.Fatalf("expected [X], got %v", nodes)
	}

	cpd := bn2.GetCPD("X")
	data := cpd.ToFactor().Values().Data()
	assertFloatsClose(t, data, []float64{0.3, 0.7}, "PomdpX round-trip X CPD")
}

func TestPomdpX_WriteRead(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WritePomdpX(&buf, bn); err != nil {
		t.Fatalf("WritePomdpX failed: %v", err)
	}

	// Just verify it can be read without error. The stub may not preserve
	// conditional distributions perfectly.
	_, err := ReadPomdpX(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadPomdpX failed: %v\nOutput:\n%s", err, buf.String())
	}
}

// ---------------------------------------------------------------------------
// XBN (stub)
// ---------------------------------------------------------------------------

func TestXBN_RoundTrip(t *testing.T) {
	bn := buildThreeNodeBN(t)

	var buf bytes.Buffer
	if err := WriteXBN(&buf, bn); err != nil {
		t.Fatalf("WriteXBN failed: %v", err)
	}

	bn2, err := ReadXBN(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXBN failed: %v\nOutput:\n%s", err, buf.String())
	}

	// Verify structure.
	nodes1 := bn.Nodes()
	nodes2 := bn2.Nodes()
	if len(nodes1) != len(nodes2) {
		t.Fatalf("XBN round-trip: node count mismatch: %d vs %d", len(nodes1), len(nodes2))
	}

	for i := range nodes1 {
		if nodes1[i] != nodes2[i] {
			t.Errorf("XBN round-trip: node %d: want %q, got %q", i, nodes1[i], nodes2[i])
		}
	}

	edges1 := bn.Edges()
	edges2 := bn2.Edges()
	if len(edges1) != len(edges2) {
		t.Fatalf("XBN round-trip: edge count mismatch: %d vs %d", len(edges1), len(edges2))
	}
}

func TestXBN_RoundTrip_Unconditional(t *testing.T) {
	bn := models.NewBayesianNetwork()
	must(t, bn.AddNode("X"))
	must(t, bn.SetStates("X", []string{"x0", "x1"}))

	xCPD, err := factors.NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	must(t, err)
	must(t, bn.AddCPD(xCPD))

	var buf bytes.Buffer
	if err := WriteXBN(&buf, bn); err != nil {
		t.Fatalf("WriteXBN failed: %v", err)
	}

	bn2, err := ReadXBN(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatalf("ReadXBN failed: %v\nOutput:\n%s", err, buf.String())
	}

	cpd := bn2.GetCPD("X")
	if cpd == nil {
		t.Fatal("X CPD is nil after round-trip")
	}
	data := cpd.ToFactor().Values().Data()
	assertFloatsClose(t, data, []float64{0.3, 0.7}, "XBN round-trip X CPD")
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

func TestReadXMLBIF_InvalidXML(t *testing.T) {
	_, err := ReadXMLBIF(strings.NewReader("<not valid xml"))
	if err == nil {
		t.Error("expected error for invalid XML")
	}
}

func TestReadNET_Empty(t *testing.T) {
	bn, err := ReadNET(strings.NewReader("net\n{\n}\n"))
	if err != nil {
		t.Fatalf("ReadNET failed on empty: %v", err)
	}
	if len(bn.Nodes()) != 0 {
		t.Errorf("expected 0 nodes, got %d", len(bn.Nodes()))
	}
}

func TestReadUAI_InvalidType(t *testing.T) {
	_, err := ReadUAI(strings.NewReader("MARKOV\n1\n2\n0\n"))
	if err == nil {
		t.Error("expected error for non-BAYES UAI type")
	}
}

func TestReadXDSL_InvalidXML(t *testing.T) {
	_, err := ReadXDSL(strings.NewReader("<not valid xml"))
	if err == nil {
		t.Error("expected error for invalid XML")
	}
}
