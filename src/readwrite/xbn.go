package readwrite

import (
	"encoding/xml"
	"fmt"
	"io"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// XBN XML structures for Microsoft XBN format.
type xbnAnalysisNotebook struct {
	XMLName xml.Name `xml:"ANALYSISNOTEBOOK"`
	BNMODEL xbnModel `xml:"BNMODEL"`
}

type xbnModel struct {
	Name       string        `xml:"NAME,attr"`
	StaticProp xbnStaticProp `xml:"STATICPROPERTIES"`
	DynaProp   xbnDynaProp   `xml:"DYNAMICPROPERTIES"`
}

type xbnStaticProp struct {
	NodeList xbnNodeList `xml:"NODELIST"`
	ArcList  xbnArcList  `xml:"ARCLIST"`
}

type xbnNodeList struct {
	Nodes []xbnNode `xml:"NODE"`
}

type xbnNode struct {
	Name   string     `xml:"NAME,attr"`
	States []xbnState `xml:"STATENAME"`
}

type xbnState struct {
	Value string `xml:",chardata"`
}

type xbnArcList struct {
	Arcs []xbnArc `xml:"ARC"`
}

type xbnArc struct {
	Parent string `xml:"PARENT,attr"`
	Child  string `xml:"CHILD,attr"`
}

type xbnDynaProp struct {
	Formats []xbnFormat `xml:"FORMAT"`
	Dists   []xbnDist   `xml:"DISTRIBS>DIST"`
}

type xbnFormat struct {
	// Placeholder; not used in basic parsing.
}

type xbnDist struct {
	Type     string        `xml:"TYPE,attr"`
	CondElem []xbnCondElem `xml:"CONDSET>CONDELEM"`
	DPIs     []xbnDPI      `xml:"DPIS>DPI"`
	PrivDist []xbnPrivDist `xml:"PRIVATE>DPIS>DPI"`
}

type xbnCondElem struct {
	Name string `xml:"NAME,attr"`
}

type xbnDPI struct {
	Indexes string `xml:"INDEXES,attr,omitempty"`
	Values  string `xml:",chardata"`
}

type xbnPrivDist struct {
	Indexes string `xml:"INDEXES,attr,omitempty"`
	Values  string `xml:",chardata"`
}

// ReadXBN parses a Microsoft XBN format file and returns a BayesianNetwork.
// This is a basic/stub implementation.
func ReadXBN(r io.Reader) (*models.BayesianNetwork, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading XBN: %w", err)
	}

	var notebook xbnAnalysisNotebook
	if err := xml.Unmarshal(data, &notebook); err != nil {
		return nil, fmt.Errorf("readwrite: error parsing XBN: %w", err)
	}

	bn := models.NewBayesianNetwork()

	type varInfo struct {
		card   int
		states []string
	}
	varMap := make(map[string]*varInfo)

	// Add nodes.
	var nodeOrder []string
	for _, node := range notebook.BNMODEL.StaticProp.NodeList.Nodes {
		name := node.Name
		states := make([]string, len(node.States))
		for i, s := range node.States {
			states[i] = strings.TrimSpace(s.Value)
		}
		if len(states) == 0 {
			states = []string{"s0", "s1"} // default binary
		}
		if err := bn.AddNode(name); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		if err := bn.SetStates(name, states); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		varMap[name] = &varInfo{card: len(states), states: states}
		nodeOrder = append(nodeOrder, name)
	}

	// Add arcs.
	for _, arc := range notebook.BNMODEL.StaticProp.ArcList.Arcs {
		if err := bn.AddEdge(arc.Parent, arc.Child); err != nil {
			if !strings.Contains(err.Error(), "already exists") {
				return nil, fmt.Errorf("readwrite: %w", err)
			}
		}
	}

	// Parse distributions.
	for i, dist := range notebook.BNMODEL.DynaProp.Dists {
		if i >= len(nodeOrder) {
			break
		}
		child := nodeOrder[i]
		childInfo := varMap[child]

		var parents []string
		var evidenceCard []int
		for _, ce := range dist.CondElem {
			p := ce.Name
			if p == child {
				continue
			}
			pi := varMap[p]
			if pi == nil {
				continue
			}
			parents = append(parents, p)
			evidenceCard = append(evidenceCard, pi.card)
		}

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		// Collect all DPI values.
		var allVals []float64
		for _, dpi := range dist.DPIs {
			vals, err := xmlbifParseFloats(dpi.Values)
			if err != nil {
				return nil, fmt.Errorf("readwrite: error parsing XBN dist for %q: %w", child, err)
			}
			allVals = append(allVals, vals...)
		}

		expectedLen := childInfo.card * numParentConfigs
		if len(allVals) != expectedLen {
			// Try to use uniform if data mismatch.
			allVals = make([]float64, expectedLen)
			prob := 1.0 / float64(childInfo.card)
			for j := range allVals {
				allVals[j] = prob
			}
		}

		// XBN ordering: each DPI row is one parent config, child states listed.
		values := make([][]float64, childInfo.card)
		for cs := 0; cs < childInfo.card; cs++ {
			values[cs] = make([]float64, numParentConfigs)
		}

		idx := 0
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childInfo.card; cs++ {
				values[cs][pc] = allVals[idx]
				idx++
			}
		}

		cpd, err := factors.NewTabularCPD(child, childInfo.card, values, parents, evidenceCard)
		if err != nil {
			return nil, fmt.Errorf("readwrite: failed to create CPD for %q: %w", child, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
	}

	// For nodes without dists, create uniform CPDs.
	for _, name := range nodeOrder {
		if bn.GetCPD(name) != nil {
			continue
		}
		info := varMap[name]
		prob := 1.0 / float64(info.card)
		values := make([][]float64, info.card)
		for cs := 0; cs < info.card; cs++ {
			values[cs] = []float64{prob}
		}
		cpd, err := factors.NewTabularCPD(name, info.card, values, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("readwrite: failed to create default CPD for %q: %w", name, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
	}

	return bn, nil
}

// WriteXBN serializes a BayesianNetwork to Microsoft XBN format (basic stub).
func WriteXBN(w io.Writer, bn *models.BayesianNetwork) error {
	nodes := bn.Nodes()

	// Build node list.
	var xbnNodes []xbnNode
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}
		xStates := make([]xbnState, len(states))
		for i, s := range states {
			xStates[i] = xbnState{Value: s}
		}
		xbnNodes = append(xbnNodes, xbnNode{Name: node, States: xStates})
	}

	// Build arc list.
	edges := bn.Edges()
	arcs := make([]xbnArc, len(edges))
	for i, e := range edges {
		arcs[i] = xbnArc{Parent: e[0], Child: e[1]}
	}

	// Build distributions.
	var dists []xbnDist
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			return fmt.Errorf("readwrite: variable %q has no CPD", node)
		}

		evidence := cpd.Evidence()
		evidenceCard := cpd.EvidenceCard()
		childCard := cpd.VariableCard()

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		data := cpd.ToFactor().Values().Data()

		var condElems []xbnCondElem
		for _, ev := range evidence {
			condElems = append(condElems, xbnCondElem{Name: ev})
		}

		var dpis []xbnDPI
		for pc := 0; pc < numParentConfigs; pc++ {
			var parts []string
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
			}
			dpis = append(dpis, xbnDPI{Values: strings.Join(parts, " ")})
		}

		dists = append(dists, xbnDist{
			Type:     "discrete",
			CondElem: condElems,
			DPIs:     dpis,
		})
	}

	notebook := xbnAnalysisNotebook{
		BNMODEL: xbnModel{
			Name: "unknown",
			StaticProp: xbnStaticProp{
				NodeList: xbnNodeList{Nodes: xbnNodes},
				ArcList:  xbnArcList{Arcs: arcs},
			},
			DynaProp: xbnDynaProp{
				Dists: dists,
			},
		},
	}

	if _, err := fmt.Fprint(w, xml.Header); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	enc := xml.NewEncoder(w)
	enc.Indent("", "  ")
	if err := enc.Encode(notebook); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}
	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	return nil
}
