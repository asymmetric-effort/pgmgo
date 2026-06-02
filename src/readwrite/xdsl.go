package readwrite

import (
	"encoding/xml"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// XDSL XML structures for GeNIe format.
type xdslSmile struct {
	XMLName xml.Name  `xml:"smile"`
	ID      string    `xml:"id,attr"`
	Nodes   xdslNodes `xml:"nodes"`
	Exts    *xdslExts `xml:"extensions,omitempty"`
}

type xdslNodes struct {
	CPTs []xdslCPT `xml:"cpt"`
}

type xdslCPT struct {
	ID            string      `xml:"id,attr"`
	States        []xdslState `xml:"state"`
	Parents       string      `xml:"parents,omitempty"`
	Probabilities string      `xml:"probabilities"`
}

type xdslState struct {
	ID string `xml:"id,attr"`
}

type xdslExts struct {
	// Placeholder for extensions; not parsed.
}

// ReadXDSL parses a GeNIe XDSL format file and returns a BayesianNetwork.
func ReadXDSL(r io.Reader) (*models.BayesianNetwork, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading XDSL: %w", err)
	}

	var smile xdslSmile
	if err := xml.Unmarshal(data, &smile); err != nil {
		return nil, fmt.Errorf("readwrite: error parsing XDSL: %w", err)
	}

	bn := models.NewBayesianNetwork()

	type varInfo struct {
		card   int
		states []string
	}
	varMap := make(map[string]*varInfo)

	// First pass: add all nodes and states.
	for _, cpt := range smile.Nodes.CPTs {
		name := cpt.ID
		states := make([]string, len(cpt.States))
		for i, s := range cpt.States {
			states[i] = s.ID
		}
		if err := bn.AddNode(name); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		if err := bn.SetStates(name, states); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		varMap[name] = &varInfo{card: len(states), states: states}
	}

	// Second pass: add edges and CPDs.
	for _, cpt := range smile.Nodes.CPTs {
		child := cpt.ID
		childInfo := varMap[child]

		var parents []string
		var evidenceCard []int
		if strings.TrimSpace(cpt.Parents) != "" {
			for _, p := range strings.Fields(cpt.Parents) {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				parents = append(parents, p)
				pi := varMap[p]
				if pi == nil {
					return nil, fmt.Errorf("readwrite: CPT references unknown parent %q", p)
				}
				evidenceCard = append(evidenceCard, pi.card)
				if err := bn.AddEdge(p, child); err != nil {
					if !strings.Contains(err.Error(), "already exists") {
						return nil, fmt.Errorf("readwrite: %w", err)
					}
				}
			}
		}

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		// Parse probabilities.
		vals, err := xmlbifParseFloats(cpt.Probabilities)
		if err != nil {
			return nil, fmt.Errorf("readwrite: error parsing probabilities for %q: %w", child, err)
		}

		expectedLen := childInfo.card * numParentConfigs
		if len(vals) != expectedLen {
			return nil, fmt.Errorf("readwrite: CPT for %q has %d values, expected %d",
				child, len(vals), expectedLen)
		}

		// XDSL ordering: parent configs outer, child states inner.
		values := make([][]float64, childInfo.card)
		for cs := 0; cs < childInfo.card; cs++ {
			values[cs] = make([]float64, numParentConfigs)
		}

		idx := 0
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childInfo.card; cs++ {
				values[cs][pc] = vals[idx]
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

	return bn, nil
}

// WriteXDSL serializes a BayesianNetwork to GeNIe XDSL format.
func WriteXDSL(w io.Writer, bn *models.BayesianNetwork) error {
	nodes := bn.Nodes()

	var cpts []xdslCPT
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}

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

		// Build probability string: parent configs outer, child states inner.
		var parts []string
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
			}
		}

		xStates := make([]xdslState, len(states))
		for i, s := range states {
			xStates[i] = xdslState{ID: s}
		}

		cpts = append(cpts, xdslCPT{
			ID:            node,
			States:        xStates,
			Parents:       strings.Join(evidence, " "),
			Probabilities: strings.Join(parts, " "),
		})
	}

	smile := xdslSmile{
		ID:    "unknown",
		Nodes: xdslNodes{CPTs: cpts},
	}

	if _, err := fmt.Fprint(w, xml.Header); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	enc := xml.NewEncoder(w)
	enc.Indent("", "  ")
	if err := enc.Encode(smile); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}
	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	return nil
}

// xdslFormatProbs formats a slice of float64 values to a space-separated string.
func xdslFormatProbs(vals []float64) string {
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = strconv.FormatFloat(v, 'g', 10, 64)
	}
	return strings.Join(parts, " ")
}
