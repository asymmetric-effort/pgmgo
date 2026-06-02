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

// xmlBIF is the top-level XML BIF element.
type xmlBIF struct {
	XMLName xml.Name   `xml:"BIF"`
	Version string     `xml:"VERSION,attr"`
	Network xmlNetwork `xml:"NETWORK"`
}

type xmlNetwork struct {
	Name        string          `xml:"NAME"`
	Variables   []xmlVariable   `xml:"VARIABLE"`
	Definitions []xmlDefinition `xml:"DEFINITION"`
}

type xmlVariable struct {
	Type     string   `xml:"TYPE,attr"`
	Name     string   `xml:"NAME"`
	Outcomes []string `xml:"OUTCOME"`
}

type xmlDefinition struct {
	For   string   `xml:"FOR"`
	Given []string `xml:"GIVEN"`
	Table string   `xml:"TABLE"`
}

// ReadXMLBIF parses an XMLBIF format file and returns a BayesianNetwork.
func ReadXMLBIF(r io.Reader) (*models.BayesianNetwork, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading XMLBIF: %w", err)
	}

	var bif xmlBIF
	if err := xml.Unmarshal(data, &bif); err != nil {
		return nil, fmt.Errorf("readwrite: error parsing XMLBIF: %w", err)
	}

	bn := models.NewBayesianNetwork()

	// Variable info map for CPD construction.
	type varInfo struct {
		card   int
		states []string
	}
	varMap := make(map[string]*varInfo)

	// Add variables.
	for _, v := range bif.Network.Variables {
		name := strings.TrimSpace(v.Name)
		if err := bn.AddNode(name); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		states := make([]string, len(v.Outcomes))
		for i, o := range v.Outcomes {
			states[i] = strings.TrimSpace(o)
		}
		if err := bn.SetStates(name, states); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		varMap[name] = &varInfo{card: len(states), states: states}
	}

	// Add definitions (CPDs).
	for _, def := range bif.Network.Definitions {
		child := strings.TrimSpace(def.For)
		childInfo := varMap[child]
		if childInfo == nil {
			return nil, fmt.Errorf("readwrite: definition references unknown variable %q", child)
		}

		var parents []string
		var evidenceCard []int
		for _, g := range def.Given {
			p := strings.TrimSpace(g)
			parents = append(parents, p)
			pi := varMap[p]
			if pi == nil {
				return nil, fmt.Errorf("readwrite: definition references unknown parent %q", p)
			}
			evidenceCard = append(evidenceCard, pi.card)
			if err := bn.AddEdge(p, child); err != nil {
				if !strings.Contains(err.Error(), "already exists") {
					return nil, fmt.Errorf("readwrite: %w", err)
				}
			}
		}

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		// Parse table values.
		vals, err := xmlbifParseFloats(def.Table)
		if err != nil {
			return nil, fmt.Errorf("readwrite: error parsing table for %q: %w", child, err)
		}

		expectedLen := childInfo.card * numParentConfigs
		if len(vals) != expectedLen {
			return nil, fmt.Errorf("readwrite: table for %q has %d values, expected %d",
				child, len(vals), expectedLen)
		}

		// XMLBIF table is row-major: iterates parent configs, then child states.
		// i.e., for each parent config, list all child state probabilities.
		// values[childState][parentConfig]
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

// WriteXMLBIF serializes a BayesianNetwork to XMLBIF format.
func WriteXMLBIF(w io.Writer, bn *models.BayesianNetwork) error {
	nodes := bn.Nodes()

	var variables []xmlVariable
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}
		variables = append(variables, xmlVariable{
			Type:     "nature",
			Name:     node,
			Outcomes: states,
		})
	}

	var definitions []xmlDefinition
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

		// XMLBIF table: for each parent config, list child state probs.
		var parts []string
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
			}
		}

		definitions = append(definitions, xmlDefinition{
			For:   node,
			Given: evidence,
			Table: strings.Join(parts, " "),
		})
	}

	bif := xmlBIF{
		Version: "0.3",
		Network: xmlNetwork{
			Name:        "unknown",
			Variables:   variables,
			Definitions: definitions,
		},
	}

	if _, err := fmt.Fprint(w, xml.Header); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	enc := xml.NewEncoder(w)
	enc.Indent("", "  ")
	if err := enc.Encode(bif); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}
	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	return nil
}

// xmlbifParseFloats parses whitespace-separated float values.
func xmlbifParseFloats(s string) ([]float64, error) {
	fields := strings.Fields(strings.TrimSpace(s))
	vals := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid probability value %q: %w", f, err)
		}
		vals = append(vals, v)
	}
	return vals, nil
}

// formatFloat formats a float64 for output with minimal precision.
func formatFloat(v float64) string {
	return fmt.Sprintf("%.10g", v)
}
