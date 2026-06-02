package readwrite

import (
	"encoding/xml"
	"fmt"
	"io"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// PomdpX XML structures (basic/stub support).
type pomdpxDoc struct {
	XMLName         xml.Name          `xml:"pomdpx"`
	Version         string            `xml:"version,attr"`
	ID              string            `xml:"id,attr,omitempty"`
	Description     string            `xml:"Description,omitempty"`
	Variables       pomdpxVarBlock    `xml:"Variable"`
	InitBelief      *pomdpxInit       `xml:"InitialStateBelief,omitempty"`
	StateTransition *pomdpxTransBlock `xml:"StateTransitionFunction,omitempty"`
}

type pomdpxVarBlock struct {
	StateVars []pomdpxStateVar `xml:"StateVar"`
}

type pomdpxStateVar struct {
	VarName    string   `xml:"vnamePrev,attr"`
	NumValues  int      `xml:"numValues,attr,omitempty"`
	ValueNames []string `xml:"ValueEnum,omitempty"`
}

type pomdpxInit struct {
	CondProbs []pomdpxCondProb `xml:"CondProb"`
}

type pomdpxTransBlock struct {
	CondProbs []pomdpxCondProb `xml:"CondProb"`
}

type pomdpxCondProb struct {
	Name    string         `xml:"name,attr,omitempty"`
	Var     []pomdpxVar    `xml:"Var"`
	Parents []pomdpxParent `xml:"Parent"`
	Params  []pomdpxParam  `xml:"Parameter"`
}

type pomdpxVar struct {
	Name string `xml:",chardata"`
}

type pomdpxParent struct {
	Names string `xml:",chardata"`
}

type pomdpxParam struct {
	Type    string        `xml:"type,attr,omitempty"`
	Entries []pomdpxEntry `xml:"Entry"`
}

type pomdpxEntry struct {
	Instance  string `xml:"Instance,omitempty"`
	ProbTable string `xml:"ProbTable"`
}

// ReadPomdpX parses a PomdpX format file with basic support and returns a BayesianNetwork.
// This is a stub implementation that handles simple state variable definitions with
// initial belief distributions.
func ReadPomdpX(r io.Reader) (*models.BayesianNetwork, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading PomdpX: %w", err)
	}

	var doc pomdpxDoc
	if err := xml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("readwrite: error parsing PomdpX: %w", err)
	}

	bn := models.NewBayesianNetwork()

	type varInfo struct {
		card   int
		states []string
	}
	varMap := make(map[string]*varInfo)

	// Add state variables.
	for _, sv := range doc.Variables.StateVars {
		name := sv.VarName
		if name == "" {
			continue
		}

		var states []string
		if len(sv.ValueNames) > 0 {
			// ValueEnum is space-separated in the XML.
			for _, ve := range sv.ValueNames {
				for _, s := range strings.Fields(ve) {
					states = append(states, s)
				}
			}
		}

		numVals := sv.NumValues
		if numVals <= 0 {
			numVals = len(states)
		}
		if len(states) == 0 {
			// Generate default state names.
			states = make([]string, numVals)
			for i := 0; i < numVals; i++ {
				states[i] = fmt.Sprintf("s%d", i)
			}
		}

		if err := bn.AddNode(name); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		if err := bn.SetStates(name, states); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		varMap[name] = &varInfo{card: len(states), states: states}
	}

	// Parse initial belief as unconditional CPDs.
	if doc.InitBelief != nil {
		for _, cp := range doc.InitBelief.CondProbs {
			if len(cp.Var) == 0 {
				continue
			}
			child := strings.TrimSpace(cp.Var[0].Name)
			childInfo := varMap[child]
			if childInfo == nil {
				continue
			}

			// Get probability values from entries.
			var allVals []float64
			for _, param := range cp.Params {
				for _, entry := range param.Entries {
					vals, err := xmlbifParseFloats(entry.ProbTable)
					if err != nil {
						return nil, fmt.Errorf("readwrite: error parsing PomdpX probs for %q: %w", child, err)
					}
					allVals = append(allVals, vals...)
				}
			}

			if len(allVals) == childInfo.card {
				values := make([][]float64, childInfo.card)
				for cs := 0; cs < childInfo.card; cs++ {
					values[cs] = []float64{allVals[cs]}
				}

				cpd, err := factors.NewTabularCPD(child, childInfo.card, values, nil, nil)
				if err != nil {
					return nil, fmt.Errorf("readwrite: failed to create CPD for %q: %w", child, err)
				}
				if err := bn.AddCPD(cpd); err != nil {
					return nil, fmt.Errorf("readwrite: %w", err)
				}
			}
		}
	}

	// For any variable without a CPD, create a uniform distribution.
	for name, info := range varMap {
		if bn.GetCPD(name) == nil {
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
	}

	return bn, nil
}

// WritePomdpX serializes a BayesianNetwork to PomdpX format (basic stub).
// Only unconditional variables are supported. Conditional distributions are
// written as state transition functions.
func WritePomdpX(w io.Writer, bn *models.BayesianNetwork) error {
	nodes := bn.Nodes()

	var stateVars []pomdpxStateVar
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}
		stateVars = append(stateVars, pomdpxStateVar{
			VarName:    node,
			NumValues:  len(states),
			ValueNames: []string{strings.Join(states, " ")},
		})
	}

	// Build initial belief for root nodes and state transition for conditional.
	var initCondProbs []pomdpxCondProb
	var transCondProbs []pomdpxCondProb

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

		if len(evidence) == 0 {
			// Unconditional: initial belief.
			var parts []string
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs]))
			}

			initCondProbs = append(initCondProbs, pomdpxCondProb{
				Var: []pomdpxVar{{Name: node}},
				Params: []pomdpxParam{{
					Type: "TBL",
					Entries: []pomdpxEntry{{
						ProbTable: strings.Join(parts, " "),
					}},
				}},
			})
		} else {
			// Conditional: state transition.
			var parts []string
			for pc := 0; pc < numParentConfigs; pc++ {
				for cs := 0; cs < childCard; cs++ {
					parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
				}
			}

			transCondProbs = append(transCondProbs, pomdpxCondProb{
				Var:     []pomdpxVar{{Name: node}},
				Parents: []pomdpxParent{{Names: strings.Join(evidence, " ")}},
				Params: []pomdpxParam{{
					Type: "TBL",
					Entries: []pomdpxEntry{{
						ProbTable: strings.Join(parts, " "),
					}},
				}},
			})
		}
	}

	doc := pomdpxDoc{
		Version:   "1.0",
		Variables: pomdpxVarBlock{StateVars: stateVars},
	}

	if len(initCondProbs) > 0 {
		doc.InitBelief = &pomdpxInit{CondProbs: initCondProbs}
	}
	if len(transCondProbs) > 0 {
		doc.StateTransition = &pomdpxTransBlock{CondProbs: transCondProbs}
	}

	if _, err := fmt.Fprint(w, xml.Header); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	enc := xml.NewEncoder(w)
	enc.Indent("", "  ")
	if err := enc.Encode(doc); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}
	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	return nil
}
