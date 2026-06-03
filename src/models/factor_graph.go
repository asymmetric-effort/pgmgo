package models

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// FactorGraph represents a factor graph — a bipartite graph of variable nodes
// and factor nodes. Each factor is connected to the variables in its scope.
type FactorGraph struct {
	variables    map[string]int                       // variable name -> cardinality
	factorList   []*factors.DiscreteFactor            // all factors in insertion order
	varToFactors map[string][]*factors.DiscreteFactor // variable -> factors containing it
}

// NewFactorGraph creates a new empty FactorGraph.
func NewFactorGraph() *FactorGraph {
	return &FactorGraph{
		variables:    make(map[string]int),
		factorList:   nil,
		varToFactors: make(map[string][]*factors.DiscreteFactor),
	}
}

// AddVariable adds a variable node with the given name and cardinality.
// Returns an error if the variable already exists or cardinality is invalid.
func (fg *FactorGraph) AddVariable(name string, card int) error {
	if _, exists := fg.variables[name]; exists {
		return fmt.Errorf("models: variable %q already exists", name)
	}
	if card <= 0 {
		return fmt.Errorf("models: cardinality must be positive, got %d", card)
	}
	fg.variables[name] = card
	return nil
}

// AddFactor adds a factor node to the graph. All variables in the factor's
// scope must already exist in the graph, and their cardinalities must match.
func (fg *FactorGraph) AddFactor(f *factors.DiscreteFactor) error {
	if f == nil {
		return fmt.Errorf("models: factor must not be nil")
	}

	vars := f.Variables()
	card := f.Cardinality()
	for i, v := range vars {
		expectedCard, exists := fg.variables[v]
		if !exists {
			return fmt.Errorf("models: factor references unknown variable %q", v)
		}
		if card[i] != expectedCard {
			return fmt.Errorf("models: factor cardinality %d for variable %q does not match expected %d",
				card[i], v, expectedCard)
		}
	}

	fg.factorList = append(fg.factorList, f)
	for _, v := range vars {
		fg.varToFactors[v] = append(fg.varToFactors[v], f)
	}
	return nil
}

// GetFactors returns all factors in the graph.
func (fg *FactorGraph) GetFactors() []*factors.DiscreteFactor {
	result := make([]*factors.DiscreteFactor, len(fg.factorList))
	copy(result, fg.factorList)
	return result
}

// GetVariables returns a sorted list of all variable names.
func (fg *FactorGraph) GetVariables() []string {
	vars := make([]string, 0, len(fg.variables))
	for v := range fg.variables {
		vars = append(vars, v)
	}
	sort.Strings(vars)
	return vars
}

// GetFactorsOf returns all factors that include the given variable in their scope.
// Returns nil if the variable does not exist or has no factors.
func (fg *FactorGraph) GetFactorsOf(variable string) []*factors.DiscreteFactor {
	fs := fg.varToFactors[variable]
	if len(fs) == 0 {
		return nil
	}
	result := make([]*factors.DiscreteFactor, len(fs))
	copy(result, fs)
	return result
}

// ToMarkovNetwork converts the factor graph to a Markov network.
// For each factor, edges are added between all pairs of variables in its scope.
// All factors are added to the resulting Markov network.
func (fg *FactorGraph) ToMarkovNetwork() (*MarkovNetwork, error) {
	if err := fg.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: ToMarkovNetwork: %w", err)
	}

	mn := NewMarkovNetwork()

	// Add all variables as nodes.
	for _, v := range fg.GetVariables() {
		if err := mn.AddNode(v); err != nil {
			return nil, fmt.Errorf("models: ToMarkovNetwork: %w", err)
		}
	}

	// Track edges already added to avoid duplicate edge errors.
	type edgePair struct{ a, b string }
	added := make(map[edgePair]bool)

	// For each factor, add edges between all pairs of variables in its scope.
	for _, f := range fg.factorList {
		vars := f.Variables()
		for i := 0; i < len(vars); i++ {
			for j := i + 1; j < len(vars); j++ {
				a, b := vars[i], vars[j]
				if a > b {
					a, b = b, a
				}
				if added[edgePair{a, b}] {
					continue
				}
				if err := mn.AddEdge(a, b); err != nil {
					return nil, fmt.Errorf("models: ToMarkovNetwork: %w", err)
				}
				added[edgePair{a, b}] = true
			}
		}
	}

	// Add copies of all factors to the Markov network.
	for _, f := range fg.factorList {
		if err := mn.AddFactor(f.Copy()); err != nil {
			return nil, fmt.Errorf("models: ToMarkovNetwork: %w", err)
		}
	}

	return mn, nil
}

// CheckModel validates the factor graph:
//  1. At least one variable exists.
//  2. At least one factor exists.
//  3. Every factor's variables exist in the graph with matching cardinalities.
//  4. Every variable is referenced by at least one factor.
func (fg *FactorGraph) CheckModel() error {
	if len(fg.variables) == 0 {
		return fmt.Errorf("models: factor graph has no variables")
	}
	if len(fg.factorList) == 0 {
		return fmt.Errorf("models: factor graph has no factors")
	}

	// Verify all factors reference valid variables with correct cardinalities.
	for i, f := range fg.factorList {
		vars := f.Variables()
		card := f.Cardinality()
		for j, v := range vars {
			expectedCard, exists := fg.variables[v]
			if !exists {
				return fmt.Errorf("models: factor %d references unknown variable %q", i, v)
			}
			if card[j] != expectedCard {
				return fmt.Errorf("models: factor %d cardinality %d for variable %q does not match expected %d",
					i, card[j], v, expectedCard)
			}
		}
	}

	// Verify every variable is referenced by at least one factor.
	for v := range fg.variables {
		if len(fg.varToFactors[v]) == 0 {
			return fmt.Errorf("models: variable %q is not referenced by any factor", v)
		}
	}

	return nil
}
