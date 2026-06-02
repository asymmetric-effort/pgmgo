package models

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// FunctionalBayesianNetwork is a Bayesian network where nodes use
// FunctionalCPDs instead of TabularCPDs. It embeds *BayesianNetwork for
// graph structure (nodes, edges) and stores functional CPDs in a separate map.
type FunctionalBayesianNetwork struct {
	*BayesianNetwork
	funcCPDs map[string]*factors.FunctionalCPD
}

// NewFunctionalBayesianNetwork creates a new empty FunctionalBayesianNetwork.
func NewFunctionalBayesianNetwork() *FunctionalBayesianNetwork {
	return &FunctionalBayesianNetwork{
		BayesianNetwork: NewBayesianNetwork(),
		funcCPDs:        make(map[string]*factors.FunctionalCPD),
	}
}

// AddFunctionalCPD stores a FunctionalCPD for its variable. It validates that
// the variable exists in the graph and that the CPD's evidence matches the
// node's parents in the DAG.
func (fbn *FunctionalBayesianNetwork) AddFunctionalCPD(cpd *factors.FunctionalCPD) error {
	if cpd == nil {
		return fmt.Errorf("models: cpd must not be nil")
	}

	v := cpd.Variable()
	if !fbn.dag.HasNode(v) {
		return fmt.Errorf("models: variable %q is not a node in the network", v)
	}

	// Verify evidence matches parents.
	parents := fbn.dag.Parents(v) // sorted
	evidence := cpd.Evidence()
	sortedEvidence := make([]string, len(evidence))
	copy(sortedEvidence, evidence)
	sort.Strings(sortedEvidence)

	if len(parents) != len(sortedEvidence) {
		return fmt.Errorf("models: FunctionalCPD for %q has evidence %v but node has parents %v",
			v, evidence, parents)
	}
	for i := range parents {
		if parents[i] != sortedEvidence[i] {
			return fmt.Errorf("models: FunctionalCPD for %q has evidence %v but node has parents %v",
				v, evidence, parents)
		}
	}

	fbn.funcCPDs[v] = cpd
	return nil
}

// GetFunctionalCPD returns the FunctionalCPD for the given variable, or nil
// if none is set.
func (fbn *FunctionalBayesianNetwork) GetFunctionalCPD(variable string) *factors.FunctionalCPD {
	return fbn.funcCPDs[variable]
}

// CheckModel validates the FunctionalBayesianNetwork:
//  1. Every node has a FunctionalCPD.
//  2. Each CPD's evidence matches the node's parents in the DAG.
//  3. Each CPD passes Validate().
func (fbn *FunctionalBayesianNetwork) CheckModel() error {
	nodes := fbn.dag.Nodes()

	for _, node := range nodes {
		cpd, ok := fbn.funcCPDs[node]
		if !ok {
			return fmt.Errorf("models: node %q has no FunctionalCPD", node)
		}

		// Check evidence matches parents.
		parents := fbn.dag.Parents(node) // sorted
		evidence := cpd.Evidence()
		sortedEvidence := make([]string, len(evidence))
		copy(sortedEvidence, evidence)
		sort.Strings(sortedEvidence)

		if len(parents) != len(sortedEvidence) {
			return fmt.Errorf("models: FunctionalCPD for %q has evidence %v but node has parents %v",
				node, evidence, parents)
		}
		for i := range parents {
			if parents[i] != sortedEvidence[i] {
				return fmt.Errorf("models: FunctionalCPD for %q has evidence %v but node has parents %v",
					node, evidence, parents)
			}
		}

		if err := cpd.Validate(); err != nil {
			return fmt.Errorf("models: FunctionalCPD for %q failed validation: %w", node, err)
		}
	}
	return nil
}

// Copy returns a deep copy of the FunctionalBayesianNetwork.
func (fbn *FunctionalBayesianNetwork) Copy() *FunctionalBayesianNetwork {
	newCPDs := make(map[string]*factors.FunctionalCPD, len(fbn.funcCPDs))
	for k, v := range fbn.funcCPDs {
		newCPDs[k] = v.Copy()
	}
	return &FunctionalBayesianNetwork{
		BayesianNetwork: fbn.BayesianNetwork.Copy(),
		funcCPDs:        newCPDs,
	}
}
