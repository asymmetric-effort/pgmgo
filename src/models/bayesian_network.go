package models

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/asymmetric-effort/pgmgo/src/base"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// BayesianNetwork represents a Bayesian network — a DAG where each node is
// associated with a conditional probability distribution (CPD).
type BayesianNetwork struct {
	dag    *base.DAG
	cpds   map[string]*factors.TabularCPD
	states map[string][]string // variable -> state names (optional)
}

// NewBayesianNetwork creates a new empty BayesianNetwork.
func NewBayesianNetwork() *BayesianNetwork {
	return &BayesianNetwork{
		dag:    base.NewDAG(),
		cpds:   make(map[string]*factors.TabularCPD),
		states: make(map[string][]string),
	}
}

// AddNode adds a node to the network.
func (bn *BayesianNetwork) AddNode(node string) error {
	return bn.dag.AddNode(node)
}

// AddEdge adds a directed edge from -> to. Both nodes must exist and the edge
// must not create a cycle.
func (bn *BayesianNetwork) AddEdge(from, to string) error {
	return bn.dag.AddEdge(from, to)
}

// Nodes returns a sorted list of all nodes in the network.
func (bn *BayesianNetwork) Nodes() []string {
	return bn.dag.Nodes()
}

// Edges returns all directed edges as [2]string pairs, sorted lexicographically.
func (bn *BayesianNetwork) Edges() [][2]string {
	raw := bn.dag.Edges()
	result := make([][2]string, len(raw))
	for i, e := range raw {
		result[i] = [2]string{e.Src, e.Dst}
	}
	return result
}

// Parents returns the sorted parents of a node.
func (bn *BayesianNetwork) Parents(node string) []string {
	return bn.dag.Parents(node)
}

// Children returns the sorted children of a node.
func (bn *BayesianNetwork) Children(node string) []string {
	return bn.dag.Children(node)
}

// AddCPD stores a CPD for its variable. The variable must be a node in the DAG.
func (bn *BayesianNetwork) AddCPD(cpd *factors.TabularCPD) error {
	if cpd == nil {
		return fmt.Errorf("models: cpd must not be nil")
	}
	v := cpd.Variable()
	if !bn.dag.HasNode(v) {
		return fmt.Errorf("models: variable %q is not a node in the network", v)
	}
	bn.cpds[v] = cpd
	return nil
}

// RemoveCPD removes the CPD for the given variable.
func (bn *BayesianNetwork) RemoveCPD(variable string) {
	delete(bn.cpds, variable)
}

// GetCPD returns the CPD for the given variable, or nil if none is set.
func (bn *BayesianNetwork) GetCPD(variable string) *factors.TabularCPD {
	return bn.cpds[variable]
}

// GetCPDs returns all CPDs, sorted by variable name.
func (bn *BayesianNetwork) GetCPDs() []*factors.TabularCPD {
	keys := make([]string, 0, len(bn.cpds))
	for k := range bn.cpds {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := make([]*factors.TabularCPD, len(keys))
	for i, k := range keys {
		result[i] = bn.cpds[k]
	}
	return result
}

// SetStates sets the state names for a variable.
func (bn *BayesianNetwork) SetStates(variable string, stateNames []string) error {
	if !bn.dag.HasNode(variable) {
		return fmt.Errorf("models: variable %q is not a node in the network", variable)
	}
	s := make([]string, len(stateNames))
	copy(s, stateNames)
	bn.states[variable] = s
	return nil
}

// GetStates returns the state names for a variable, or nil if none are set.
func (bn *BayesianNetwork) GetStates(variable string) []string {
	s, ok := bn.states[variable]
	if !ok {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

// CheckModel validates the Bayesian network:
//  1. Every node has a CPD.
//  2. Each CPD's evidence matches the node's parents in the DAG.
//  3. Each CPD passes Validate().
func (bn *BayesianNetwork) CheckModel() error {
	nodes := bn.dag.Nodes()

	for _, node := range nodes {
		cpd, ok := bn.cpds[node]
		if !ok {
			return fmt.Errorf("models: node %q has no CPD", node)
		}

		// Check evidence matches parents.
		parents := bn.dag.Parents(node) // sorted
		evidence := cpd.Evidence()
		sortedEvidence := make([]string, len(evidence))
		copy(sortedEvidence, evidence)
		sort.Strings(sortedEvidence)

		if len(parents) != len(sortedEvidence) {
			return fmt.Errorf("models: CPD for %q has evidence %v but node has parents %v",
				node, evidence, parents)
		}
		for i := range parents {
			if parents[i] != sortedEvidence[i] {
				return fmt.Errorf("models: CPD for %q has evidence %v but node has parents %v",
					node, evidence, parents)
			}
		}

		// Validate the CPD itself (columns sum to 1).
		if err := cpd.Validate(); err != nil {
			return fmt.Errorf("models: CPD for %q failed validation: %w", node, err)
		}
	}
	return nil
}

// Copy returns a deep copy of the BayesianNetwork.
func (bn *BayesianNetwork) Copy() *BayesianNetwork {
	newCPDs := make(map[string]*factors.TabularCPD, len(bn.cpds))
	for k, v := range bn.cpds {
		newCPDs[k] = v.Copy()
	}
	newStates := make(map[string][]string, len(bn.states))
	for k, v := range bn.states {
		s := make([]string, len(v))
		copy(s, v)
		newStates[k] = s
	}
	return &BayesianNetwork{
		dag:    bn.dag.Copy(),
		cpds:   newCPDs,
		states: newStates,
	}
}

// GetRandomCPDs generates random TabularCPDs for all nodes in the network
// based on graph structure. Each node gets a CPD with the given number of
// states. Columns are normalized to sum to 1. The seed controls the RNG.
func (bn *BayesianNetwork) GetRandomCPDs(nStates int, seed int64) error {
	if nStates <= 0 {
		return fmt.Errorf("models: nStates must be positive, got %d", nStates)
	}

	rng := rand.New(rand.NewSource(seed))
	nodes := bn.dag.Nodes()

	for _, node := range nodes {
		parents := bn.dag.Parents(node)
		evidenceCard := make([]int, len(parents))
		for i := range parents {
			evidenceCard[i] = nStates
		}

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		vals := make([][]float64, nStates)
		for i := range vals {
			vals[i] = make([]float64, numParentConfigs)
		}

		for j := 0; j < numParentConfigs; j++ {
			sum := 0.0
			for i := 0; i < nStates; i++ {
				v := rng.Float64()
				vals[i][j] = v
				sum += v
			}
			for i := 0; i < nStates; i++ {
				vals[i][j] /= sum
			}
		}

		cpd, err := factors.NewTabularCPD(node, nStates, vals, parents, evidenceCard)
		if err != nil {
			return fmt.Errorf("models: failed to create random CPD for %q: %w", node, err)
		}
		bn.cpds[node] = cpd
	}
	return nil
}

// ToMarkovFactors converts all CPDs to discrete factors suitable for inference.
// CheckModel is called first to ensure the network is valid.
func (bn *BayesianNetwork) ToMarkovFactors() ([]*factors.DiscreteFactor, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: cannot convert to factors: %w", err)
	}

	nodes := bn.dag.Nodes() // sorted
	result := make([]*factors.DiscreteFactor, 0, len(nodes))
	for _, node := range nodes {
		cpd := bn.cpds[node]
		result = append(result, cpd.ToFactor())
	}
	return result, nil
}
