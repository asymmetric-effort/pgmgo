package models

import (
	"fmt"
	"math/rand"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// Simulate generates n samples from the Bayesian network using forward
// (ancestral) sampling. If evidence is non-nil, rejection sampling is used to
// condition on the evidence. The seed controls the RNG. Returns a DataFrame
// with one column per variable and one row per accepted sample.
func (bn *BayesianNetwork) Simulate(n int, evidence map[string]int, seed int64) (*tabgo.DataFrame, error) {
	if n <= 0 {
		return nil, fmt.Errorf("models: n must be positive, got %d", n)
	}
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: Simulate requires a valid model: %w", err)
	}

	rng := rand.New(rand.NewSource(seed))
	nodes := bn.Nodes()

	order, err := bn.dag.TopologicalOrder()
	if err != nil {
		order = nodes
	}

	samples := make(map[string][]any, len(nodes))
	for _, node := range nodes {
		samples[node] = make([]any, 0, n)
	}

	maxAttempts := n * 1000 // avoid infinite loops for impossible evidence
	if evidence == nil || len(evidence) == 0 {
		maxAttempts = n
	}

	accepted := 0
	for attempt := 0; attempt < maxAttempts && accepted < n; attempt++ {
		assignment := make(map[string]int, len(nodes))
		sampleBN(bn, order, assignment, rng)

		// Check evidence
		if evidence != nil && len(evidence) > 0 {
			match := true
			for v, val := range evidence {
				if assignment[v] != val {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}

		for _, node := range nodes {
			samples[node] = append(samples[node], assignment[node])
		}
		accepted++
	}

	if accepted < n {
		return nil, fmt.Errorf("models: Simulate: only %d/%d samples accepted after %d attempts", accepted, n, maxAttempts)
	}

	seriesMap := make(map[string]*tabgo.Series, len(nodes))
	for _, node := range nodes {
		seriesMap[node] = tabgo.NewSeries(node, samples[node])
	}
	return tabgo.NewDataFrame(seriesMap), nil
}

// HasNode returns true if the node exists in the network.
func (bn *BayesianNetwork) HasNode(node string) bool {
	return bn.dag.HasNode(node)
}

// HasEdge returns true if the directed edge exists in the network.
func (bn *BayesianNetwork) HasEdge(from, to string) bool {
	return bn.dag.HasEdge(from, to)
}

// TopologicalOrder returns a topological ordering of the network nodes.
func (bn *BayesianNetwork) TopologicalOrder() ([]string, error) {
	return bn.dag.TopologicalOrder()
}

// NumberOfNodes returns the number of nodes.
func (bn *BayesianNetwork) NumberOfNodes() int {
	return len(bn.Nodes())
}

// NumberOfEdges returns the number of edges.
func (bn *BayesianNetwork) NumberOfEdges() int {
	return len(bn.Edges())
}

// GetAllStates returns a map from each variable to its state names.
func (bn *BayesianNetwork) GetAllStates() map[string][]string {
	result := make(map[string][]string)
	for _, node := range bn.Nodes() {
		states := bn.GetStates(node)
		if states != nil {
			result[node] = states
		}
	}
	return result
}
