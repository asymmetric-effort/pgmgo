package learning

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// MaximumLikelihoodEstimator estimates CPD parameters for a BayesianNetwork
// from observed data using maximum likelihood estimation.
type MaximumLikelihoodEstimator struct {
	bn   *models.BayesianNetwork
	data *tabgo.DataFrame
}

// NewMLE creates a new MaximumLikelihoodEstimator. The BayesianNetwork must
// already have nodes and edges defined. The DataFrame should have columns
// matching the node names, with integer state indices as values.
func NewMLE(bn *models.BayesianNetwork, data *tabgo.DataFrame) *MaximumLikelihoodEstimator {
	return &MaximumLikelihoodEstimator{
		bn:   bn,
		data: data,
	}
}

// Estimate fits CPDs for all nodes in the Bayesian network from the data.
// For each node it counts occurrences of each (node_value, parent_values)
// combination, normalizes per parent configuration, and stores the resulting
// TabularCPD in the network.
func (mle *MaximumLikelihoodEstimator) Estimate() error {
	if mle.bn == nil {
		return fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if mle.data == nil {
		return fmt.Errorf("learning: data is nil")
	}

	nodes := mle.bn.Nodes()
	if len(nodes) == 0 {
		return fmt.Errorf("learning: BayesianNetwork has no nodes")
	}

	// Validate that all required columns exist in the data.
	dataColumns := make(map[string]bool)
	for _, c := range mle.data.Columns() {
		dataColumns[c] = true
	}
	for _, node := range nodes {
		if !dataColumns[node] {
			return fmt.Errorf("learning: data is missing column for node %q", node)
		}
	}

	for _, node := range nodes {
		cpd, err := mle.estimateNode(node)
		if err != nil {
			return fmt.Errorf("learning: failed to estimate CPD for %q: %w", node, err)
		}
		if err := mle.bn.AddCPD(cpd); err != nil {
			return fmt.Errorf("learning: failed to add CPD for %q: %w", node, err)
		}
	}
	return nil
}

// GetParameters estimates and returns the CPD for a single node.
func (mle *MaximumLikelihoodEstimator) GetParameters(node string) (*factors.TabularCPD, error) {
	if mle.bn == nil {
		return nil, fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if mle.data == nil {
		return nil, fmt.Errorf("learning: data is nil")
	}

	// Check that the node exists.
	found := false
	for _, n := range mle.bn.Nodes() {
		if n == node {
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("learning: node %q not found in BayesianNetwork", node)
	}

	// Validate required columns.
	dataColumns := make(map[string]bool)
	for _, c := range mle.data.Columns() {
		dataColumns[c] = true
	}
	if !dataColumns[node] {
		return nil, fmt.Errorf("learning: data is missing column for node %q", node)
	}
	parents := mle.bn.Parents(node)
	for _, p := range parents {
		if !dataColumns[p] {
			return nil, fmt.Errorf("learning: data is missing column for parent %q", p)
		}
	}

	return mle.estimateNode(node)
}

// estimateNode computes the MLE CPD for a single node.
func (mle *MaximumLikelihoodEstimator) estimateNode(node string) (*factors.TabularCPD, error) {
	parents := mle.bn.Parents(node) // sorted
	nRows := mle.data.Len()

	// Get column data as int slices.
	nodeVals := mle.data.Column(node).Int()

	parentVals := make([][]int, len(parents))
	for i, p := range parents {
		parentVals[i] = mle.data.Column(p).Int()
	}

	// Determine cardinalities from the data.
	nodeCard := maxVal(nodeVals) + 1
	if nodeCard < 1 {
		nodeCard = 1
	}

	parentCards := make([]int, len(parents))
	for i := range parents {
		parentCards[i] = maxVal(parentVals[i]) + 1
		if parentCards[i] < 1 {
			parentCards[i] = 1
		}
	}

	// Number of parent configurations.
	numParentConfigs := 1
	for _, pc := range parentCards {
		numParentConfigs *= pc
	}

	// Count occurrences: counts[childState][parentConfig].
	counts := make([][]float64, nodeCard)
	for i := range counts {
		counts[i] = make([]float64, numParentConfigs)
	}

	for row := 0; row < nRows; row++ {
		childState := nodeVals[row]
		if childState < 0 || childState >= nodeCard {
			continue
		}

		parentConfig := 0
		valid := true
		for i := range parents {
			pv := parentVals[i][row]
			if pv < 0 || pv >= parentCards[i] {
				valid = false
				break
			}
			parentConfig = parentConfig*parentCards[i] + pv
		}
		if !valid {
			continue
		}

		counts[childState][parentConfig]++
	}

	// Normalize each column (parent configuration) to sum to 1.
	// If a column has zero total counts, use uniform distribution.
	for pc := 0; pc < numParentConfigs; pc++ {
		colSum := 0.0
		for cs := 0; cs < nodeCard; cs++ {
			colSum += counts[cs][pc]
		}
		if colSum == 0 {
			// Uniform distribution for unobserved parent configurations.
			uniform := 1.0 / float64(nodeCard)
			for cs := 0; cs < nodeCard; cs++ {
				counts[cs][pc] = uniform
			}
		} else {
			for cs := 0; cs < nodeCard; cs++ {
				counts[cs][pc] /= colSum
			}
		}
	}

	return factors.NewTabularCPD(node, nodeCard, counts, parents, parentCards)
}

// EstimatePotentials estimates unnormalized factors (potentials) for a Markov
// network. For each edge in the network's underlying graph structure (derived
// from the BayesianNetwork's edges), it counts joint occurrences of variable
// combinations from data and returns them as unnormalized DiscreteFactor values.
//
// This method treats each edge in the BN as defining a pairwise factor scope.
// For nodes with no edges, a unary factor is created.
func (mle *MaximumLikelihoodEstimator) EstimatePotentials() ([]*factors.DiscreteFactor, error) {
	if mle.bn == nil {
		return nil, fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if mle.data == nil {
		return nil, fmt.Errorf("learning: data is nil")
	}

	nodes := mle.bn.Nodes()
	if len(nodes) == 0 {
		return nil, fmt.Errorf("learning: BayesianNetwork has no nodes")
	}

	dataColumns := make(map[string]bool)
	for _, c := range mle.data.Columns() {
		dataColumns[c] = true
	}
	for _, node := range nodes {
		if !dataColumns[node] {
			return nil, fmt.Errorf("learning: data is missing column for node %q", node)
		}
	}

	nRows := mle.data.Len()

	// Determine cardinalities from data.
	cardMap := make(map[string]int)
	colVals := make(map[string][]int)
	for _, node := range nodes {
		vals := mle.data.Column(node).Int()
		colVals[node] = vals
		cardMap[node] = maxVal(vals) + 1
		if cardMap[node] < 1 {
			cardMap[node] = 1
		}
	}

	// Collect edges as pairwise factor scopes.
	edges := mle.bn.Edges()
	var result []*factors.DiscreteFactor

	// Track which nodes are covered by pairwise factors.
	covered := make(map[string]bool)

	for _, e := range edges {
		u, v := e[0], e[1]
		covered[u] = true
		covered[v] = true

		vars := []string{u, v}
		card := []int{cardMap[u], cardMap[v]}
		totalSize := card[0] * card[1]
		counts := make([]float64, totalSize)

		uVals := colVals[u]
		vVals := colVals[v]

		for row := 0; row < nRows; row++ {
			ui := uVals[row]
			vi := vVals[row]
			if ui >= 0 && ui < card[0] && vi >= 0 && vi < card[1] {
				counts[ui*card[1]+vi]++
			}
		}

		f, err := factors.NewDiscreteFactor(vars, card, counts)
		if err != nil {
			return nil, fmt.Errorf("learning: failed to create factor for (%q, %q): %w", u, v, err)
		}
		result = append(result, f)
	}

	// Add unary factors for isolated nodes.
	for _, node := range nodes {
		if !covered[node] {
			card := cardMap[node]
			counts := make([]float64, card)
			vals := colVals[node]
			for row := 0; row < nRows; row++ {
				vi := vals[row]
				if vi >= 0 && vi < card {
					counts[vi]++
				}
			}
			f, err := factors.NewDiscreteFactor([]string{node}, []int{card}, counts)
			if err != nil {
				return nil, fmt.Errorf("learning: failed to create unary factor for %q: %w", node, err)
			}
			result = append(result, f)
		}
	}

	return result, nil
}

// maxVal returns the maximum value in a slice of ints, or -1 if empty.
func maxVal(vals []int) int {
	if len(vals) == 0 {
		return -1
	}
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}
